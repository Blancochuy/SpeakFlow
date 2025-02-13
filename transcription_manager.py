import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import numpy as np
import pyaudiowpatch as pyaudio
import noisereduce as nr
import logging
import threading
import queue
import time
import wave
from langdetect import detect, DetectorFactory
from io import BytesIO
from audio_utils import remove_repeated_words  # Import shared utility
from audio_utils import remove_overlap

# To repeat deterministic language detection across invocations
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

MODEL_ID = "openai/whisper-large-v3-turbo"
TRANSLATION_MODELS = {
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "English": None
}
TARGET_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

class TranscriptionManager:
    def __init__(self, batch_size=4, record_seconds=6, overlap_seconds=1.5):
        self.audio_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.new_result_event = threading.Event()
        self.running = False
        self.batch_size = batch_size
        self.record_seconds = record_seconds
        self.overlap_seconds = overlap_seconds

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True

        logger.info("Cargando modelo y procesador para transcripción...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        ).to(self.device)

        if self.device != "cpu" and hasattr(torch, "compile"):
            logger.info("Compilando modelo para reducir latencia...")
            self.model = torch.compile(self.model)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        logger.info("Modelo de transcripción cargado correctamente.")

        self.target_language = "Spanish"
        self._init_translation_pipeline()
        self.translation_cache = {}

    def _init_translation_pipeline(self):
        if self.target_language != "English":
            lang_code = TARGET_CODES[self.target_language]
            pipeline_task = f"translation_en_to_{lang_code}"
            self.translation_pipe = pipeline(
                pipeline_task,
                model=TRANSLATION_MODELS[self.target_language],
                device=self.device,
                torch_dtype=self.torch_dtype
            )
            logger.info(f"Pipeline de traducción para {self.target_language} cargado correctamente.")
        else:
            self.translation_pipe = None
            logger.info("Traducción desactivada (salida en English).")

    def set_target_language(self, new_language):
        self.target_language = new_language
        self._init_translation_pipeline()

    def record_audio(self):
        while self.running:
            try:
                with pyaudio.PyAudio() as p:
                    try:
                        default_speakers = p.get_default_wasapi_loopback()
                    except Exception as e:
                        logger.error(f"Dispositivo de audio no encontrado: {e}")
                        time.sleep(2)
                        continue

                    RATE = int(default_speakers["defaultSampleRate"])
                    CHANNELS = default_speakers["maxInputChannels"]
                    logger.info(f"Grabando desde: {default_speakers['name']} (RATE: {RATE}, CHANNELS: {CHANNELS})")

                    with p.open(
                        format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=default_speakers["index"],
                    ) as stream:
                        logger.info("Grabando audio en loopback...")
                        frames = []
                        num_frames = int(RATE / 1024 * self.record_seconds)
                        for _ in range(num_frames):
                            if not self.running:
                                break
                            data = stream.read(1024)
                            frames.append(data)
                        audio_bytes = b"".join(frames)
                        self.audio_queue.put((audio_bytes, RATE, CHANNELS))
                    time.sleep(self.overlap_seconds)
            except Exception as e:
                logger.error(f"Error durante la grabación: {e}. Reintentando en 2 segundos...")
                time.sleep(2)

    def process_audio(self, audio_tuple):
        try:
            audio_bytes, sample_rate, channels = audio_tuple
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            wav_buffer.seek(0)
            waveform, sr = torchaudio.load(wav_buffer)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            target_sr = 16000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            audio_np = waveform.cpu().numpy()
            audio_np = nr.reduce_noise(y=audio_np, sr=target_sr)
            if np.abs(audio_np).max() < 0.01:
                logger.warning("Segmento de audio descartado por bajo nivel de sonido.")
                return None
            return audio_np
        except Exception as e:
            logger.error(f"Error procesando el audio: {e}")
            return None

    def transcription_worker(self):
        while self.running or not self.audio_queue.empty():
            batch_audio = []
            try:
                audio_tuple = self.audio_queue.get(timeout=1)
                batch_audio.append(audio_tuple)
            except queue.Empty:
                continue

            for _ in range(self.batch_size - 1):
                try:
                    audio_tuple = self.audio_queue.get(timeout=0.1)
                    batch_audio.append(audio_tuple)
                except queue.Empty:
                    break

            audio_list = [self.process_audio(at) for at in batch_audio]
            audio_list = [a for a in audio_list if a is not None]

            if audio_list:
                try:
                    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.float16) if self.device != "cpu" else torch.no_grad()):
                        gen_kwargs = {"language": "en", "task": "transcribe"} if self.target_language == "English" else {}
                        results = self.pipe(
                            audio_list,
                            return_timestamps=True,
                            return_language=True,
                            generate_kwargs=gen_kwargs
                        )
                    if isinstance(results, dict):
                        results = [results]

                    texts_to_translate = []
                    indices_to_translate = []
                    final_texts = [None] * len(results)

                    for i, res in enumerate(results):
                        text = res.get("text", "").strip()
                        if not text:
                            logger.warning("Transcripción vacía.")
                            continue
                        if self.target_language == "English":
                            final_texts[i] = remove_repeated_words(text)
                        else:
                            try:
                                detected_lang = detect(text)
                            except Exception:
                                detected_lang = None
                            if detected_lang is not None and detected_lang == TARGET_CODES[self.target_language]:
                                final_texts[i] = remove_repeated_words(text)
                            else:
                                if text in self.translation_cache:
                                    final_texts[i] = self.translation_cache[text]
                                else:
                                    texts_to_translate.append(text)
                                    indices_to_translate.append(i)
                    if texts_to_translate and self.translation_pipe is not None:
                        translations = self.translation_pipe(texts_to_translate)
                        for j, tr in enumerate(translations):
                            trans_text = tr.get("translation_text", "").strip()
                            trans_text = remove_repeated_words(trans_text)
                            self.translation_cache[texts_to_translate[j]] = trans_text
                            final_texts[indices_to_translate[j]] = trans_text
                    for text_final in final_texts:
                        if text_final:
                            logger.info(f"Transcripción final: {text_final}")
                            self.results_queue.put(text_final)
                            self.new_result_event.set()
                except Exception as e:
                    logger.error(f"Error en procesamiento batch: {e}")

    def start(self):
        self.running = True
        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        self.record_thread.start()
        self.transcription_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join()