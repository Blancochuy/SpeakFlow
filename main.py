import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import numpy as np
import pyaudiowpatch as pyaudio
import noisereduce as nr  # Reducción de ruido
import logging
import threading
import queue
import time
import tkinter as tk
from tkinter import scrolledtext
import re
from io import BytesIO
import wave
from langdetect import detect, DetectorFactory

# Para obtener resultados consistentes en langdetect
DetectorFactory.seed = 0

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parámetros globales
MODEL_ID = "openai/whisper-large-v3-turbo"

# Diccionario para modelos de traducción
TRANSLATION_MODELS = {
    "Español": "Helsinki-NLP/opus-mt-en-es",
    "Français": "Helsinki-NLP/opus-mt-en-fr",
    "Deutsch": "Helsinki-NLP/opus-mt-en-de",
    "English": None  # Sin traducción, salida en English
}

# Mapeo de idiomas para comparar (usando códigos ISO 639-1)
TARGET_CODES = {"Español": "es", "Français": "fr", "Deutsch": "de", "English": "en"}

# ============================================================
# Funciones de post-procesamiento para deduplicar texto
# ============================================================
def remove_repeated_words(text):
    pattern = re.compile(r'\b(\S+)(?:\s+\1\b)+', flags=re.IGNORECASE)
    return pattern.sub(r'\1', text)

def remove_overlap(prev_text, new_text):
    max_overlap = min(len(prev_text), len(new_text))
    for i in range(max_overlap, 0, -1):
        if prev_text[-i:] == new_text[:i]:
            return new_text[i:]
    return new_text

# ============================================================
# Clase que administra la grabación y transcripción en segundo plano
# ============================================================
class TranscriptionManager:
    def __init__(self, batch_size=4, record_seconds=6, overlap_seconds=1.5):
        self.audio_queue = queue.Queue()
        self.results_queue = queue.Queue()
        # Evento para notificar nuevos resultados
        self.new_result_event = threading.Event()
        self.running = False
        self.batch_size = batch_size
        self.record_seconds = record_seconds
        self.overlap_seconds = overlap_seconds

        # Configuración de dispositivo y dtype
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Optimización GPU: activar benchmark
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True

        # Cargar modelo y procesador para transcripción
        logger.info("Cargando modelo y procesador para transcripción...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        ).to(self.device)

        # Uso de torch.compile (o TorchScript) para mejorar la velocidad en GPU
        if self.device != "cpu" and hasattr(torch, "compile"):
            logger.info("Compilando modelo para reducir latencia...")
            self.model = torch.compile(self.model)

        # Pipeline para transcripción
        # Si se quiere salida en English se forzará; en otros casos se deja que se auto-detecte.
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        logger.info("Modelo de transcripción cargado correctamente.")

        # Idioma de salida por defecto
        self.target_language = "Español"
        self._init_translation_pipeline()

        # Caché para traducciones
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
        """Captura audio desde el dispositivo de loopback con manejo de errores y reconexión."""
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
        """Procesa el audio: crea un WAV en memoria, lo carga con torchaudio, lo convierte a mono y resamplea a 16 kHz."""
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
                sr = target_sr
            audio_np = waveform.cpu().numpy()
            audio_np = nr.reduce_noise(y=audio_np, sr=sr)
            if np.abs(audio_np).max() < 0.01:
                logger.warning("Segmento de audio descartado por bajo nivel de sonido.")
                return None
            return audio_np
        except Exception as e:
            logger.error(f"Error procesando el audio: {e}")
            return None

    def transcription_worker(self):
        """
        Procesa batches de audio, transcribe y luego:
         - Si el idioma detectado (usando langdetect) coincide con el idioma de salida, usa el texto tal cual.
         - De lo contrario, traduce (en batch y usando caché) el texto.
        Finalmente, notifica a la GUI mediante el evento.
        """
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
                    # Usar autocast para mayor eficiencia en GPU
                    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.float16) if self.device != "cpu" else torch.no_grad()):
                        # Si salida es English se fuerza; de lo contrario se deja auto-detección
                        if self.target_language == "English":
                            gen_kwargs = {"language": "en", "task": "transcribe"}
                        else:
                            gen_kwargs = {}
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
                            # Usar langdetect para determinar el idioma del texto
                            try:
                                detected_lang = detect(text)
                            except Exception:
                                detected_lang = None
                            if detected_lang is not None and detected_lang == TARGET_CODES[self.target_language]:
                                final_texts[i] = remove_repeated_words(text)
                            else:
                                # Se requiere traducción; usar caché si ya existe
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

# ============================================================
# Interfaz Gráfica (GUI) con Tkinter – Mejor UX y animación de tecleo
# ============================================================
class Application(tk.Tk):
    def __init__(self, transcription_manager):
        super().__init__()
        self.transcription_manager = transcription_manager
        self.title("Transcripción y Traducción en Tiempo Real")
        self.geometry("800x600")
        self.configure(bg="#f0f0f0")
        self.last_transcription = ""
        self.typing_queue = []
        self.typing_in_progress = False
        self.animation_paused = False
        self.create_widgets()
        self.bind_events()
        self.poll_results()

    def create_widgets(self):
        title_label = tk.Label(self, text="Transcripción y Traducción en Tiempo Real",
                               font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=10)

        control_frame = tk.Frame(self, bg="#f0f0f0")
        control_frame.pack(pady=5)

        self.start_button = tk.Button(control_frame, text="Iniciar", command=self.start_transcription,
                                      font=("Helvetica", 12), width=12, bg="#4CAF50", fg="white")
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(control_frame, text="Detener", command=self.stop_transcription,
                                     font=("Helvetica", 12), width=12, bg="#F44336", fg="white", state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10)

        idioma_label = tk.Label(control_frame, text="Idioma de salida:",
                                font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
        idioma_label.grid(row=0, column=2, padx=5)

        self.language_var = tk.StringVar(value=self.transcription_manager.target_language)
        opciones = ["English", "Español", "Français", "Deutsch"]
        idioma_menu = tk.OptionMenu(control_frame, self.language_var, *opciones, command=self.change_language)
        idioma_menu.config(font=("Helvetica", 12), bg="#ddd")
        idioma_menu.grid(row=0, column=3, padx=5)

        self.clear_button = tk.Button(control_frame, text="Limpiar", command=self.clear_text,
                                      font=("Helvetica", 12), width=12, bg="#2196F3", fg="white")
        self.clear_button.grid(row=0, column=4, padx=10)

        self.status_label = tk.Label(self, text="🔘 Inactivo | 0 palabras", font=("Helvetica", 12),
                                     bg="#f0f0f0", fg="#555")
        self.status_label.pack(pady=5)

        self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=90, height=25,
                                                   font=("Helvetica", 12))
        self.text_area.pack(pady=10)
        self.text_area.configure(state=tk.DISABLED)

    def bind_events(self):
        self.text_area.bind("<FocusIn>", lambda e: self.pause_animation())
        self.text_area.bind("<FocusOut>", lambda e: self.resume_animation())

    def pause_animation(self):
        self.animation_paused = True

    def resume_animation(self):
        self.animation_paused = False

    def update_status(self, recording=False):
        text = self.text_area.get("1.0", tk.END)
        word_count = len(text.split())
        status = f"🔴 Grabando... | {word_count} palabras" if recording else f"🔘 Inactivo | {word_count} palabras"
        self.status_label.config(text=status)

    def change_language(self, new_language):
        self.transcription_manager.set_target_language(new_language)
        self.status_label.config(text=f"Idioma actualizado a: {new_language}")

    def start_transcription(self):
        self.transcription_manager.start()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="🔴 Grabando... | 0 palabras")
        self.word_counter_running = True
        threading.Thread(target=self.word_counter, daemon=True).start()

    def stop_transcription(self):
        self.transcription_manager.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="🔘 Inactivo | 0 palabras")
        self.word_counter_running = False

    def clear_text(self):
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.configure(state=tk.DISABLED)
        self.last_transcription = ""
        self.typing_queue = []

    def poll_results(self):
        if self.transcription_manager.new_result_event.is_set():
            try:
                while True:
                    new_text = self.transcription_manager.results_queue.get_nowait()
                    if self.last_transcription:
                        new_text = remove_overlap(self.last_transcription, new_text)
                    if new_text.strip():
                        self.typing_queue.append(new_text.strip() + " ")
                        self.last_transcription += " " + new_text
                    # Limitar a las últimas 500 líneas
                    if int(self.text_area.index('end-1c').split('.')[0]) > 500:
                        self.text_area.configure(state=tk.NORMAL)
                        self.text_area.delete("1.0", "100.0")
                        self.text_area.configure(state=tk.DISABLED)
            except queue.Empty:
                pass
            self.transcription_manager.new_result_event.clear()
            self.process_typing_queue()
        self.after(20, self.poll_results)

    def process_typing_queue(self):
        if not self.typing_in_progress and self.typing_queue and not self.animation_paused:
            self.typing_in_progress = True
            text_to_type = self.typing_queue.pop(0)
            delay = 30 if len(text_to_type) > 100 else 50
            self.type_text(text_to_type, 0, delay)

    def type_text(self, text, index, delay):
        if self.animation_paused:
            self.after(100, lambda: self.type_text(text, index, delay))
            return
        if index < len(text):
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert(tk.END, text[index])
            self.text_area.see(tk.END)
            self.text_area.configure(state=tk.DISABLED)
            self.after(delay, lambda: self.type_text(text, index + 1, delay))
        else:
            self.typing_in_progress = False
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert(tk.END, "\n")
            self.text_area.see(tk.END)
            self.text_area.configure(state=tk.DISABLED)
            self.process_typing_queue()

    def word_counter(self):
        while getattr(self, "word_counter_running", False):
            self.update_status(recording=True)
            time.sleep(1)

# ============================================================
# Función main
# ============================================================
if __name__ == "__main__":
    manager = TranscriptionManager(batch_size=4, record_seconds=6, overlap_seconds=1.5)
    app = Application(manager)
    app.mainloop()
