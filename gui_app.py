import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import queue
from audio_utils import remove_overlap

class Application(tk.Tk):
    def __init__(self, transcription_manager):
        super().__init__()
        self.transcription_manager = transcription_manager
        self.title("Real-time Transcription and Translation")
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
        title_label = tk.Label(self, text="Real-time Transcription and Translation",
                               font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=10)

        control_frame = tk.Frame(self, bg="#f0f0f0")
        control_frame.pack(pady=5)

        self.start_button = tk.Button(control_frame, text="Start", command=self.start_transcription,
                                      font=("Helvetica", 12), width=12, bg="#4CAF50", fg="white")
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_transcription,
                                     font=("Helvetica", 12), width=12, bg="#F44336", fg="white", state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10)

        language_label = tk.Label(control_frame, text="Output Language:",
                                  font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
        language_label.grid(row=0, column=2, padx=5)

        self.language_var = tk.StringVar(value=self.transcription_manager.target_language)
        options = ["English", "Spanish", "French", "German"]
        language_menu = tk.OptionMenu(control_frame, self.language_var, *options, command=self.change_language)
        language_menu.config(font=("Helvetica", 12), bg="#ddd")
        language_menu.grid(row=0, column=3, padx=5)

        self.clear_button = tk.Button(control_frame, text="Clear", command=self.clear_text,
                                      font=("Helvetica", 12), width=12, bg="#2196F3", fg="white")
        self.clear_button.grid(row=0, column=4, padx=10)

        self.status_label = tk.Label(self, text="Inactive | 0 words", font=("Helvetica", 12),
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
        status = f"Recording... | {word_count} words" if recording else f"Inactive | {word_count} words"
        self.status_label.config(text=status)

    def change_language(self, new_language):
        self.transcription_manager.set_target_language(new_language)
        self.status_label.config(text=f"Language updated to: {new_language}")

    def start_transcription(self):
        self.transcription_manager.start()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Recording... | 0 words")
        self.word_counter_running = True
        threading.Thread(target=self.word_counter, daemon=True).start()

    def stop_transcription(self):
        self.transcription_manager.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Inactive | 0 words")
        self.word_counter_running = False

    def clear_text(self):
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
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