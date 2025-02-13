import logging
import tkinter as tk
from transcription_manager import TranscriptionManager
from gui_app import Application
from logging_config import setup_logging

setup_logging(level=logging.DEBUG, log_file='app.log')

if __name__ == "__main__":
    manager = TranscriptionManager(batch_size=4, record_seconds=6, overlap_seconds=1.5)
    app = Application(manager)
    app.mainloop()