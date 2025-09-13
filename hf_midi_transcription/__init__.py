"""
MIDI Transcription Model for Hugging Face Hub
"""

from .model import MidiTranscriptionModel, SaxophoneTranscriptionModel
from .cli import main as cli_main

__version__ = "0.1.0"
__all__ = ["MidiTranscriptionModel", "SaxophoneTranscriptionModel", "cli_main"]
