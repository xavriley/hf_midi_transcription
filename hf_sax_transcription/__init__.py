"""
Saxophone Transcription Model for Hugging Face Hub
"""

from .model import SaxophoneTranscriptionModel
from .cli import main as cli_main

__version__ = "0.1.0"
__all__ = ["SaxophoneTranscriptionModel", "cli_main"]
