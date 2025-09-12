"""
Command Line Interface for Saxophone Transcription Model
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .model import SaxophoneTranscriptionModel


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="sax_transcription",
        description="Transcribe saxophone audio to MIDI using neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sax_transcription audio.wav output.mid
  sax_transcription recording.mp3 transcription.mid --device cuda
  sax_transcription input.wav output.mid --batch-size 16 --checkpoint custom_model.pth
  sax_transcription audio.wav output.mid --model-id username/custom-sax-model
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to input audio file (WAV, MP3, etc.)"
    )
    
    parser.add_argument(
        "midi_output_path", 
        type=str,
        help="Path for output MIDI file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (default: auto - chooses cuda if available, else cpu)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing audio segments (default: 8)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="filosax_25k.pth",
        help="Path to model checkpoint file (default: filosax_25k.pth)"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model ID to load pretrained model from (e.g., username/model-name)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check if input file exists
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Input audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check if output directory exists and is writable
    output_path = Path(args.midi_output_path)
    output_dir = output_path.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.verbose:
                print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error: Cannot create output directory {output_dir}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # If using local checkpoint, check if it exists
    if not args.model_id and args.checkpoint != "filosax_25k.pth":
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_args(args)
    
    if args.verbose:
        print("Saxophone Transcription CLI")
        print("=" * 40)
        print(f"Input audio: {args.audio_path}")
        print(f"Output MIDI: {args.midi_output_path}")
        print(f"Device: {args.device or 'auto'}")
        print(f"Batch size: {args.batch_size}")
        if args.model_id:
            print(f"Model ID: {args.model_id}")
        else:
            print(f"Checkpoint: {args.checkpoint}")
        print()
    
    try:
        # Initialize model
        if args.verbose:
            print("Loading model...")
        
        # Handle device selection
        device = args.device
        if device == "auto" or device is None:
            device = None  # Let the model choose automatically
        
        if args.model_id:
            # Load from Hugging Face Hub
            model = SaxophoneTranscriptionModel.from_pretrained(
                args.model_id,
                device=device,
                batch_size=args.batch_size
            )
        else:
            # Load from local checkpoint
            model = SaxophoneTranscriptionModel(
                device=device,
                sax_checkpoint_path=args.checkpoint,
                batch_size=args.batch_size
            )
        
        if args.verbose:
            print("✓ Model loaded successfully")
            print("Starting transcription...")
        
        # Perform transcription
        output_path = model.transcribe(args.audio_path, args.midi_output_path)
        
        # Success message
        print(f"✓ Transcription completed successfully!")
        print(f"Output saved to: {output_path}")
        
        if args.verbose:
            output_size = Path(output_path).stat().st_size
            print(f"Output file size: {output_size} bytes")
        
    except KeyboardInterrupt:
        print("\n⚠ Transcription interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error during transcription: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
