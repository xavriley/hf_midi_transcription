"""
Script to upload the MIDI Transcription Model to Hugging Face Hub
"""

import os
import json
from pathlib import Path
from hf_midi_transcription import MidiTranscriptionModel

def main():
    print("Uploading MIDI Transcription Models to Hugging Face Hub")
    print("=" * 60)
    
    # Load instrument configuration
    config_path = Path("instruments.json")
    if not config_path.exists():
        print("✗ Error: instruments.json not found")
        print("Please ensure the instruments configuration file is in the current directory")
        return
    
    with open(config_path, 'r') as f:
        instruments_config = json.load(f)
    
    # Check if checkpoint files exist
    missing_files = []
    for instrument, config in instruments_config.items():
        checkpoint_file = config["checkpoint_file"]
        if not os.path.exists(checkpoint_file):
            missing_files.append(f"{instrument}: {checkpoint_file}")
    
    if missing_files:
        print("✗ Error: Missing checkpoint files:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("Please ensure all checkpoint files are in the same directory as this script")
        print("Note: These large model files should be uploaded to HF Hub")
        return
    
    try:
        # Upload models for each instrument
        repo_name = "xavriley/midi-transcription-models"
        
        for i, (instrument, config) in enumerate(instruments_config.items(), 1):
            print(f"{i}. Processing {instrument} model...")
            
            # Initialize model for this instrument
            model = MidiTranscriptionModel(
                instrument=instrument,
                checkpoint_path=config["checkpoint_file"],
                device="cpu",  # Use CPU for upload to avoid memory issues
                batch_size=8
            )
            
            # Save locally first (optional, for backup)
            local_dir = f"midi-transcription-model-{instrument}-local"
            print(f"   Saving {instrument} model locally...")
            model.save_pretrained(local_dir)
            print(f"   ✓ Local save completed for {instrument}")
        
        # Upload the first model to create the repository structure
        print(f"\n{len(instruments_config) + 1}. Uploading to Hugging Face Hub...")
        print(f"   Repository: {repo_name}")
        
        # Use the first instrument model for the initial upload
        first_instrument = list(instruments_config.keys())[0]
        first_config = instruments_config[first_instrument]
        
        model = MidiTranscriptionModel(
            instrument=first_instrument,
            checkpoint_path=first_config["checkpoint_file"],
            device="cpu",
            batch_size=8
        )
        
        model.push_to_hub(
            repo_name,
            commit_message="Initial upload of multi-instrument MIDI transcription models",
            private=False,
        )
        
        print("✓ Upload completed successfully!")
        print("\n🎉 Your models are now available on Hugging Face Hub!")
        print(f"🔗 https://huggingface.co/{repo_name}")
        
        print("\nUsers can now use your models with:")
        print("```python")
        print("from hf_midi_transcription import MidiTranscriptionModel")
        print(f"model = MidiTranscriptionModel.from_pretrained('{repo_name}', instrument='saxophone')")
        print("model.transcribe('audio.wav', 'output.mid')")
        print("```")
        
        print("\nAvailable instruments:")
        for instrument in instruments_config.keys():
            print(f"  - {instrument}")
        
        print("\nCLI usage:")
        print("```bash")
        print("midi_transcription audio.wav output.mid --instrument saxophone")
        print("midi_transcription audio.wav output.mid --instrument bass")
        print("```")
        
    except Exception as e:
        print(f"✗ Upload failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify the repository name is available")
        print("4. Ensure you have write permissions to the repository")

if __name__ == "__main__":
    main()
