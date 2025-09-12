"""
Script to upload the Saxophone Transcription Model to Hugging Face Hub
"""

import os
from hf_sax_transcription import SaxophoneTranscriptionModel

def main():
    print("Uploading Saxophone Transcription Model to Hugging Face Hub")
    print("=" * 60)
    
    # Check if checkpoint file exists
    checkpoint_file = "filosax_25k.pth"
    if not os.path.exists(checkpoint_file):
        print(f"✗ Error: {checkpoint_file} not found in current directory")
        print("Please ensure the checkpoint file is in the same directory as this script")
        print("Note: This large model file should be uploaded separately to HF Hub")
        return
    
    try:
        # Initialize model
        print("1. Initializing model...")
        model = SaxophoneTranscriptionModel(
            sax_checkpoint_path="filosax_25k.pth",
            device="cpu",  # Use CPU for upload to avoid memory issues
            batch_size=8
        )
        
        # Save locally first (optional, for backup)
        print("2. Saving model locally...")
        model.save_pretrained("sax-transcription-model-local")
        print("✓ Local save completed")
        
        # Upload to Hub
        print("3. Uploading to Hugging Face Hub...")
        print("   Repository: xavriley/sax-transcription-model")
        
        model.push_to_hub(
            "xavriley/sax-transcription-model",
            # Optional: Add commit message
            commit_message="Initial upload of saxophone transcription model",
            # Optional: Make it private initially
            private=False,
        )
        
        print("✓ Upload completed successfully!")
        print("\n🎉 Your model is now available on Hugging Face Hub!")
        print("🔗 https://huggingface.co/xavriley/sax-transcription-model")
        
        print("\nUsers can now use your model with:")
        print("```python")
        print("from hf_sax_transcription import SaxophoneTranscriptionModel")
        print("model = SaxophoneTranscriptionModel.from_pretrained('xavriley/sax-transcription-model')")
        print("model.transcribe('audio.wav', 'output.mid')")
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
