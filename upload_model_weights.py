"""
Script to upload the large model checkpoint files to Hugging Face Hub
This should be run separately from the main model upload.
"""

import os
import json
from huggingface_hub import HfApi, upload_file
from pathlib import Path

def upload_model_checkpoint(repo_id: str = "xavriley/midi-transcription-models",
                            checkpoint_file: str = None):
    """Upload model checkpoint files to Hugging Face Hub."""
    
    print("Uploading Model Checkpoints to Hugging Face Hub")
    print("=" * 50)
    
    # Load instrument configuration
    config_path = Path("instruments.json")
    if not config_path.exists():
        print("✗ Error: instruments.json not found")
        print("Please ensure the instruments configuration file is in the current directory")
        return False
    
    with open(config_path, 'r') as f:
        instruments_config = json.load(f)
    
    # If specific checkpoint file provided, upload only that one
    if checkpoint_file:
        if not os.path.exists(checkpoint_file):
            print(f"✗ Error: {checkpoint_file} not found in current directory")
            print("Please ensure the checkpoint file is in the same directory as this script")
            return False
        files_to_upload = [checkpoint_file]
    else:
        # Upload all checkpoint files from config
        files_to_upload = []
        for instrument, config in instruments_config.items():
            checkpoint = config["checkpoint_file"]
            if os.path.exists(checkpoint):
                files_to_upload.append(checkpoint)
            else:
                print(f"⚠ Warning: {checkpoint} not found, skipping {instrument}")
        
        if not files_to_upload:
            print("✗ Error: No checkpoint files found")
            return False
    
    # Upload each file
    for i, file_path in enumerate(files_to_upload, 1):
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"📁 File {i}/{len(files_to_upload)}: {file_path} ({file_size_mb:.1f} MB)")
    
    try:
        print(f"🚀 Uploading to repository: {repo_id}")
        print("⏳ This may take a few minutes due to file size...")
        
        # Upload each file
        for file_path in files_to_upload:
            print(f"   Uploading {file_path}...")
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload model checkpoint: {file_path}"
            )
            print(f"   ✓ {file_path} uploaded successfully!")
        
        print("✅ All model checkpoints uploaded successfully!")
        print(f"🔗 Available at: https://huggingface.co/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify you have write access to the repository")
        print("4. The repository should exist (create it first if needed)")
        
        return False

def create_repository_if_needed(repo_id: str = "xavriley/midi-transcription-models"):
    """Create the repository if it doesn't exist."""
    try:
        api = HfApi()
        
        # Check if repo exists
        try:
            api.repo_info(repo_id=repo_id)
            print(f"✓ Repository {repo_id} already exists")
            return True
        except:
            # Repository doesn't exist, create it
            print(f"Creating repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,  # Make it public
                exist_ok=True
            )
            print(f"✓ Repository {repo_id} created successfully")
            return True
            
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False

def main():
    print("MIDI Transcription Models - Checkpoint Upload")
    print("=" * 55)
    
    # Step 1: Create repository if needed
    print("Step 1: Checking/creating repository...")
    if not create_repository_if_needed():
        return
    
    print()
    
    # Step 2: Upload all checkpoints
    print("Step 2: Uploading model checkpoints...")
    if upload_model_checkpoint():
        print("\n🎉 Upload completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python upload_to_hf.py' to upload the model code")
        print("2. Your users can now use the models without needing local files")
        print("3. The models will be automatically downloaded on first use")
        print("\nAvailable instruments:")
        
        # Load and display available instruments
        try:
            with open("instruments.json", 'r') as f:
                instruments_config = json.load(f)
            for instrument in instruments_config.keys():
                print(f"  - {instrument}")
        except:
            print("  - saxophone, bass, guitar, piano")
    else:
        print("\n❌ Upload failed. Please check the errors above.")

if __name__ == "__main__":
    main()
