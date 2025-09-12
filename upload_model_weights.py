"""
Script to upload the large model checkpoint file to Hugging Face Hub
This should be run separately from the main model upload.
"""

import os
from huggingface_hub import HfApi, upload_file
from pathlib import Path

def upload_model_checkpoint():
    """Upload the model checkpoint file to Hugging Face Hub."""
    
    print("Uploading Model Checkpoint to Hugging Face Hub")
    print("=" * 50)
    
    # Configuration
    repo_id = "xavriley/sax-transcription-model"
    checkpoint_file = "filosax_25k.pth"
    
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_file):
        print(f"✗ Error: {checkpoint_file} not found in current directory")
        print("Please ensure the checkpoint file is in the same directory as this script")
        return False
    
    # Get file size
    file_size = os.path.getsize(checkpoint_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📁 File: {checkpoint_file} ({file_size_mb:.1f} MB)")
    
    try:
        print(f"🚀 Uploading to repository: {repo_id}")
        print("⏳ This may take a few minutes due to file size...")
        
        # Upload the file
        upload_file(
            path_or_fileobj=checkpoint_file,
            path_in_repo=checkpoint_file,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload model checkpoint: {checkpoint_file}"
        )
        
        print("✅ Model checkpoint uploaded successfully!")
        print(f"🔗 Available at: https://huggingface.co/{repo_id}/blob/main/{checkpoint_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify you have write access to the repository")
        print("4. The repository should exist (create it first if needed)")
        
        return False

def create_repository_if_needed():
    """Create the repository if it doesn't exist."""
    repo_id = "xavriley/sax-transcription-model"
    
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
    print("Saxophone Transcription Model - Checkpoint Upload")
    print("=" * 55)
    
    # Step 1: Create repository if needed
    print("Step 1: Checking/creating repository...")
    if not create_repository_if_needed():
        return
    
    print()
    
    # Step 2: Upload checkpoint
    print("Step 2: Uploading model checkpoint...")
    if upload_model_checkpoint():
        print("\n🎉 Upload completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python upload_to_hf.py' to upload the model code")
        print("2. Your users can now use the model without needing the local file")
        print("3. The model will be automatically downloaded on first use")
    else:
        print("\n❌ Upload failed. Please check the errors above.")

if __name__ == "__main__":
    main()
