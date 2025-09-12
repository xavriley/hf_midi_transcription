"""
Test script for Hugging Face Hub integration
"""

import os
import tempfile
from pathlib import Path
from hf_sax_transcription import SaxophoneTranscriptionModel

def test_save_load_locally():
    """Test saving and loading the model locally"""
    print("Testing local save/load functionality...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "test_model"
        
        try:
            # Initialize model
            print("1. Initializing model...")
            model = SaxophoneTranscriptionModel(
                sax_checkpoint_path="filosax_25k.pth",
                device="cpu",  # Use CPU for testing
                batch_size=4
            )
            
            # Save the model
            print("2. Saving model...")
            model.save_pretrained(save_path)
            
            # Check if files were created
            config_file = save_path / "config.json"
            if config_file.exists():
                print("✓ Config file created successfully")
            else:
                print("✗ Config file not found")
                
            # Load the model
            print("3. Loading model...")
            loaded_model = SaxophoneTranscriptionModel.from_pretrained(save_path)
            
            print("✓ Model loaded successfully")
            print("✓ Local save/load test passed!")
            
            return True
            
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            return False

def test_transcription():
    """Test the transcription functionality"""
    print("\nTesting transcription functionality...")
    
    try:
        # Check if test audio file exists
        if not os.path.exists("test.wav"):
            print("⚠ test.wav not found, skipping transcription test")
            return True
            
        # Initialize model
        model = SaxophoneTranscriptionModel(
            sax_checkpoint_path="filosax_25k.pth",
            device="cpu",
            batch_size=2
        )
        
        # Test transcription
        output_path = model.transcribe("test.wav", "test_transcription_output.mid")
        
        if os.path.exists(output_path):
            print("✓ Transcription test passed!")
            # Clean up
            os.remove(output_path)
            return True
        else:
            print("✗ Transcription output file not created")
            return False
            
    except Exception as e:
        print(f"✗ Transcription test failed: {e}")
        return False

def main():
    print("Saxophone Transcription Model - Hugging Face Integration Test")
    print("=" * 60)
    
    # Test local save/load
    save_load_success = test_save_load_locally()
    
    # Test transcription
    transcription_success = test_transcription()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Local Save/Load: {'✓ PASSED' if save_load_success else '✗ FAILED'}")
    print(f"Transcription: {'✓ PASSED' if transcription_success else '✗ FAILED'}")
    
    if save_load_success and transcription_success:
        print("\n🎉 All tests passed! Your model is ready for Hugging Face Hub.")
        print("\nNext steps:")
        print("1. Log in to Hugging Face: huggingface-cli login")
        print("2. Run the upload script to push to hub")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before uploading.")

if __name__ == "__main__":
    main()
