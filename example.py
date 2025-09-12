"""
Example usage of the Saxophone Transcription Model
"""

from hf_sax_transcription import SaxophoneTranscriptionModel

def main():
    print("Saxophone Transcription Model Example")
    print("=====================================")
    
    print("Method 1: Command Line Interface (Recommended)")
    print("----------------------------------------------")
    print("The easiest way to use the model:")
    print("$ sax_transcription test.wav output.mid")
    print("$ sax_transcription audio.wav output.mid --device cuda --verbose")
    print("$ sax_transcription audio.wav output.mid --model-id username/model")
    print()
    
    print("Method 2: Python API")
    print("--------------------")
    
    # Initialize model with local checkpoint
    print("Initializing model...")
    model = SaxophoneTranscriptionModel(
        sax_checkpoint_path="filosax_25k.pth",
        device=None,  # Will automatically choose cuda or cpu
        batch_size=8
    )
    
    # Example transcription
    print("Transcribing test.wav to test_output.mid...")
    try:
        output_path = model.transcribe("test.wav", "test_output.mid")
        print(f"Transcription completed! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error during transcription: {e}")
    
    print("\nTo save the model for Hugging Face Hub:")
    print("model.save_pretrained('sax-transcription-model')")
    print("model.push_to_hub('your-username/sax-transcription-model')")

if __name__ == "__main__":
    main()
