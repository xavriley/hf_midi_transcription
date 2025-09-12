# Saxophone Transcription Model

This code transcribes solo, monophonic saxophone from audio to MIDI. If you need to separate saxophone from a mix first please check out UVR or MVSep.com


**Key Features:**
- Audio-to-MIDI transcription for saxophone
- Batch processing support for efficient inference
- GPU acceleration support

## Model Details

- **Model Type:** Audio-to-MIDI transcription
- **Architecture:** Convolutional Recurrent Neural Network (CRNN) with onset/offset/frame/velocity regression
- **Input:** Audio files (WAV, MP3, etc.)
- **Output:** MIDI files
- **Sample Rate:** 16 kHz
- **License:** MIT

## Installation

### Option 1: Install via pip (Recommended)

```bash
pip install hf-sax-transcription
```

### Option 2: Install from source

```bash
git clone https://github.com/xavriley/hf_sax_transcription.git
cd sax_transcription
pip install -e .
```

## Quick Start

### Using the Command Line Interface (CLI)

The simplest way to use the model is through the command line:

```bash
# Basic usage
sax_transcription input_audio.wav output.mid

# With options
sax_transcription recording.mp3 transcription.mid --device cuda --verbose

# Using a custom checkpoint
sax_transcription audio.wav output.mid --checkpoint my_model.pth --batch-size 16

# Using a model from Hugging Face Hub
sax_transcription audio.wav output.mid --model-id username/custom-sax-model
```

**CLI Options:**
- `--device {auto,cpu,cuda}` - Choose compute device (default: auto)
- `--batch-size N` - Set batch size for processing (default: 8)
- `--checkpoint PATH` - Use custom checkpoint file
- `--model-id ID` - Load model from Hugging Face Hub
- `--verbose, -v` - Enable detailed output
- `--help` - Show full help

### Using the Python API

```python
from hf_sax_transcription import SaxophoneTranscriptionModel

# Load the pre-trained model
model = SaxophoneTranscriptionModel.from_pretrained("xavriley/sax-transcription-model")

# Transcribe audio file to MIDI
model.transcribe("saxophone_recording.wav", "output.mid")
```

### Using with audio arrays

```python
import librosa
from hf_sax_transcription import SaxophoneTranscriptionModel

# Load model
model = SaxophoneTranscriptionModel.from_pretrained("xavriley/sax-transcription-model")

# Load audio as array
audio, sr = librosa.load("saxophone_recording.wav", sr=16000)

# Transcribe audio array to MIDI
model.transcribe_audio_array(audio, "output.mid")
```

### Automatic model download

The model will automatically download the required checkpoint file from Hugging Face Hub on first use:

```python
from hf_sax_transcription import SaxophoneTranscriptionModel

# Model will automatically download filosax_25k.pth if not found locally
model = SaxophoneTranscriptionModel(
    device="cuda",  # or "cpu" 
    batch_size=8
)

# Transcribe
model.transcribe("input.wav", "output.mid")
```

### Using local checkpoint files

If you have a local checkpoint file:

```python
from hf_sax_transcription import SaxophoneTranscriptionModel

# Use local checkpoint
model = SaxophoneTranscriptionModel(
    sax_checkpoint_path="path/to/your_model.pth",
    device="cuda",
    batch_size=8
)

# Transcribe
model.transcribe("input.wav", "output.mid")
```

## Model Performance

The model has been trained on a dataset of saxophone recordings and optimized for:
- Monophonic saxophone performance (single note at a time)
- Various saxophone types (soprano, alto, tenor, baritone)
- Different playing styles and dynamics
- Clean and reverberant recording conditions

## Limitations

- Optimized for monophonic saxophone (single notes, not chords)
- Performance may vary with heavily processed or distorted audio
- Best results with clean, well-recorded saxophone audio
- For 

## Model File Management

### Automatic Downloads

The saxophone transcription model (~95MB) is automatically downloaded from Hugging Face Hub when needed:

- **First use**: Model downloads automatically to your HF cache directory
- **Subsequent uses**: Model loads from cache (no re-download)
- **Offline use**: Works if model is already cached
- **Custom models**: Supports loading custom checkpoints from local files or other HF repositories

### Cache Location

Models are cached in your system's Hugging Face cache directory:
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

### Advanced Usage

```python
# Use a different model from Hugging Face Hub
model = SaxophoneTranscriptionModel(
    sax_checkpoint_path="username/custom-sax-model/model.pth"
)

# Force re-download (useful for model updates)
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download(
    repo_id="xavriley/sax-transcription-model",
    filename="filosax_25k.pth",
    force_download=True
)
model = SaxophoneTranscriptionModel(sax_checkpoint_path=checkpoint)
```

## Technical Details

### Model Architecture

The model uses a Convolutional Recurrent Neural Network (CRNN) architecture with:
- **Onset detection:** Identifies when notes begin
- **Offset detection:** Identifies when notes end  
- **Frame classification:** Determines which notes are active at each time frame
- **Velocity estimation:** Estimates the intensity/volume of each note

### Input Processing

- Audio is resampled to 16 kHz
- Processed in segments of 10 seconds with overlapping windows
- Spectral features are extracted and fed to the neural network

### Dependencies

- Python >= 3.9
- PyTorch >= 1.9.0
- librosa >= 0.9.0
- huggingface-hub >= 0.16.0
- numpy >= 1.21.0
- safetensors >= 0.3.0
- piano-transcription-inference (custom fork)

## Citation

If you use this model in your research, please cite:

```bibtex
@inproceedings{charlie_parker,
    author = {Riley, Xavier and Dixon, Simon},
    title = {Reconstructing the Charlie Parker Omnibook using an audio-to-score automatic transcription pipeline},
    booktitle = {Proceedings of the 21st Sound and Music Computing Conference},
    year = 2024,
    address = {Porto, Portugal}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please:
1. Check the [Issues](https://github.com/xavriley/hf_sax_transcription/issues) page
2. Create a new issue with a detailed description of your problem
3. Include information about your environment and the audio files you're trying to process

## Related Work

This model builds upon the piano transcription work and adapts it specifically for saxophone audio. The underlying architecture is based on proven methods in automatic music transcription.

## Changelog

### v0.1.0
- Initial release
- Basic saxophone-to-MIDI transcription functionality
- Hugging Face Hub integration
- Support for various audio formats
