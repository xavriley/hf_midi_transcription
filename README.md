# MIDI Transcription Model

This code transcribes solo, monophonic audio from multiple instruments to MIDI. Currently supports saxophone, bass, guitar, and piano. If you need to separate instruments from a mix first please check out UVR or MVSep.com

For drum transcription please check out [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch).

**Key Features:**
- Audio-to-MIDI transcription for multiple instruments (saxophone, bass, guitar, piano)
- Instrument-specific model selection
- Batch processing support for efficient inference
- GPU acceleration support
- Simple JSON configuration for adding new instruments

## Model Details

- **Model Type:** Audio-to-MIDI transcription
- **Architecture:** Convolutional Recurrent Neural Network (CRNN) with onset/offset/frame/velocity regression
- **Supported Instruments:** Saxophone, Bass, Guitar, Piano
- **Input:** Audio files (WAV, MP3, etc.)
- **Output:** MIDI files
- **Sample Rate:** 16 kHz
- **License:** MIT

## Installation

### Option 1: Install via pip (Recommended)

```bash
pip install hf-midi-transcription
```

### Option 2: Install from source

```bash
git clone https://github.com/xavriley/hf_midi_transcription.git
cd hf_midi_transcription
pip install -e .
```

## Quick Start

### Using the Command Line Interface (CLI)

The simplest way to use the model is through the command line:

```bash
# Basic usage - specify instrument
midi_transcription input_audio.wav output.mid --instrument saxophone

# Different instruments
midi_transcription bass_line.wav output.mid --instrument bass
midi_transcription guitar_solo.wav output.mid --instrument guitar
midi_transcription piano_piece.wav output.mid --instrument piano

# With additional options
midi_transcription recording.mp3 transcription.mid --instrument saxophone --device cuda --verbose

# Using a custom checkpoint
midi_transcription audio.wav output.mid --instrument bass --checkpoint my_bass_model.pth --batch-size 16

# Using a model from Hugging Face Hub
midi_transcription audio.wav output.mid --instrument guitar --model-id username/custom-model
```

**CLI Options:**
- `--instrument {saxophone,bass,guitar,piano}` - Choose instrument to transcribe (required)
- `--device {auto,cpu,cuda}` - Choose compute device (default: auto)
- `--batch-size N` - Set batch size for processing (default: 8)
- `--checkpoint PATH` - Use custom checkpoint file (default: auto-selected based on instrument)
- `--model-id ID` - Load model from Hugging Face Hub
- `--verbose, -v` - Enable detailed output
- `--help` - Show full help

### Using the Python API

```python
from hf_midi_transcription import MidiTranscriptionModel

# Load the pre-trained model for a specific instrument
model = MidiTranscriptionModel.from_pretrained(
    "xavriley/midi-transcription-models", 
    instrument="saxophone"
)

# Transcribe audio file to MIDI
model.transcribe("saxophone_recording.wav", "output.mid")

# Use different instruments
bass_model = MidiTranscriptionModel.from_pretrained(
    "xavriley/midi-transcription-models", 
    instrument="bass"
)
bass_model.transcribe("bass_line.wav", "bass_output.mid")
```

### Using with audio arrays

```python
import librosa
from hf_midi_transcription import MidiTranscriptionModel

# Load model for specific instrument
model = MidiTranscriptionModel.from_pretrained(
    "xavriley/midi-transcription-models", 
    instrument="guitar"
)

# Load audio as array
audio, sr = librosa.load("guitar_recording.wav", sr=16000)

# Transcribe audio array to MIDI
model.transcribe_audio_array(audio, "output.mid")
```

### Automatic model download

The model will automatically download the required checkpoint file from Hugging Face Hub on first use:

```python
from hf_midi_transcription import MidiTranscriptionModel

# Model will automatically download the appropriate checkpoint for the instrument
model = MidiTranscriptionModel(
    instrument="piano",
    device="cuda",  # or "cpu" 
    batch_size=8
)

# Transcribe
model.transcribe("piano_input.wav", "output.mid")
```

### Using local checkpoint files

If you have a local checkpoint file:

```python
from hf_midi_transcription import MidiTranscriptionModel

# Use local checkpoint for specific instrument
model = MidiTranscriptionModel(
    instrument="bass",
    checkpoint_path="path/to/your_bass_model.pth",
    device="cuda",
    batch_size=8
)

# Transcribe
model.transcribe("bass_input.wav", "output.mid")
```

## Model Performance

The models have been trained on datasets specific to each instrument and optimized for:
- **Saxophone**: Monophonic performance (single note at a time), various saxophone types (soprano, alto, tenor, baritone)
- **Bass**: Electric and acoustic bass lines, various playing techniques
- **Guitar**: Solo guitar performances, fingerpicking and strumming patterns
- **Piano**: Monophonic piano melodies and simple accompaniments
- Different playing styles and dynamics for each instrument
- Clean and reverberant recording conditions

## Limitations

- Optimized for monophonic performance (single notes, not chords) across all instruments
- Performance may vary with heavily processed or distorted audio
- Best results with clean, well-recorded audio for the selected instrument
- Instrument selection is crucial - using the wrong instrument model will produce poor results 

## Model File Management

### Automatic Downloads

The instrument-specific models are automatically downloaded from Hugging Face Hub when needed:

- **First use**: Models download automatically to your HF cache directory
- **Subsequent uses**: Models load from cache (no re-download)
- **Offline use**: Works if models are already cached
- **Custom models**: Supports loading custom checkpoints from local files or other HF repositories
- **Model sizes**: Vary by instrument (~95MB each)

### Cache Location

Models are cached in your system's Hugging Face cache directory:
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

### Advanced Usage

```python
# Use a different model from Hugging Face Hub
model = MidiTranscriptionModel(
    instrument="guitar",
    checkpoint_path="username/custom-guitar-model/model.pth"
)

# Force re-download (useful for model updates)
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download(
    repo_id="xavriley/midi-transcription-models",
    filename="guitar_model.pth",
    force_download=True
)
model = MidiTranscriptionModel(instrument="guitar", checkpoint_path=checkpoint)
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
1. Check the [Issues](https://github.com/xavriley/hf_midi_transcription/issues) page
2. Create a new issue with a detailed description of your problem
3. Include information about your environment, the instrument you're transcribing, and the audio files you're trying to process

## Related Work

This model builds upon the piano transcription work and adapts it for multiple instruments including saxophone, bass, guitar, and piano. The underlying architecture is based on proven methods in automatic music transcription.

## Changelog

### v0.1.0
- Initial release with multi-instrument support
- Support for saxophone, bass, guitar, and piano transcription
- Instrument-specific model selection via CLI and Python API
- Hugging Face Hub integration with unified model repository
- JSON-based instrument configuration system
- Backward compatibility with saxophone-only usage
