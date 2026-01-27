import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from piano_transcription_inference import PianoTranscription, sample_rate
from librosa.core import load
import os
import json


class MidiTranscriptionModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/xavriley/hf_midi_transcription/",
    pipeline_tag="audio-to-midi",
    license="mit",
    tags=["audio", "midi", "transcription", "multi-instrument", "music"],
):
    """
    A multi-instrument transcription model that converts audio to MIDI using specialized neural networks.
    
    This model is based on piano transcription techniques adapted for various instruments including
    saxophone, bass, guitar, and piano. It can transcribe monophonic audio files to MIDI format.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None, 
        instrument: str = "saxophone",
        checkpoint_path: Optional[str] = None,
        batch_size: int = 8,
        onset_threshold: float = 0.3,
        offset_threshold: float = 0.3,
        frame_threshold: float = 0.1,
        pedal_offset_threshold: float = 0.2,
        **kwargs
    ):
        """
        Initialize the MIDI Transcription Model.
        
        Args:
            device (str, optional): Device to run the model on ('cuda', 'cpu'). 
                                  If None, automatically selects based on availability.
            instrument (str): Instrument to transcribe ('saxophone', 'bass', 'guitar', 'piano').
            checkpoint_path (str, optional): Path to model checkpoint file. If None, uses default for instrument.
            batch_size (int): Batch size for processing audio segments.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__()
        
        # Load instrument configuration
        self.instrument_config = self._load_instrument_config()
        
        # Validate instrument
        if instrument not in self.instrument_config:
            available = list(self.instrument_config.keys())
            raise ValueError(f"Unsupported instrument '{instrument}'. Available: {available}")
        
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.instrument_config[instrument]["checkpoint_file"]
        
        # Store configuration for serialization
        self.config = {
            "device": device,
            "instrument": instrument,
            "checkpoint_path": checkpoint_path,
            "batch_size": batch_size,
            "sample_rate": sample_rate,
            "onset_threshold": onset_threshold,
            "offset_threshold": offset_threshold,
            "frame_threshold": frame_threshold,
            "pedal_offset_threshold": pedal_offset_threshold,
        }
        
        if not device or device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.instrument = instrument
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        
        # Initialize the transcriptor
        self._init_transcriptor(instrument)
    
    def _load_instrument_config(self) -> Dict[str, Any]:
        """Load instrument configuration from instruments.json file."""
        # Look for config file in package directory or current directory
        config_paths = [
            Path(__file__).parent.parent / "instruments.json",  # Package root
            Path("instruments.json"),  # Current directory
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        
        # Fallback configuration if file not found
        return {
            "saxophone": {"checkpoint_file": "filosax_25k.pth", "description": "Saxophone model"},
            "bass": {"checkpoint_file": "filobass_20000_iterations.pth", "description": "Bass model"},
            "guitar": {"checkpoint_file": "guitar-gaps.pth", "description": "Guitar model"},
            "piano": {"checkpoint_file": "piano.pth", "description": "Piano model"},
        }
    
    def _download_model_if_needed(self, checkpoint_path: str) -> str:
        """
        Download model from Hugging Face Hub if not found locally.
        
        Args:
            checkpoint_path: Path to checkpoint file (local or HF Hub)
            
        Returns:
            str: Local path to the checkpoint file
        """
        # If it's already a local file that exists, return it
        if os.path.exists(checkpoint_path):
            return checkpoint_path
            
        # Default model repository
        default_repo = "xavriley/midi-transcription-models"
        
        # Handle different input formats
        if checkpoint_path in [config["checkpoint_file"] for config in self.instrument_config.values()]:
            # Download the model for this instrument
            try:
                print(f"Downloading {self.instrument} model from {default_repo}...")
                local_path = hf_hub_download(
                    repo_id=default_repo,
                    filename=checkpoint_path,
                    cache_dir=None  # Use default HF cache
                )
                print(f"✓ Model downloaded to: {local_path}")
                return local_path
            except Exception as e:
                # Fallback: look for local file in current directory
                if os.path.exists(checkpoint_path):
                    print(f"⚠ Download failed ({e}), using local file: {checkpoint_path}")
                    return checkpoint_path
                else:
                    raise FileNotFoundError(
                        f"Could not download {self.instrument} model from {default_repo} and no local file found. "
                        f"Error: {e}\n"
                        f"Please ensure you have internet access or provide a local checkpoint file."
                    )
        else:
            # Custom checkpoint path - check if it exists, otherwise assume it's a HF repo
            if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
                # Assume it's a HF Hub path like "user/repo/filename.pth"
                parts = checkpoint_path.split("/")
                if len(parts) >= 3:
                    repo_id = "/".join(parts[:-1])
                    filename = parts[-1]
                    try:
                        print(f"Downloading model from {repo_id}...")
                        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                        print(f"✓ Model downloaded to: {local_path}")
                        return local_path
                    except Exception as e:
                        raise FileNotFoundError(f"Could not download {checkpoint_path}: {e}")
            
            # If we get here, it should be a local file
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            return checkpoint_path

    def _init_transcriptor(self, 
        instrument: str = "saxophone",
        onset_threshold: float = 0.3,
        offset_threshold: float = 0.3,
        frame_threshold: float = 0.1,
        pedal_offset_threshold: float = 0.2):
        """Initialize the piano transcription model adapted for the selected instrument."""
        # Ensure we have the model file (download if necessary)
        actual_checkpoint_path = self._download_model_if_needed(self.checkpoint_path)
        
        self.transcriptor = PianoTranscription(
            "Note_pedal" if instrument == "piano" else "Regress_onset_offset_frame_velocity_CRNN",
            device=self.device,
            checkpoint_path=actual_checkpoint_path,
            segment_samples=10 * sample_rate,
            batch_size=self.batch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch compatibility.
        
        Note: This model is primarily designed for inference via the transcribe method.
        The forward method is included for PyTorch compatibility but may not be 
        the primary interface for users.
        
        Args:
            x (torch.Tensor): Input audio tensor
            
        Returns:
            torch.Tensor: Processed output (implementation depends on use case)
        """
        # For now, this is a placeholder as the main interface is the transcribe method
        # In a full implementation, this would process the tensor through the model
        return x
    
    def transcribe(
        self, 
        audio_path: Union[str, Path], 
        midi_output_path: Union[str, Path],
        activations: Optional[bool] = False
    ) -> str:
        """
        Transcribe audio file to MIDI.
        
        Args:
            audio_path (Union[str, Path]): Path to input audio file
            midi_output_path (Union[str, Path]): Path for output MIDI file
            
        Returns:
            str: Path to the generated MIDI file
        """
        # Load audio with librosa
        audio, _ = load(str(audio_path), sr=sample_rate)
        
        # Transcribe using the underlying model
        if activations:
            result = self.transcriptor.transcribe(audio, str(midi_output_path))
            return str(midi_output_path), result
        else:
            result = self.transcriptor.transcribe(audio, str(midi_output_path))
            return str(midi_output_path)
    
    def transcribe_audio_array(
        self, 
        audio: torch.Tensor, 
        midi_output_path: Union[str, Path],
        activations: Optional[bool] = False
    ) -> str:
        """
        Transcribe audio array/tensor to MIDI.
        
        Args:
            audio (torch.Tensor): Audio data as tensor
            midi_output_path (Union[str, Path]): Path for output MIDI file
            
        Returns:
            str: Path to the generated MIDI file
        """
        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
        # Transcribe using the underlying model
        result = self.transcriptor.transcribe(audio, str(midi_output_path))
        
        if activations:
            return str(midi_output_path), result
        else:
            return str(midi_output_path)
    
    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save model configuration and checkpoint.
        This method is called by PyTorchModelHubMixin.save_pretrained().
        """
        save_directory = Path(save_directory)
        
        # Copy the checkpoint file to the save directory if it exists
        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.exists():
            import shutil
            target_checkpoint = save_directory / checkpoint_path.name
            shutil.copy2(checkpoint_path, target_checkpoint)
            
            # Update config to use relative path
            self.config["checkpoint_path"] = checkpoint_path.name
        
        # Save a dummy state dict for PyTorch compatibility
        # Since this model wraps an external transcriptor, we'll save minimal state
        state_dict = {
            "dummy_param": torch.tensor([1.0]),  # Placeholder parameter
        }
        
        # Save as safetensors (this is what PyTorchModelHubMixin expects)
        import safetensors.torch
        safetensors.torch.save_file(state_dict, save_directory / "model.safetensors")
    
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[str],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """
        Load model from pretrained checkpoint.
        This method is called by PyTorchModelHubMixin.from_pretrained().
        """
        # Load the saved state and config
        import safetensors.torch
        from huggingface_hub import snapshot_download
        import json
        
        # Download model files
        if isinstance(model_id, Path) or (isinstance(model_id, str) and Path(model_id).exists()):
            # Local path
            model_path = Path(model_id)
        else:
            # Remote model - download from hub
            model_path = Path(snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
            ))
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
        
        # Load model state (mainly for PyTorch compatibility)
        state_dict_path = model_path / "model.safetensors"
        if state_dict_path.exists():
            state_dict = safetensors.torch.load_file(state_dict_path, device=map_location)
            # The state dict only contains dummy parameters for PyTorch compatibility
        
        # Update checkpoint path to absolute path if it exists in the model directory
        if "checkpoint_path" in config:
            checkpoint_file = model_path / config["checkpoint_path"]
            if checkpoint_file.exists():
                config["checkpoint_path"] = str(checkpoint_file)
        
        # Create model instance with loaded config
        model = cls(**{**config, **model_kwargs})
        
        return model


# Backward compatibility alias
SaxophoneTranscriptionModel = MidiTranscriptionModel
