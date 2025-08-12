## sd-sound-experiment

Generate a short video from an input audio file using Meta's ImageBind audio embeddings and Stable Diffusion (v1-5). The script slices the audio, encodes each slice to an embedding, synthesizes a frame per slice via Stable Diffusion, and stitches frames with the original audio into an MP4.

### Features

- Converts audio into visual frames using ImageBind audio embeddings
- Uses Stable Diffusion v1-5 with custom prompt embeddings (no text prompt)
- Assembles frames into an H.264/AAC MP4 with the original audio track
- Works on CPU, CUDA GPUs, and Apple Silicon (MPS)

## Requirements

- Python 3.9+ recommended
- PyTorch (CPU/CUDA or MPS) compatible with your platform
- FFmpeg available on your system for `moviepy`
  - macOS (Homebrew): `brew install ffmpeg`

Python packages are listed in `requirements.txt` and include:

- `torch`, `torchaudio`, `torchvision`
- `diffusers`, `transformers`, `accelerate`, `safetensors`
- `pytorchvideo`, `fvcore`, `iopath`, `timm`, `ftfy`, `regex`, `einops`
- `moviepy`, `librosa`, `soundfile`, `Pillow`, `opencv-python`, `numpy`, `scipy`, `matplotlib`

## Setup

1. Clone this repository.
2. Create and activate a virtual environment (recommended).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Fetch the ImageBind source into a local subfolder so it can be imported:

```bash
# From the repo root
git clone https://github.com/facebookresearch/ImageBind.git ImageBind
```

The script applies a small compatibility shim in `imagebind_patch.py` to work with newer torchvision/pytorchvideo versions.

4. (Optional, but recommended) Authenticate with Hugging Face if needed for model downloads and license acceptance for `runwayml/stable-diffusion-v1-5`.

```bash
pip install huggingface_hub
huggingface-cli login
```

## Usage

Basic example:

```bash
python main.py path/to/audio.wav --device mps --fps 6 --samples-per-window 2
```

- `audio_path` (positional): Path to the input audio (e.g., WAV/MP3). The script resamples to 16 kHz internally.
- `--device`: `cpu`, `cuda`, or `mps` (Apple Silicon). Default is `cpu` on macOS, else `cuda`.
- `--fps`: Output video frames per second (default: `4`).
- `--samples-per-window`: Number of slices per 2-second window (1–5, default: `2`). Higher values produce more overlap and more frames.

Output is an MP4 named like `output_XXXXXXXX.mp4` in the current directory.

### How it works

- The audio is resampled to 16 kHz and split into 2-second windows with hop size = `window_sec / samples_per_window`.
- Each slice is converted to a mel spectrogram and passed through ImageBind to obtain a 1024-dim audio embedding.
- The embedding is truncated to 768 dims and tiled across the Stable Diffusion tokenizer length to form `prompt_embeds` (no text).
- Stable Diffusion v1-5 generates one image per slice (30 steps, guidance scale 5.0 by default in the script).
- Frames are written as PNGs, then combined with the original audio via `moviepy` into an H.264/AAC MP4.

## Tips and troubleshooting

- Apple Silicon (MPS): set `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow CPU fallback for unsupported ops.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

- CUDA OOM: reduce the frame count (lower `--fps` or `--samples-per-window`). If needed, you can also lower SD inference steps or guidance scale by editing `main.py`.
- First run model download: the Stable Diffusion weights will be downloaded on first use. Ensure you have accepted the model license on Hugging Face and are logged in if required.
- FFmpeg errors: install FFmpeg so `moviepy` can encode MP4s.
- Safety checker: the SD safety checker is explicitly disabled in `main.py` for experimentation.

## Project structure

```
.
├── main.py                # Entry point: audio → frames via ImageBind → SD → video
├── imagebind_patch.py     # Compatibility shim for torchvision/pytorchvideo
├── requirements.txt       # Python dependencies
└── ImageBind/             # (You clone this) Meta's ImageBind source
```

## Acknowledgements

- ImageBind by Meta AI Research
- Stable Diffusion and Diffusers by Stability AI / Runway / Hugging Face ecosystem
- MoviePy for video assembly, PyTorch/Torchaudio for audio processing

## License

This repository is for research/experimentation. Respect and comply with the licenses and usage terms of all included/required models and libraries (e.g., ImageBind, Stable Diffusion, Diffusers, and FFmpeg). No license is implied for third-party assets.
