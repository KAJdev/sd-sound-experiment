#!/usr/bin/env python
"""
Audio-to-video generation using ImageBind and Stable Diffusion.
Converts audio embeddings into visual representations through a diffusion pipeline.
"""

import argparse
import uuid
import os
import sys
import tempfile
import torch
import torchaudio

exec(open(os.path.join(os.path.dirname(__file__), "imagebind_patch.py")).read())
sys.path.append(os.path.join(os.path.dirname(__file__), "ImageBind"))

from diffusers import StableDiffusionPipeline
from imagebind import imagebind_model, data as ib_data
from imagebind.models.imagebind_model import ModalityType
from moviepy import ImageSequenceClip, AudioFileClip

parser = argparse.ArgumentParser(
    description="Generate video from audio using ImageBind and Stable Diffusion"
)
parser.add_argument("audio_path", type=str, help="Path to input audio file")
parser.add_argument(
    "--device",
    default="cpu" if sys.platform == "darwin" else "cuda",
    choices=["cuda", "cpu", "mps"],
    help="Device to run inference on",
)
parser.add_argument("--fps", type=int, default=4, help="Output video frame rate")
parser.add_argument(
    "--samples-per-window",
    type=int,
    default=2,
    choices=range(1, 6),
    help="Number of samples per audio window",
)
args = parser.parse_args()

wav, sr = torchaudio.load(args.audio_path)
wav = torchaudio.functional.resample(wav, sr, 16_000)
sr = 16_000

window_sec = 2.0
hop_sec = window_sec / args.samples_per_window
window_samples = int(window_sec * sr)
hop_samples = int(hop_sec * sr)

slices = []
for s in range(0, wav.shape[1] - window_samples, hop_samples):
    slice_ = wav[:, s : s + window_samples]
    slices.append(slice_)

print(f"Processing {len(slices)} audio slices → {len(slices)/args.fps:.1f}s video")

device = torch.device(args.device)
ib = imagebind_model.ImageBindModel().to(device)
ib.eval()

dtype = torch.float32 if device.type == "cpu" else torch.float16
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
).to(device)
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

token_len = pipe.tokenizer.model_max_length
embed_dim = pipe.text_encoder.config.hidden_size

frames = []
tmpdir = tempfile.mkdtemp()


def process_audio_tensor(audio_tensor, device):
    """Convert audio tensor to ImageBind-compatible mel spectrogram."""
    if audio_tensor.dim() == 2:
        waveform = audio_tensor
    elif audio_tensor.dim() == 3:
        waveform = audio_tensor[0]
    elif audio_tensor.dim() == 1:
        waveform = audio_tensor.unsqueeze(0)
    else:
        waveform = audio_tensor

    melspec = ib_data.waveform2melspec(
        waveform=waveform, sample_rate=16000, num_mel_bins=128, target_length=204
    )
    return melspec.unsqueeze(0).to(device)


with torch.inference_mode():
    for i, audio_clip in enumerate(slices):
        audio_input = process_audio_tensor(audio_clip, device)

        inputs = {ModalityType.AUDIO: audio_input}
        emb_1024 = ib(inputs)[ModalityType.AUDIO]

        emb_768 = emb_1024[:, :embed_dim]
        prompt_embeds = emb_768.unsqueeze(1).repeat(1, token_len, 1)

        image = pipe(
            prompt_embeds=prompt_embeds, num_inference_steps=30, guidance_scale=5.0
        ).images[0]

        frame_path = os.path.join(tmpdir, f"{i:06}.png")
        image.save(frame_path)
        frames.append(frame_path)

        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{len(slices)} frames")

clip = ImageSequenceClip(frames, fps=args.fps)
audioclip = AudioFileClip(args.audio_path)
clip = clip.with_audio(audioclip)

outfile = f"output_{uuid.uuid4().hex[:8]}.mp4"
clip.write_videofile(outfile, codec="libx264", audio_codec="aac")
print(f"Video saved: {outfile}")
