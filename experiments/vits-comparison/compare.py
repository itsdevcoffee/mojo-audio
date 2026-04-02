#!/usr/bin/env python3
"""End-to-end RVC v2 voice conversion comparison script.

Runs VoiceConverter.from_pretrained + convert on an input WAV,
saves the output, and prints diagnostic stats.

Usage (run on Spark where memory is not constrained):
    python compare.py --input path/to/input.wav --model path/to/model.pth

Optional flags:
    --output          Output WAV path (default: input stem + _converted.wav)
    --pitch-shift     Semitones to shift (default: 0)
    --hubert          HuggingFace model ID for content encoder (default: facebook/hubert-base-ls960)
    --device          "auto", "cpu", or "gpu" (default: auto)
    --no-audio        Skip writing audio, just print stats
"""

from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="RVC v2 voice conversion via mojo-audio")
    p.add_argument("--input", required=True, help="Input WAV file")
    p.add_argument("--model", required=True, help="RVC v2 .pth checkpoint")
    p.add_argument("--output", default=None, help="Output WAV path")
    p.add_argument("--pitch-shift", type=float, default=0.0, metavar="SEMITONES",
                   help="Pitch shift in semitones (default: 0)")
    p.add_argument("--hubert", default="facebook/hubert-base-ls960",
                   help="HuggingFace model ID for HuBERT/ContentVec")
    p.add_argument("--rmvpe", default="lj1995/VoiceConversionWebUI",
                   help="HuggingFace repo for RMVPE weights")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--no-audio", action="store_true", help="Skip saving output audio")
    return p.parse_args()


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file to float32 numpy array.

    Returns:
        (audio, sample_rate): audio is [1, N] float32 in [-1, 1].
    """
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns [N, C]; take first channel, reshape to [1, N]
    mono = data[:, 0]
    return mono[np.newaxis, :], sr


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Save [1, N] or [N] float32 audio to WAV at `sr` Hz."""
    import soundfile as sf
    wav = audio.squeeze()
    sf.write(path, wav, sr, subtype="PCM_16")


def print_stats(audio_in: np.ndarray, audio_out: np.ndarray, sr_in: int, sr_out: int,
                elapsed: float) -> None:
    """Print diagnostic information about the conversion."""
    dur_in = audio_in.shape[-1] / sr_in
    dur_out = audio_out.shape[-1] / sr_out
    rms_in = float(np.sqrt(np.mean(audio_in ** 2)))
    rms_out = float(np.sqrt(np.mean(audio_out ** 2)))

    print("\n--- Conversion Stats ---")
    print(f"  Input duration:   {dur_in:.2f}s  ({sr_in} Hz)")
    print(f"  Output duration:  {dur_out:.2f}s  ({sr_out} Hz)")
    print(f"  Input RMS:        {rms_in:.5f}")
    print(f"  Output RMS:       {rms_out:.5f}")
    print(f"  Output range:     [{audio_out.min():.4f}, {audio_out.max():.4f}]")
    print(f"  Wall-clock time:  {elapsed:.1f}s")
    print(f"  RTF:              {elapsed / dur_in:.2f}x  (lower is faster)")
    print("------------------------\n")


def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # Resolve paths
    # ---------------------------------------------------------------
    input_path = os.path.abspath(args.input)
    model_path = os.path.abspath(args.model)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: Model checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + "_converted.wav"
    else:
        output_path = os.path.abspath(args.output)

    # ---------------------------------------------------------------
    # Add mojo-audio src to path
    # ---------------------------------------------------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # ---------------------------------------------------------------
    # Load input audio
    # ---------------------------------------------------------------
    print(f"Loading input: {input_path}")
    audio_in, sr_in = load_wav(input_path)
    print(f"  Shape: {audio_in.shape}  SR: {sr_in} Hz")

    # ---------------------------------------------------------------
    # Build pipeline
    # ---------------------------------------------------------------
    print(f"\nLoading VoiceConverter from: {model_path}")
    print(f"  HuBERT/ContentVec: {args.hubert}")
    print(f"  RMVPE:             {args.rmvpe}")
    print(f"  Device:            {args.device}")
    t_load_start = time.time()

    from models.voice_converter import VoiceConverter
    vc = VoiceConverter.from_pretrained(
        model_path,
        hubert_path=args.hubert,
        rmvpe_path=args.rmvpe,
        device=args.device,
    )

    t_load_end = time.time()
    print(f"  Model loaded in {t_load_end - t_load_start:.1f}s")

    # ---------------------------------------------------------------
    # Run conversion
    # ---------------------------------------------------------------
    print(f"\nConverting... (pitch_shift={args.pitch_shift:+.1f} semitones)")
    t_infer_start = time.time()

    audio_out = vc.convert(audio_in, pitch_shift=args.pitch_shift, sr=sr_in)

    t_infer_end = time.time()
    elapsed = t_infer_end - t_infer_start
    print(f"  Conversion done in {elapsed:.2f}s")

    # ---------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------
    sr_out = vc._config.get("sr", 48000)
    print_stats(audio_in, audio_out, sr_in, sr_out, elapsed)

    # ---------------------------------------------------------------
    # Save output
    # ---------------------------------------------------------------
    if not args.no_audio:
        print(f"Saving output: {output_path}")
        save_wav(output_path, audio_out, sr_out)
        print("  Done.")
    else:
        print("(--no-audio: skipping WAV write)")


if __name__ == "__main__":
    main()
