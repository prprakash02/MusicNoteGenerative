import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro

from src.loading import get_spectrogram
from src.models import MODELS


@dataclass
class Args:
    audio_path: str
    model_path: str = None
    model_class: str = "Identity"
    output_dir: str = "notes"


def extract_notes(
    audio_path: str,
    model_path: str = None,
    model_class: str = "Identity",
) -> tuple[np.ndarray, int]:
    model = MODELS[model_class]()
    if model_path is not None:
        model.load(torch.load(model_path, map_location=torch.device("cpu")))

    spectrogram, sr = get_spectrogram(audio_path, return_sr=True)
    spectrogram = torch.Tensor(spectrogram)
    notes = model.extract_notes(spectrogram).numpy().astype(int)
    return notes, sr


def save_notes(notes: np.ndarray, sr: int, output_dir: str) -> None:
    path = os.path.join(output_dir, "notes.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"sr: {sr}")
    with open(path, "a") as f:
        np.savetxt(f, notes, fmt='%i', delimiter=",")


def main() -> None:
    args = tyro.cli(Args)
    notes, sr = extract_notes(args.audio_path, args.model_path, args.model_class)
    save_notes(notes, sr, args.output_dir)


if __name__ == "__main__":
    main()
