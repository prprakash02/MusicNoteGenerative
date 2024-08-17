import glob
import os

import numpy as np
import librosa
import torch

N_MELS = 128


def get_spectrogram(audio_file: str, return_sr: bool = False) -> np.ndarray | tuple[np.ndarray, int]:
    amplitudes, sr = librosa.load(audio_file)
    spectrogram = librosa.feature.melspectrogram(y=amplitudes, sr=sr, n_mels=N_MELS)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    if return_sr:
        return spectrogram, sr
    return spectrogram


class BachDataset(torch.utils.data.Dataset):
    def __init__(self, data_root_path: str):
        mp3_files = []
        audio = os.path.join(data_root_path, "audio")
        for artist in os.listdir(audio):
            artist_path = os.path.join(audio, artist)
            mp3_files.extend(glob.glob(artist_path + "/*.mp3"))

        self.mp3_files = mp3_files

    def __len__(self):
        return len(self.mp3_files)

    def __getitem__(self, idx):
        audio_file = self.mp3_files[idx]
        spectrogram = get_spectrogram(audio_file)
        spectrogram = torch.Tensor(spectrogram)
        return spectrogram


def test():
    dataset = BachDataset(data_root_path="../bach-dataset")
    test = dataset[1]


if __name__ == "__main__":
    test()
