import os
from glob import glob
from torch import Tensor
from typing import Tuple


from clmr.datasets import Dataset


class AUDIO(Dataset):
    """Create a Dataset for any folder of audio files.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        src_ext_audio (str): The extension of the audio files to analyze.
    """

    def __init__(
        self,
        root: str,
        src_ext_audio: str = ".wav",
        n_classes: int = 1,
    ) -> None:
        super(AUDIO, self).__init__(root)

        self.labels = [
            "BassHouse",
            "Drum&Bass",
            "FutureHouse",
            "Hardstyle",
            "House",
            "MelodicDubstep"
            ]

        self.label2idx = {}
        for idx, label in enumerate(self.labels):
            self.label2idx[label] = idx

        self._path = root
        self._src_ext_audio = src_ext_audio
        self.n_classes = len(self.label2idx.keys())

        self.fl = glob(
            os.path.join(self._path, "**", "*{}".format(self._src_ext_audio)),
            recursive=True,
        )

        if len(self.fl) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    self._path
                )
            )

    def file_path(self, n: int) -> str:
        fp = self.fl[n]
        return fp

    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple [Tensor, str]: ``(waveform, label)``
        """
        audio, _, label = self.load(n)
        label = self.label2idx[label]
        return audio, label

    def __len__(self) -> int:
        return len(self.fl)
