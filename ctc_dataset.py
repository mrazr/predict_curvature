import typing
from pathlib import Path
import os
from typing import Optional, Union, Iterable, Sequence

import albumentations as A
import cv2
from skimage import io
import torch
from torch.utils import data
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t


class CTCDataset(data.Dataset):
    SEG_PREFIX = 'man_seg'

    def __init__(self, root_folder: Path, ann_type: str, train: bool=True):
        self.root_folder: Path = root_folder
        self._train: bool = train

        self.ann_type: str = ann_type.upper()

        self.example_paths: typing.List[typing.Tuple[Path, Path]] = []
        self._load_example_paths()
        # print(self.example_paths)

        if self._train:
            self.transform = A.Compose([
                # A.LongestMaxSize(max_size=256),
                A.ToFloat(),
                A.Flip(p=0.5)
            ])
        else:
            self.transform = A.Compose([
                # A.LongestMaxSize(max_size=256),
                A.ToFloat()
            ])

    def _load_example_paths(self):
        sequence_folders: typing.List[Path] = [Path(entry.path) for entry in os.scandir(self.root_folder) if entry.is_dir() and entry.name.isnumeric()]

        truth_type = 'ST' if self._train else 'GT'

        for sequence_folder in sequence_folders:
            truth_folder = Path(f'{sequence_folder}_{truth_type}') / self.ann_type
            if not truth_folder.exists():
                continue
            for fname in os.listdir(truth_folder):
                img_name = 't' + fname[len(self.SEG_PREFIX):]
                self.example_paths.append((sequence_folder / img_name, truth_folder / fname))

    def __len__(self) -> int:
        return len(self.example_paths)

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        img_path, ann_path = self.example_paths[idx]

        img = io.imread(img_path)
        ann = io.imread(ann_path)

        transformed = self.transform(image=img, mask=ann)

        return torch.unsqueeze(torch.tensor(transformed['image']), dim=0), torch.unsqueeze(torch.tensor(transformed['mask']), dim=0)

    def _getitem(self, idx: int):
        img_path, ann_path = self.example_paths[idx]

        img = io.imread(img_path)
        ann = io.imread(ann_path)

        transformed = self.transform(image=img, mask=ann)

        return img, transformed['image'], ann ,transformed['mask']
