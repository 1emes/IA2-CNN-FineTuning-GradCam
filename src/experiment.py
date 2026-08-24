"""Treina ResNet-18 e compara regiões Grad-CAM entre estratégias de transferência."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STRATEGIES = ("frozen", "full")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_indices(
    labels: Iterable[int], validation_fraction: float, train_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        validation_size = max(1, round(len(indices) * validation_fraction))
        validation_indices.extend(indices[:validation_size])
        candidates = indices[validation_size:]
        retained = max(1, round(len(candidates) * train_fraction))
        train_indices.extend(candidates[:retained])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


class PetEvaluationDataset(Dataset):
    """Aplica transformações determinísticas separadamente à imagem e à máscara."""

    def __init__(self, root: str, download: bool = True) -> None:
        self.dataset = OxfordIIITPet(
            root=root,
            split="test",
            target_types=("category", "segmentation"),
            download=download,
        )
        self.classes = self.dataset.classes
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(224),
                transforms.PILToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        label, trimap = target
        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(trimap).squeeze(0)
        # Oxford-IIIT Pet: 1 = animal, 2 = fundo, 3 = contorno.
        foreground = (mask_tensor != 2).to(torch.bool)
        return image_tensor, int(label), foreground


if __name__ == "__main__":
    print("Módulo de experimento – em construção.")
