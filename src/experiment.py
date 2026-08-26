"""Treina ResNet-18 e compara regiões Grad-CAM entre estratégias de transferência."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STRATEGIES = ("frozen", "full")


@dataclass
class RunConfig:
    data_dir: str
    output_dir: str
    epochs: int = 15
    batch_size: int = 64
    workers: int = 4
    seed: int = 42
    train_fraction: float = 1.0
    learning_rate_head: float = 1e-3
    learning_rate_backbone: float = 1e-4
    weight_decay: float = 1e-4
    gradcam_samples: int = 200
    saliency_quantile: float = 0.80
    test_limit: int = 0


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


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/main")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradcam-samples", type=int, default=200)
    parser.add_argument("--saliency-quantile", type=float, default=0.80)
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limita o teste para depuração; zero usa todo o conjunto oficial.",
    )
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction deve estar no intervalo (0, 1].")
    if not 0 < args.saliency_quantile < 1:
        parser.error("--saliency-quantile deve estar no intervalo (0, 1).")
    if args.test_limit < 0:
        parser.error("--test-limit não pode ser negativo.")
    return RunConfig(**vars(args))


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


def make_dataloaders(config: RunConfig):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    metadata = OxfordIIITPet(
        root=config.data_dir,
        split="trainval",
        target_types="category",
        download=True,
    )
    labels = getattr(metadata, "_labels", None)
    if labels is None:
        labels = [metadata[index][1] for index in range(len(metadata))]
    train_indices, validation_indices = stratified_indices(
        labels, validation_fraction=0.20, train_fraction=config.train_fraction, seed=config.seed
    )

    training_data = OxfordIIITPet(
        root=config.data_dir,
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=False,
    )
    validation_data = OxfordIIITPet(
        root=config.data_dir,
        split="trainval",
        target_types="category",
        transform=evaluation_transform,
        download=False,
    )
    test_data = PetEvaluationDataset(config.data_dir, download=True)
    if config.test_limit:
        test_data = Subset(test_data, range(min(config.test_limit, len(test_data))))

    generator = torch.Generator().manual_seed(config.seed)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(training_data, train_indices),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices), shuffle=False, **loader_options
    )
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_loader, validation_loader, test_loader, metadata.classes


def build_model(strategy: str, number_of_classes: int, device: torch.device) -> nn.Module:
    if strategy not in STRATEGIES:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, number_of_classes)
    if strategy == "frozen":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    return model.to(device)


def make_optimizer(model: nn.Module, strategy: str, config: RunConfig):
    if strategy == "frozen":
        groups = [{"params": model.fc.parameters(), "lr": config.learning_rate_head}]
    else:
        backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
        groups = [
            {"params": backbone, "lr": config.learning_rate_backbone},
            {"params": model.fc.parameters(), "lr": config.learning_rate_head},
        ]
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor, classes: int) -> torch.Tensor:
    encoded = labels * classes + predictions
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def classification_metrics(
    labels: torch.Tensor, predictions: torch.Tensor, probabilities: torch.Tensor, classes: int
) -> dict[str, float]:
    matrix = confusion_matrix(labels, predictions, classes).to(torch.float64)
    true_positive = matrix.diag()
    precision = true_positive / matrix.sum(dim=0).clamp_min(1)
    recall = true_positive / matrix.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = true_positive.sum() / matrix.sum().clamp_min(1)

    confidence, _ = probabilities.max(dim=1)
    correctness = predictions.eq(labels).to(torch.float32)
    expected_calibration_error = torch.tensor(0.0)
    boundaries = torch.linspace(0, 1, 11)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            expected_calibration_error += selected.float().mean() * (
                correctness[selected].mean() - confidence[selected].mean()
            ).abs()

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "ece_10_bins": float(expected_calibration_error),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, classes: int):
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        total_loss += float(loss) * images.size(0)
        total_items += images.size(0)
        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_probabilities.append(probabilities.cpu())
    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)
    probabilities = torch.cat(all_probabilities)
    metrics = classification_metrics(labels, predictions, probabilities, classes)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def train_strategy(
    strategy: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    classes: int,
    config: RunConfig,
    output_dir: Path,
):
    optimizer = make_optimizer(model, strategy, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_score = -float("inf")
    history: list[dict[str, float]] = []
    checkpoint = output_dir / f"resnet18_{strategy}.pt"
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        if strategy == "frozen":
            # Mantém BatchNorm do backbone em modo de avaliação.
            model.eval()
            model.fc.train()
        running_loss = 0.0
        samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * images.size(0)
            samples += images.size(0)
        scheduler.step()
        validation = evaluate(model, validation_loader, device, classes)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(samples, 1),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(
            f"[{strategy}] época {epoch:02d}/{config.epochs}: "
            f"loss={record['train_loss']:.4f}, macro-F1={validation['macro_f1']:.4f}"
        )
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            torch.save(model.state_dict(), checkpoint)

    elapsed = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with (output_dir / f"history_{strategy}.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    return model, history, elapsed




if __name__ == "__main__":
    print("Módulo de experimento – em construção.")
