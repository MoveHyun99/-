"""A lightweight CNN training script using PyTorch and Torchvision.

This module defines a simple convolutional neural network suitable for
image-classification tasks on small datasets such as CIFAR-10 or
custom image folders. It also includes helper utilities for preparing
train/validation dataloaders and running a basic training loop.

Example
-------
>>> from cnn_model import SimpleCNN, TrainingConfig, build_cifar10_loaders, train
>>> model = SimpleCNN(num_classes=10)
>>> config = TrainingConfig(epochs=1, batch_size=64)
>>> train_loader, val_loader = build_cifar10_loaders("./data", config.batch_size)
>>> metrics = train(model, train_loader, val_loader, config)
>>> print(metrics)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class TrainingConfig:
    """Configuration values for the training loop."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 2
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_every: int = 50


class SimpleCNN(nn.Module):
    """A straightforward CNN architecture for small image datasets."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train/validation loaders with standard transforms."""

    data_root = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean()


def train_epoch(
    model: nn.Module,
    data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_every: int = 50,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    step_count = 0

    for step_count, (images, labels) in enumerate(data, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(logits, labels).item()
        running_loss += loss.item()
        running_acc += batch_acc

        if step_count % log_every == 0:
            mean_loss = running_loss / step_count
            mean_acc = running_acc / step_count
            print(f"[train] step {step_count}: loss={mean_loss:.4f} acc={mean_acc:.4f}")

    steps = max(step_count, 1)
    return {"loss": running_loss / steps, "accuracy": running_acc / steps}


def evaluate(
    model: nn.Module,
    data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    steps = 0

    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item()
            acc_sum += accuracy(logits, labels).item()
            steps += 1

    steps = max(steps, 1)
    return {"loss": loss_sum / steps, "accuracy": acc_sum / steps}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> Dict[str, Dict[str, float]]:
    """Run the training loop and return metrics for the final epoch."""

    device = torch.device(config.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, Dict[str, float]] = {}
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            log_every=config.log_every,
        )
        val_metrics = evaluate(model, val_loader, criterion, device=device)
        history[f"epoch_{epoch}"] = {"train_loss": train_metrics["loss"], "train_acc": train_metrics["accuracy"],
                                    "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]}
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

    return history


if __name__ == "__main__":
    config = TrainingConfig(epochs=1, batch_size=64)
    train_loader, val_loader = build_cifar10_loaders("./data", batch_size=config.batch_size)
    model = SimpleCNN(num_classes=10)
    train(model, train_loader, val_loader, config)
