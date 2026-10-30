"""Подготовка отдельных опытов LeNet; сами аугментации задаются в ноутбуке."""

import random
import re
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from aim import Run
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

from utils.data_utils import AugmentedDataset, create_dataloader, standardize_batch
from utils.train_utils import train


class LeNet5(nn.Module):
    """Архитектура из LeNet.ipynb для RGB-изображений 32×32 и 10 классов"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def _weights_path(directory: Path, name: str, epochs: int) -> Path:
    """повторы имён для весов получают суффиксы _2, _3 и далее."""
    name = name.replace("Без аугментаций", "no_augmentation")
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "experiment"
    stem = f"lenet_{name}_{epochs}epochs"
    path = directory / f"{stem}.pth"
    repeat = 2
    while path.exists():
        path = directory / f"{stem}_{repeat}.pth"
        repeat += 1
    return path


def new_results_dir(root: Path) -> Path:
    """Отдельная папка для серии, чтобы не смешивать разные настройки обучения."""
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def show_examples(dataset: Dataset, transform, count: int = 3) -> None:
    """Показывает преобразования с p=1 на копии политики, не меняя обучение."""
    preview = deepcopy(transform)
    pending, visited = [preview], set()
    while pending:
        operation = pending.pop()
        if id(operation) in visited:
            continue
        visited.add(id(operation))
        if isinstance(operation, (v2.RandomApply, v2.RandomHorizontalFlip, v2.RandomErasing)):
            operation.p = 1.0
        pending.extend(getattr(operation, "transforms", ()))
        if isinstance(operation, nn.Module):
            pending.extend(operation.children())
    count = min(count, len(dataset))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(52)
        fig, axes = plt.subplots(2, count, figsize=(2.5 * count, 4), squeeze=False)
        for col in range(count):
            original, _ = dataset[col]
            augmented = preview(original.clone())
            for row, (image, title) in enumerate(((original, "Исходное"), (augmented, "После аугментации"))):
                axes[row, col].imshow(image.permute(1, 2, 0).numpy())
                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis("off")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def select_transforms(names: list[str], catalog: dict, limit: int = 4) -> dict:
    """Проверяет небольшой список выбранных опытов до начала обучения."""
    if len(names) != len(set(names)):
        raise ValueError("Названия выбранных операций не должны повторяться")
    if len(names) > limit:
        raise ValueError(f"Выбери не больше {limit} вариантов")
    unknown = set(names) - catalog.keys()
    if unknown:
        raise ValueError(f"Неизвестные или ещё не проверенные варианты: {', '.join(sorted(unknown))}")
    return {name: catalog[name] for name in names}


def cumulative_combinations(ranked: list[str], catalog: dict) -> dict:
    """Сочетания топ-2/3/4; порядок применения всегда задан catalog, а не рейтингом."""
    select_transforms(ranked, catalog)
    return {
        " + ".join(ranked[:size]): v2.Compose([
            transform for name, transform in catalog.items() if name in ranked[:size]
        ])
        for size in range(2, len(ranked) + 1)
    }


class _TrainingLog:
    """Передаёт метрики в Aim и сохраняет кривую для сравнения коротких запусков"""

    def __init__(self, run, epochs: int):
        self.run, self.epochs = run, epochs
        self.rows = {}

    def track(self, value, *, name, step, context):
        self.run.track(value, name=name, step=step, context=context)
        self.rows.setdefault(step, {"epoch": step})[f"{context['subset']}_{name}"] = float(value)
        if name == "accuracy" and context["subset"] == "val" and (step == 1 or step % 25 == 0 or step == self.epochs):
            print(f"Эпоха {step}/{self.epochs}: val_accuracy={value:.4f}", flush=True)

    def __setitem__(self, key, value):
        self.run[key] = value

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows.values())


def _budget_results(results_dir: Path, epochs: int, stage: str | tuple[str, ...] | None) -> pd.DataFrame:
    path = Path(results_dir) / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    results = pd.read_csv(path)
    if not {"epoch_budget", "stage"}.issubset(results.columns):
        raise ValueError("В старом CSV нет бюджета эпох. Открой его отдельно через pd.read_csv")
    results = results[results["epoch_budget"] == epochs]
    if stage is not None:
        stages = (stage,) if isinstance(stage, str) else stage
        results = results[results["stage"].isin(stages)]
    return results.drop_duplicates(["stage", "variant_name"], keep="last")


def completed_transforms(
    results_dir: Path, catalog: dict, *, epochs: int, stage: str | tuple[str, ...],
) -> dict:
    """Допускает к отбору только варианты, прошедшие весь предыдущий бюджет."""
    results = _budget_results(results_dir, epochs, stage)
    names = set(results.loc[results["last_epoch"] == epochs, "variant_name"]) if not results.empty else set()
    return {name: transform for name, transform in catalog.items() if name in names}


def experiment_table(results_dir: Path, *, epochs: int, stage: str | None = None) -> pd.DataFrame:
    """Сравнивает только один бюджет, используя последние завершённые запуски."""
    columns = ["Вариант", "Лучшая val, %", "Val в конце, %", "Тренд val, п.п.", "Лучшая эпоха", "Эпох"]
    results = _budget_results(results_dir, epochs, stage)
    if results.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({
        "Вариант": results["variant_name"],
        "Лучшая val, %": results["best_val_accuracy"] * 100,
        "Val в конце, %": results["val_accuracy_last"] * 100,
        "Тренд val, п.п.": results["val_tail_gain_pp"],
        "Лучшая эпоха": results["best_epoch"],
        "Эпох": results["last_epoch"],
    }).sort_values("Лучшая val, %", ascending=False, kind="stable").round(3)


def plot_learning_curves(results_dir: Path, *, epochs: int, stage: str | None = None) -> None:
    """Показывает validation-кривые только для указанного бюджета эпох."""
    results = _budget_results(results_dir, epochs, stage)
    if results.empty:
        print(f"Пока нет завершённых запусков с бюджетом {epochs} эпох.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for row in results.itertuples(index=False):
        history = pd.read_csv(row.history_path)
        ax.plot(history["epoch"], history["val_accuracy"] * 100, label=row.variant_name)
    ax.set(xlabel="Эпоха", ylabel="Validation accuracy, %", title=f"Бюджет: {epochs} эпох")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_lenet_experiment(
    name: str,
    transform,
    *,
    ds_train: Dataset,
    ds_val: Dataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device | str,
    results_dir: Path,
    project_root: Path,
    seed: int = 52,
    batch_size: int = 512,
    num_workers: int = 4,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int | None = None,
    stage: str = "single",
) -> pd.DataFrame:
    """Обучает новую модель, сохраняет лучший checkpoint и дописывает results.csv.

    ds_train содержит uint8 без стандартизации. ds_val уже стандартизирован.
    Повтор опыта сохраняет новый run и новые веса, оставляя прежние результаты.
    При patience=None обучаем ровно epochs эпох. Каждый запуск начинается заново
    """
    if epochs < 1 or len(ds_train) == 0 or len(ds_val) == 0:
        raise ValueError("Нужны непустые train/val и хотя бы одна эпоха обучения")

    project_root, results_dir = Path(project_root).resolve(), Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    run = Run(repo=str(project_root), experiment="lenet-augmentation")
    recorder = _TrainingLog(run, epochs)
    history_path = results_dir / f"{run.hash}_history.csv"
    weights_path = _weights_path(
        project_root / "models/lenet_augmentation" / results_dir.name, name, epochs,
    )
    try:
        run.name = f"{name}-{stage}-{epochs}epochs-seed{seed}"
        run["series_id"] = results_dir.name
        run["variant_name"] = name
        run["stage"] = stage
        run["epoch_budget"] = epochs
        run["status"] = "running"
        run["hparams"] = {
            "architecture": "LeNet5", "optimizer": "Adam", "seed": seed,
            "batch_size": batch_size, "num_workers": num_workers,
            "epochs": epochs, "patience": patience, "early_stopping": patience is not None,
            "stage": stage,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "augmentation": str(transform), "device": str(device),
            "mean": mean.detach().cpu().tolist(), "std": std.detach().cpu().tolist(),
            "train_count": len(ds_train), "val_count": len(ds_val),
        }

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model = LeNet5().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        preprocess = partial(standardize_batch, mean=mean.to(device), std=std.to(device))
        dl_train = create_dataloader(
            AugmentedDataset(ds_train, transform), batch_size=batch_size,
            shuffle=True, random_seed=seed, num_workers=num_workers,
        )
        dl_val = create_dataloader(ds_val, batch_size=batch_size, random_seed=seed)
        print(f"{name}: {epochs} эпох, этап {stage}, seed {seed}. Метрики — в Aim.", flush=True)
        started = perf_counter()
        train(
            model, optimizer, nn.CrossEntropyLoss(), dl_train, dl_val, weights_path, device,
            EPOCH_COUNT=epochs, patience=patience if patience is not None else epochs + 1,
            plot=False, train_preprocess=preprocess, aim_run=recorder,
        )
        values = recorder.history()["val_accuracy"]
        tail_gain = (values.iloc[-10:].mean() - values.iloc[-20:-10].mean()) * 100 if len(values) >= 20 else float("nan")
        row = {
            "variant_name": name, "seed": seed, **run["results"],
            "training_seconds": perf_counter() - started, "run_hash": run.hash,
            "epoch_budget": epochs, "stage": stage,
            "val_accuracy_last": float(values.iloc[-1]), "val_tail_gain_pp": float(tail_gain),
            "history_path": str(history_path),
        }
        results_path = results_dir / "results.csv"
        previous = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
        results = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
        temporary_path = results_dir / "results.tmp.csv"
        results.to_csv(temporary_path, index=False)
        temporary_path.replace(results_path)
        run["results"] = row
        run["status"] = "completed"
        return pd.DataFrame([{
            "Вариант": name,
            "Val accuracy, %": row["best_val_accuracy"] * 100,
            "Лучшая эпоха": row["best_epoch"],
        }]).round(2)
    except BaseException as error:
        run["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        try:
            history = recorder.history()
            if not history.empty:
                history.to_csv(history_path, index=False)
        finally:
            run.close()
