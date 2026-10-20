
import torch
from PIL import Image
from torchvision.transforms import PILToTensor, v2
import os
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch.utils.data import Subset


class AutoAugumentationDataset(Dataset):
    """Применяет AutoAugment для CIFAR-10 при каждой выдаче изображения.
    При каждом обращении к изображению выбирается случайный вариант аугментации,
    а метка класса остаётся прежней.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.augment = v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]

        # Загрузчик и AutoAugment работают с uint8 [0, 255].
        return self.augment(image), label


def calculate_mean_std(ds_train: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    """Считает mean и std исходного uint8 train в шкале 0–255."""

    # берём только train, чтобы картинки из val не влияли на статистики
    X_train = ds_train.dataset.tensors[0][ds_train.indices].float()

    mean = X_train.mean(dim=(0, 2, 3))
    std = X_train.std(dim=(0, 2, 3), correction=0)

    return mean, std


class StandardizationDataset(Dataset):
    """Стандартизует картинки по mean и std обучающей выборки."""

    def __init__(self, dataset: Dataset, mean: torch.Tensor, std: torch.Tensor):
        self.dataset = dataset
        # добавляем размеры высоты и ширины, чтобы значения применились к каждому каналу
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]

        if isinstance(data, tuple):
            image, label = data
            return (image.float() - self.mean) / self.std, label

        return (data.float() - self.mean) / self.std


class TestImagesDataset(Dataset):
    """Подгружает тестовые картинки в порядке sampleSubmission.csv."""

    def __init__(
        self, images_path: str, sample_path: str, channels: int = 3,
    ):
        self.images_path = images_path
        self.ids = pd.read_csv(sample_path)["id"].tolist()
        self.to_tensor = PILToTensor()
        self.mode = "RGB" if channels == 3 else "L"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_id = self.ids[index]
        img_path = os.path.join(self.images_path, f"{image_id}.png")

        with Image.open(img_path) as img:
            return self.to_tensor(img.convert(self.mode))


def images_to_tensor(
    images_path: str,
    csv_path: str,
    channels: int = 3,
    has_labels: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Загружает изображения в RAM как uint8 [0, 255]."""

    # читаем метки правильных классов для фото они сопоставлены с ними по image_id, 
    # который указан в названии файла картинки(для случая когда эти метки есть, то есть train), 
    # иначе читаем id которые идут в файле sample_submission, 
    # чтобы дать предсказания для картинок в том же порядке
    df = pd.read_csv(csv_path)
    to_tensor = PILToTensor()
    img_tensors = []

    for image_id in df["id"]:
        img_path = os.path.join(images_path, f"{image_id}.png")

        with Image.open(img_path) as img:
            img_tensors.append(
                to_tensor(img.convert("RGB" if channels == 3 else "L"))
            )

    X = torch.stack(img_tensors)

    if not has_labels:
        return X

    # кодируем названия классов цифрами от 0 до 9
    classes = sorted(df["label"].unique())
    classes_nums = {name: i for i, name in enumerate(classes)}

    # превращаем метки в тензор из номеров классов
    y = torch.tensor(df["label"].map(classes_nums).tolist())

    return X, y

def train_dataset_split(ds_train: TensorDataset, train_size: float, random_seed: int = 52) -> tuple[Subset, Subset]:
    n = len(ds_train)
    train_size = int(train_size * n)
    test_size = n - train_size

    g = torch.Generator().manual_seed(random_seed) # фиксируем рандом для random_split
    ds_train, ds_test = random_split(ds_train, [train_size, test_size], generator=g)
    return ds_train, ds_test

def create_dataloader(
    ds: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    random_seed: int = 52,
    num_workers: int = 0,
) -> DataLoader:
    """При num_workers > 0 готовит батчи в постоянных процессах на CPU."""
    g = torch.Generator().manual_seed(random_seed) # фиксируем рандом для shuffle
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
