
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import Subset

def images_to_tensor(
    images_path: str,
    csv_path: str,
    channels: int = 3,
    has_labels: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    # читаем метки правильных классов для фото они сопоставлены с ними по image_id, 
    # который указан в названии файла картинки(для случая когда эти метки есть, то есть train), 
    # иначе читаем id которые идут в файле sample_submission, 
    # чтобы дать предсказания для картинок в том же порядке
    df = pd.read_csv(csv_path)
    to_tensor = ToTensor()
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

def create_dataloader(ds: TensorDataset, batch_size: int = 128, shuffle: bool = False, random_seed: int = 52) -> DataLoader:
    g = torch.Generator().manual_seed(random_seed) # фиксируем рандом для shuffle
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)
    return dl
