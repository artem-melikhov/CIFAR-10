
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import Subset

def images_to_tensor(labels_path: str, 
                     images_path: str, 
                     channels: int = 3
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    
    df = pd.read_csv(labels_path) 
    # читаем метки правильных классов для фото, 
    # они сопоставлены с ними по image_id, который указан в названии файла картинки

    to_tensor = ToTensor()
    img_tensors = []

    for image_id in df["id"]: # по id ищем для метки фото, переводим в тензор и добавляем в список
        img_path = os.path.join(images_path, f"{image_id}.png")
        # Читаем сами картинки, переводим их в тензор из чисел с помощью библиотеки PIL
        img = Image.open(img_path).convert("RGB" if channels == 3 else "L")
        img_tensors.append(to_tensor(img))

    # делаем из списка тензоров одинаковой размерности один тензор для всех фото
    X_train = torch.stack(img_tensors)

    # кодируем названия классов цифрами от 0 до 9
    classes = sorted(df['label'].unique())
    classes_nums = {c: i for i, c in enumerate(classes)}
    df['label'] = df['label'].map(classes_nums)

    # превращаем метки в тензор из номеров классов
    y_train = torch.tensor(df['label'].tolist())

    return X_train, y_train

def train_dataset_split(ds_train: TensorDataset, train_size: float) -> tuple[Subset, Subset]:
    n = len(ds_train)
    train_size = int(train_size * n)
    test_size = n - train_size

    g = torch.Generator().manual_seed(52) # фиксируем рандом для random_split
    ds_train, ds_test = random_split(ds_train, [train_size, test_size], generator=g)
    return ds_train, ds_test

def create_dataloader(ds: TensorDataset, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    g = torch.Generator().manual_seed(52) # фиксируем рандом для shuffle
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)
    return dl
