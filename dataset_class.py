import os
import random
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetClass(Dataset):
    """
    Класс для работы с данными из csv файла
    """

    def __init__(self,
                 csvpath: str,
                 mode: str,
                 height: int,
                 width: int,
                 mean_std: Tuple[List, List],
                 debug: bool = False):
        """
        Конструктор класса датасет
        Формат csv файла должен быть следующим:
        Содержит 2 поля
        1. Путь к оригинальному изображению
        2. Путь к соотвествующей маске

        :param  csvpath (str): путь к csv файлу
        :param  mode (str): режим работы. Принимает любое из 2 значений {'train', 'valid'}
        :param  height (int): высота изображения
        :param  width (int): ширина изображения
        :param  mean_std (List,List): средние и дисперсия датасета
        :param  debug (bool): режим отладки
        :return:
        """

        self.csv_file = (pd.read_csv(os.path.join(csvpath, mode + ".csv")).iloc[:, :].values)
        self.mean_std = mean_std
        self.height = height
        self.width = width
        self.mode = mode
        self.debug = debug

    def _set_seed(self, seed: int):
        """
        Функция установки сида

        :param seed (int): seed
        :return:
        """

        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        """Возвращает длинну csv файла"""
        return len(self.csv_file)

    def __getitem__(self, idx: int) -> dict:
        """
        Функция, используемая загрузчиком данных для получения элемента, содержащего функцию аугментации и нормализации

        :param idx (int): индекс в csv файле
        :return: словарь из исходного изображения и соотвествующим ему названием файла и его маской
        """
        image = cv2.imread(self.csv_file[idx, 0], cv2.IMREAD_COLOR)
        label = (cv2.imread(
            self.csv_file[idx, 1], cv2.IMREAD_GRAYSCALE))
        label[label == 100] = 1

        # если размер изображения не подходит под фомат, то принудительно ресайзим изображение
        if not (
                image.shape[1] == self.width
                and image.shape[0] == self.height
                and label.shape[1] == self.width
                and label.shape[0] == self.height
        ):
            image = cv2.resize(
                image, (self.width, self.height))
            label = cv2.resize(
                label,
                (self.width, self.height),
                cv2.INTER_NEAREST,
            )

        if self.mode == 'train':
            image, label = self.rand_crop(
                image, label, crop_size=(image.shape[0] // 2, image.shape[1] // 2))

        if self.debug:
            self.show_sample(image, label)

        transformation = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()])

        image = transforms.Normalize(
            mean=self.mean_std[0], std=self.mean_std[1])(transformation(image))
        label = torch.from_numpy(label)

        if self.mode == 'train':
            augment = [
                transforms.RandomHorizontalFlip(0.5)
            ]
            tfs = transforms.Compose(augment)
            image = tfs(image)
            label = tfs(label)

        sample = {
            "image": image,
            "label": label,
            "img_name": self.csv_file[idx, 0].split("/")[-1],
        }

        return sample

    def rand_crop(self,
                  image: np.ndarray,
                  label: np.ndarray,
                  crop_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Функция аугментации случайного кропом

        :param image (np.ndarray): массив пикселей представляющих исходное изображение
        :param label (np.ndarray): массив пикселей представляющих ground true маску
        :param crop_size (tuple) : размер кропа. Обычно равен (height//2,width//2)  оригинального изображения
        :return: кортеж из покропленного изображения и соотвествующией ему маски
        """

        height, weigth = image.shape[:-1]
        image = self._pad_image(image, height, weigth, crop_size, (0.0, 0.0, 0.0))

        new_h, new_w = image.shape[:-1]
        x = random.randint(0, new_w - crop_size[1])
        y = random.randint(0, new_h - crop_size[0])

        image = image[y: y + crop_size[0], x: x + crop_size[1]]

        if label is not None:
            label = self._pad_image(
                label, height, weigth, crop_size, (255,))
            label = label[y: y + crop_size[0], x: x + crop_size[1]]
        return image, label

    @classmethod
    def _pad_image(cls,
                   image: np.ndarray,
                   height: int,
                   weight: int,
                   crop_size: Tuple[int, int],
                   padvalue) -> np.ndarray:
        """
        Функция заполнения кроп изображения

        :param image (np.ndarray): массив пикселей представляющих исходное изображение
        :param height (int): height of the original image
        :param weight (int): width of the original image
        :param crop_size (tuple) : размер кропа. Обычно равен (height//2,width//2)  оригинального изображения
        :param padvalue (tuple) : значение заполнитель
        :return: изображение после заполнения
        """

        pad_image = image.copy()
        pad_h = max(crop_size[0] - height, 0)
        pad_w = max(crop_size[1] - weight, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue
            )
        return pad_image

    @classmethod
    def show_sample(cls, image: np.ndarray, label: np.ndarray):
        """
        Функция рисования изображения и маски

        :param image: массив пикселей представляющих исходное изображение
        :param label: массив пикселей представляющих ground true маску
        :return:
        """

        plt.imshow(label)
        plt.show()
        plt.imshow(image)
        plt.show()
