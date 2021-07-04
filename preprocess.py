import argparse
import glob
import os
import time
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils.util import tiling

np.random.seed(int(time.time()))


def get_image_names(folder_name: str) -> List[str]:
    return [os.path.splitext(name)[0] for name in os.listdir(folder_name) if
            os.path.isfile(os.path.join(folder_name, name))]


def create_empty_png_file(dir_path: str, filename: str, width: int, height: int):
    """
    Создает пустой png файл с заданными высотой и шириной
    :param dir_path: название директории для сохранения файла
    :param filename: имя файла
    :param width: ширина
    :param height: высота
    :return:
    """
    image = Image.new('RGB', (width, height))
    image.save(os.path.join(dir_path, f"{filename}.png"), "PNG")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        default='dataset/steel/image',
        help="Путь к оригинальным изображениям",
    )

    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        default='dataset/steel/label',
        help="Путь к маскам",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="dataset/steel",
        help="Путь к выходному файлу csv, обработанным изображениям и маскам",
    )

    args = parser.parse_args()

    image_filename_list = get_image_names(args.image_dir)
    label_filename_list = get_image_names(args.label_dir)

    # Проверка отсутствия vfcrb в случае хорошей стропы (то есть стропа без дефектов)
    missing_label_set = set(image_filename_list) - set(label_filename_list)
    if missing_label_set:
        [create_empty_png_file(args.label_dir, filename, 5208, 3476) for filename in missing_label_set]

    # Создаем массив numpy для сортировки тестовых и обучающих изображений
    label = np.asarray(sorted(glob.glob(args.label_dir + '/*.png')))
    train = np.asarray([i for i in sorted(glob.glob(args.image_dir + '/*.jpg'))
                        if i.replace('jpg', 'png').replace('image', 'label') in label])
    test = np.asarray([i for i in sorted(glob.glob(args.image_dir + '/*.jpg'))
                       if i.replace('jpg', 'png').replace('image', 'label') not in label])

    # TODO добавить функцию вставки на неаномальные изображения

    # Перемешиваем массив
    idx = np.random.permutation(len(label))
    train_images = train[idx[:int(len(idx) * args.train_val_split)]]
    train_labels = label[idx[:int(len(idx) * args.train_val_split)]]
    val_images = train[idx[int(len(idx) * args.train_val_split):]]
    val_labels = label[idx[int(len(idx) * args.train_val_split):]]

    # Создаем покропленные изображения для теста и сохраняем их
    for ti, tl in tqdm(zip(train_images, train_labels), total=len(train_images)):
        tiling(ti, os.path.join(args.outdir, 'train_images'),
               preprocess=True, save=True)
        tiling(tl, os.path.join(args.outdir, 'train_labels'),
               preprocess=True, save=True)

    for vi, vl in tqdm(zip(val_images, val_labels), total=len(val_images)):
        tiling(vi, os.path.join(args.outdir, 'val_images'),
               preprocess=True, save=True)
        tiling(vl, os.path.join(args.outdir, 'val_labels'),
               preprocess=True, save=True)

    train_images = np.asarray(
        sorted(glob.glob(os.path.join(args.outdir, 'train_images', '*.jpg'))))
    train_labels = np.asarray(
        sorted(glob.glob(os.path.join(args.outdir, 'train_labels', '*.png'))))
    val_images = np.asarray(
        sorted(glob.glob(os.path.join(args.outdir, 'val_images', '*.jpg'))))
    val_labels = np.asarray(
        sorted(glob.glob(os.path.join(args.outdir, 'val_labels', '*.png'))))

    train_csv = pd.DataFrame(np.concatenate(
        [train_images.reshape(-1, 1), train_labels.reshape(-1, 1)], axis=1))
    val_csv = pd.DataFrame(np.concatenate(
        [val_images.reshape(-1, 1), val_labels.reshape(-1, 1)], axis=1))

    test_csv = pd.DataFrame({'test': test})

    train_csv.to_csv(
        args.outdir + '/train.csv', index=False)
    val_csv.to_csv(
        args.outdir + '/valid.csv', index=False)
    test_csv.to_csv(args.outdir + '/test.csv', index=False)


if __name__ == "__main__":
    main()
