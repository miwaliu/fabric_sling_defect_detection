import argparse
import math
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from constants import TILE_SIZE
from models.sling_model import sling_model
from utils.util import stiching


def preprocess_image(image: np.ndarray):
    """
    Функция нормализации и конвертирования изображения в PyTorch тензор

    :param image (np.ndarray): массив пикселей представляющих исходное изображение
    :return: нормализованный PyTorch тензор
    """
    transformation = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]
    )
    image = transforms.Normalize(
        mean=[0.68577555, 0.65262334, 0.59697408],
        std=[0.12157939, 0.1254269, 0.13619352],
    )(transformation(image))
    return torch.unsqueeze(image, dim=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_images",
        type=str,
        required=True,
        default="./",
        help="Путь, по которому расположены все тестовые изображения",
    )
    parser.add_argument(
        "--path_to_weights",
        type=str,
        required=True,
        default="./",
        help="Путь к весам",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Путь для вывода результата работы",
    )
    parser.add_argument("--n_classes", type=int, default="2", help="Количество классов")

    args = parser.parse_args()

    model = sling_model(n_classes=args.n_classes)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(args.path_to_weights)["state_dict"])
    model.eval()

    os.makedirs(os.path.join(args.output_dir, "pred_mask"), exist_ok=True)
    tile_size = TILE_SIZE
    offset = TILE_SIZE

    for image_name in tqdm(os.listdir(args.path_to_images)):

        path = os.path.join(args.path_to_images, image_name)
        img = cv2.imread(path)
        data = path.split("/")
        os.makedirs(os.path.join(args.output_dir, "tiling_output"), exist_ok=True)
        img_shape = img.shape

        for i in range(int(math.ceil(img_shape[0] / (offset[0] * 1.0)))):
            for j in range(int(math.ceil(img_shape[1] / (offset[1] * 1.0)))):

                # делам кроп изображения
                cropped_img = img[
                              offset[0] * i: min(offset[0] * i + tile_size[0], img_shape[0]),
                              offset[1] * j: min(offset[1] * j + tile_size[1], img_shape[1]),
                              ]

                if cropped_img.shape[0:2] != tile_size:
                    if (
                            cropped_img.shape[0] != tile_size[0]
                            and cropped_img.shape[1] != tile_size[1]
                    ):
                        cropped_img = img[
                                      img_shape[0] - tile_size[0]: img_shape[0],
                                      img_shape[1] - tile_size[1]: img_shape[1],
                                      ]
                    elif cropped_img.shape[0] != tile_size[0]:
                        cropped_img = img[
                                      img_shape[0] - tile_size[0]: img_shape[0],
                                      offset[1] * j: min(offset[1] * j + tile_size[1], img_shape[1]),
                                      ]
                    elif cropped_img.shape[1] != tile_size[1]:
                        cropped_img = img[
                                      offset[0] * i: min(offset[0] * i + tile_size[0], img_shape[0]),
                                      img_shape[1] - tile_size[1]: img_shape[1],
                                      ]

                image = preprocess_image(cropped_img)

                if torch.cuda.is_available():
                    image = image.cuda()

                with torch.no_grad():
                    prediction = model(image, (tile_size[0], tile_size[1]))

                prediction = (
                    torch.argmax(prediction["output"], dim=1)
                        .cpu()
                        .squeeze(dim=0)
                        .numpy()
                        .astype(np.uint8)
                )

                cv2.imwrite(
                    f"{args.output_dir}/tiling_output/{data[-1][:-4]}_{str(i)}_{str(j)}{data[-1][-4:]}{prediction}")

        prediction = stiching(
            os.path.join(args.output_dir, "tiling_output"),
            img_shape[0],
            img_shape[1],
            tile_size,
            offset,
        )

        error = (np.sum(prediction) / (img_shape[0] * img_shape[1])) * 100

        print("\n\nPercentage of damage in {} is {:.3f}%".format(image_name, error))

        shutil.rmtree(os.path.join(args.output_dir, "tiling_output"))

        # добавляем процент повреждений в название выходного файла
        filename, file_extension = os.path.splitext(image_name)
        image_name = "{}_Perc_{:.3f}%{}".format(filename, error, file_extension)
        cv2.imwrite(
            os.path.join(args.output_dir, "pred_mask", image_name), prediction * 255
        )


if __name__ == "__main__":
    main()
