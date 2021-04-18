# test/inference script
import numpy as np
import torch
import argparse
import os
import cv2
from models.hrnet import hrnet
from tqdm import tqdm
from torchvision import transforms
import math
from utils.util import stiching
import shutil


def preprocess_image(image):
    """[code to normalize and convert image to torch tensor]

    Args:
        image ([numpy array]): [image read by opencv]

    Returns:
        [type]: [normalized torch tensor 4d]
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
        help="Path where all the test images are located or you can give path to video, it will break into each frame and write as a video",
    )

    parser.add_argument(
        "--path_to_weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path to save checkpoints and wandb, final output path will be this path + wandbexperiment name so the output_dir should be root directory",
    )

    parser.add_argument("--n_classes", type=int, default="2", help="number of classes")

    args = parser.parse_args()

    model = hrnet(n_classes=args.n_classes)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(args.path_to_weights)["state_dict"])
    model.eval()

    os.makedirs(os.path.join(args.output_dir, "pred_mask"), exist_ok=True)
    tile_size = (869, 1302)
    offset = (869, 1302)

    for image_path in tqdm(os.listdir(args.path_to_images)):

        path = os.path.join(args.path_to_images, image_path)
        img = cv2.imread(path)
        data = path.split("/")
        os.makedirs(os.path.join(args.output_dir, "tiling_output"), exist_ok=True)
        img_shape = img.shape

        for i in range(int(math.ceil(img_shape[0] / (offset[0] * 1.0)))):
            for j in range(int(math.ceil(img_shape[1] / (offset[1] * 1.0)))):

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
                    # * 255
                )

                cv2.imwrite(
                    args.output_dir
                    + "/tiling_output/"
                    + data[-1][:-4]
                    + "_"
                    + str(i)
                    + "_"
                    + str(j)
                    + data[-1][-4:],
                    prediction,
                )

        prediction = stiching(
            os.path.join(args.output_dir, "tiling_output"),
            img_shape[0],
            img_shape[1],
            tile_size,
            offset,
        )

        error = (np.sum(prediction) / (img_shape[0] * img_shape[1])) * 100

        print("Percentage of damage in {} is {:.3f}%".format(image_path, error))

        shutil.rmtree(os.path.join(args.output_dir, "tiling_output"))

        cv2.imwrite(
            os.path.join(args.output_dir, "pred_mask", image_path), prediction * 255
        )


if __name__ == "__main__":
    main()

# python inference.py --path_to_images dataset/steel/image --path_to_weights weigths/hrnetv2_hrnet18_steel_dataset_47.pth --output_dir output
