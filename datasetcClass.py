import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt


class classDataset(Dataset):
    def __init__(self, csvpath, mode, height, width, mean_std, debug=False):
        """      
        Constructor of Dataset 
        Format of csv file:
        It contains 2 columns
        1. Path to image
        2. Path to mask

        Parameters:
            csvpath (str): Path to the csv file Ex. dataset/steel
            mode (str): Mode of the dataset {'train', 'valid'}
            height (int): height of the image
            width (int): width of the image 
            mean_std (List,List): mean and std of the dataset 
            debug (bool): True to show some sample

        """

        self.csv_file = (
            pd.read_csv(os.path.join(csvpath, mode + ".csv"))
            .iloc[:, :]
            .values
        )
        self.mean_std = mean_std
        self.height = height
        self.width = width
        self.mode = mode
        self.debug = debug

    def _set_seed(self, seed):
        """
        Function used to set seed

        Parameters:
            seed (int): seed

        """

        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        """returns length of CSV file"""
        return len(self.csv_file)

    def __getitem__(self, idx):
        """
        Function used by dataloader to get item contain augmentation, normalization function 

        Parameters:
            idx (int): index of the csv file 

        """
        image = cv2.imread(self.csv_file[idx, 0], cv2.IMREAD_COLOR)
        # print(self.csv_file[idx, 1])
        label = (cv2.imread(
            self.csv_file[idx, 1], cv2.IMREAD_GRAYSCALE))
        label[label == 100] = 1

        if (
            image.shape[1] == self.width
            and image.shape[0] == self.height
            and label.shape[1] == self.width
            and label.shape[0] == self.height
        ):
            pass
        else:

            image = cv2.resize(
                image, (self.width, self.height))
            label = cv2.resize(
                label,
                (self.width, self.height),
                cv2.INTER_NEAREST,
            )

        if self.mode == 'train':
            image, label = self.rand_crop(
                image, label, crop_size=(image.shape[0]//2, image.shape[1]//2))

        if self.debug:
            self.show_sample(image, label)
        transformation = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()])

        image = transforms.Normalize(
            mean=self.mean_std[0], std=self.mean_std[1])(transformation(image))
        label = torch.from_numpy(label)

        if self.mode == 'train':
            # applying transforms
            augment = [
                transforms.RandomCrop(200),
                transforms.RandomHorizontalFlip(0.5)
            ]
            tfs = transforms.Compose(augment)
            # seed = random.randint(0, 2**32)
            # self._set_seed(seed)
            image = tfs(image)
            # self._set_seed(seed)
            label = tfs(label)

        sample = {
            "image": image,
            "label": label,
            "img_name": self.csv_file[idx, 0].split("/")[-1],
        }

        return sample

    def rand_crop(self, image, label, crop_size):
        """
        Spatial augmentation technique: Crop the part of the image

        Parameters:
            image (np.ndarray): image
            label (np.ndarray): label
            crop_size (tuple) : (height//2,width//2) of the original image

        Returns:
            image (np.ndarray): Cropped image
            label (np.ndarray): Cropped label

        """

        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, crop_size, (0.0, 0.0, 0.0))

        new_h, new_w = image.shape[:-1]
        x = random.randint(0, new_w - crop_size[1])
        y = random.randint(0, new_h - crop_size[0])

        image = image[y: y + crop_size[0], x: x + crop_size[1]]

        if label is not None:
            label = self.pad_image(
                label, h, w, crop_size, (255,))
            label = label[y: y + crop_size[0], x: x + crop_size[1]]
        return image, label

    def pad_image(self, image, h, w, size, padvalue):
        """
        Spatial augmentation technique: Crop the part of the image

        Parameters:
            image (np.ndarray): image
            h (int): height of the original image 
            w (int): width of the original image 
            crop_size (tuple): size of the crop image (height,width)
            padvalue (tuple) : Pading value 

        Returns:
            image (np.ndarray): Padded image

        """
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue
            )
        return pad_image

    def show_sample(self, image, label):
        """
        Show sample of the dataset 

        Parameters:
            image (np.ndarray): image
            label (np.ndarray): label

        """

        plt.imshow(image)
        plt.show()
        plt.imshow(label)
        plt.show()
