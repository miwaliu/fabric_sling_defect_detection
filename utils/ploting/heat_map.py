import os
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


class HeatMap:
    def __init__(self, image, heat_map, gaussian_std=10):
        # if image is numpy array
        if isinstance(image, np.ndarray):
            self.image_height = image.shape[0]
            self.image_width = image.shape[1]
            self.image = image
        else:
            # PIL open the image path, record the height and width
            image = Image.open(image)
            self.image_width, self.image_height = image.size
            self.image = image

        # Convert numpy heat_map values into image formate for easy upscale
        # Resize the heat_map to the size of the input image
        # Apply the gausian filter for smoothing
        # Convert back to numpy
        heatmap_image = Image.fromarray(heat_map)
        heatmap_image_resized = heatmap_image.resize((self.image_width, self.image_height))
        heatmap_image_resized = ndimage.gaussian_filter(heatmap_image_resized,
                                                        sigma=(gaussian_std, gaussian_std),
                                                        order=0)
        heatmap_image_resized = np.asarray(heatmap_image_resized)
        self.heat_map = heatmap_image_resized

    # Plot the figure
    def plot(self, transparency=0.7, color_map='bwr',
             show_axis=False, show_colorbar=False):

        dpi = mpl.rcParams['figure.dpi']
        figsize = self.image_width / float(dpi), self.image_height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Plot the heatmap
        if not show_axis:
            ax.axis('off')
        ax.imshow(self.image)
        ax.imshow(self.heat_map, alpha=transparency, cmap=color_map)
        if show_colorbar:
            ax.colorbar()
        plt.show()

    def save(self, filename, format='png', save_path=os.getcwd(),
             transparency=0.7, color_map='bwr',
             show_axis=False, show_colorbar=False, **kwargs):
        dpi = mpl.rcParams['figure.dpi']
        figsize = self.image_width / float(dpi), self.image_height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Plot the heatmap
        if not show_axis:
            ax.axis('off')
        ax.imshow(self.image)
        ax.imshow(self.heat_map, alpha=transparency, cmap=color_map)
        if show_colorbar:
            ax.colorbar()
        plt.savefig(os.path.join(save_path, filename + '.' + format),
                    format=format,
                    bbox_inches='tight',
                    pad_inches=0, **kwargs)
        print('{}.{} has been successfully saved to {}'.format(filename, format, save_path))
        plt.clf()
