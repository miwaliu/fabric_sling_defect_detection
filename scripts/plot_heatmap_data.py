from os import listdir
from os.path import join, splitext

from ploting.heat_map import HeatMap
from numpy import asarray
from PIL import Image


def plot_heatmap_data(path_to_orig_image):
    for image_file in listdir(path_to_orig_image):
        image_name, _ = splitext(image_file)

        path_to_mask_image = join(path_to_orig_image, '..', 'pred_mask')
        mask_filename = \
            [join(path_to_mask_image, image) for image in listdir(path_to_mask_image) if image_name in image][0]

        with Image.open(mask_filename) as mask:
            with Image.open(join(path_to_orig_image, image_file)) as orig_image:
                hm = HeatMap(asarray(orig_image), asarray(mask))

                hm.save(image_name,
                        transparency=0.5,
                        format='jpg',
                        save_path='heatmap_data',
                        color_map='jet')
