import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

def _overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def crop_and_paste(donor_image_path, donor_mask_path, target_image_path):
    # https://stackoverflow.com/questions/62813546/how-do-i-crop-an-image-based-on-custom-mask-in-python

    # crop mask wiht defect
    temple = plt.imread(donor_image_path)
    target_image = plt.imread(target_image_path)
    heart = plt.imread(donor_mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(heart, thresh=180, maxval=255, type=cv2.THRESH_BINARY)

    mask, _, _ = cv2.split(mask)

    temple_x, temple_y, _ = temple.shape
    heart_x, heart_y = mask.shape

    x_heart = min(temple_x, heart_x)
    x_half_heart = mask.shape[0] // 2

    heart_mask = mask[x_half_heart - x_heart // 2: x_half_heart + x_heart // 2 + 1, :temple_y]
    masked = cv2.bitwise_and(temple, temple, mask=heart_mask)

    tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(masked)
    rgba = [b, g, r, alpha]
    masked = cv2.merge(rgba, 4)

    # paste defect to target
    alpha_mask = masked[:, :, 3] / 255.0
    img_result = target_image[:, :, :3].copy()
    img_overlay = masked[:, :, :3]
    _overlay_image_alpha(img_result, img_overlay, 0, 0, alpha_mask)

    # print crop_and_pasted images
    dpi = mpl.rcParams['figure.dpi']
    figsize = temple.shape[1] / float(dpi), temple.shape[0] / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')
    ax.imshow(masked)
    plt.show()


if __name__ == '__main__':
    crop_and_paste(donor_image_path='IMG_0441.jpg',
                   donor_mask_path='IMG_0441_Perc_1.086%.jpg',
                   target_image_path='IMG_0451.jpg')
