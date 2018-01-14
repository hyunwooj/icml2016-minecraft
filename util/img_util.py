import numpy as np
from PIL import Image


def resize_img(img, w, h):
    image = Image.fromarray(img)
    resized_x = image.resize((int(w), int(h)), Image.ANTIALIAS)
    return np.asarray(resized_x)

def flip_left_right(imgs):
    new_imgs = []
    for img in imgs:
        img = np.transpose(img, (1, 2, 0))
        flipped = np.asarray(Image.fromarray(img).transpose(Image.FLIP_LEFT_RIGHT))
        new_imgs.append(np.transpose(flipped, (2, 0, 1)))
    return new_imgs
