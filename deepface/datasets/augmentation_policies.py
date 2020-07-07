import cv2
import numpy as np

def test_transform(fnames):
    imgs = []
    for fname in fnames:
        img = cv2.imread(fname)
        imgs.append(cv2.resize(img, (224, 224)))
    return (np.float32(imgs) - 128.)/128.
