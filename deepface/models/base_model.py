import cv2
import keras
import numpy as np

def _random_gray(img):
    if np.random.uniform() < 0.20:
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        return img

class BaseModel:
    def __init__(self):
        self.model = None

    def input_reader_eval(self, fnames):
        imgs = []
        for fname in fnames:
            img = cv2.imread(fname)
            imgs.append(cv2.resize(img, (224, 224)))
        return (np.float32(imgs) - 128.)/128.

    def input_reader(self, fnames):
        imgs = []
        for fname in fnames:
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            ratio = (256./min(h, w))
            img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
            h, w = img.shape[:2]
            dx, dy = np.random.randint(0, w - 224), np.random.randint(0, h - 224)
            imgs.append(_random_gray(img[dy:dy+224, dx:dx+224]))
        return (np.float32(imgs) - 128.)/128.
