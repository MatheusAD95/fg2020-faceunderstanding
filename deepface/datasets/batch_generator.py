import keras
import numpy as np
import cv2

class BatchGenerator(keras.utils.Sequence):
    def __init__(self, fnames, y, batch_size, input_reader, length=None, balance=False,
                 roundf=int, return_fnames=False):
        self.return_fnames = return_fnames
        self.batch_size = batch_size
        self.fnames = np.array(fnames)
        self.y = np.array(y)
        self.f = input_reader 
        self.n_classes = len(np.unique(self.y))
        self.len = roundf(len(y)/batch_size) if length is None else roundf(self.n_classes*length)
        self.balance = balance
        self.on_epoch_end()
        # self.size = 

    def __len__(self):
        return self.len

    def classes(self):
        return self.n_classes

    def on_epoch_end(self):
        order = np.arange(len(self.fnames))
        np.random.shuffle(order)
        self.y = self.y[order]
        self.fnames = self.fnames[order]

    def __getitem__(self, x):
        idx = np.zeros(self.fnames.shape[0], dtype=np.bool,)
        if self.balance:
            # sample `batch_size` ids
            ids = np.random.choice(np.arange(self.n_classes), self.batch_size)
            for ids_i, count_i in zip(*np.unique(ids, return_counts=True)):
                idx[np.random.choice(np.where(self.y == ids_i)[0], count_i, replace=False)] = 1
        else:
            idx[x*self.batch_size:(x + 1)*self.batch_size] = 1
        if self.n_classes > 2: # one hot encoding
            y = np.zeros((self.batch_size, self.n_classes))
            y[np.arange(self.batch_size), self.y[idx]] = 1
        else: # single neuron binary encoding
            y = self.y[idx]
        if self.return_fnames:
            return self.f(self.fnames[idx]), y, self.fnames[idx]
        return self.f(self.fnames[idx]), y
