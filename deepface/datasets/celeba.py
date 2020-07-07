import os
import re
from math import ceil

import numpy as np

from .batch_generator import BatchGenerator

def loadpaths(*args, **kwargs):
    return _loadpaths(*args, **kwargs)

def _loadpaths(celeba_path, split='train', sigmoid=False):
    split_dict = {0: 'train', 1: 'val', 2: 'test'}
    split_file = os.path.join(celeba_path, 'Eval', 'list_eval_partition.txt')
    lines = open(split_file, 'r').read().splitlines()
    partition = {}
    for line in lines:
        tokens = re.split('[ ]+', line.strip())
        partition[tokens[0]] = split_dict[int(tokens[1])]
    attr_file = os.path.join(celeba_path, 'Anno', 'list_attr_celeba.txt')
    lines = open(attr_file, 'r').read().splitlines()
    attrnames = np.array(lines[1].strip().split(' '))
    attrs, fnames = [], []
    for i, line in enumerate(lines[2:]):
        tokens = re.split('[ ]+', line.strip())
        if partition[tokens[0]] != split:
            continue
        fnames.append(os.path.join(celeba_path, 'Img', 'img_align_celeba', tokens[0]))
        attrs.append(np.int8(tokens[1:]))
    if sigmoid:
        attrs = np.ceil((np.int8(attrs) + .5))//2
    return np.array(fnames), np.int8(attrs), attrnames

def _choose_2k_samples(x, y, k):
    pos = np.random.choice(np.where(y > 0)[0], k, replace=False)
    neg = np.random.choice(np.where(y < 0)[0], k, replace=False)
    samples = np.hstack([pos, neg])
    mask = np.zeros(y.shape, dtype=np.bool)
    mask[samples] = True
    return x[samples], y[samples], mask

def create_generators(celeba_path, attr, input_reader, n_train, n_val):
    train_x, train_y, attrnames = _loadpaths(celeba_path, split='train')
    val_x, val_y, _ = _loadpaths(celeba_path, split='val')
    # we mix the train and validation sets to get enough samples for all attributes
    x = np.hstack([train_x, val_x])
    y = np.vstack([train_y, val_y])[:, attrnames == attr].ravel()
    val_x, val_y, val_mask = _choose_2k_samples(x, y, n_val)
    train_x, train_y, _ = _choose_2k_samples(x[~val_mask], y[~val_mask], n_train)
    test_x, test_y, _ = _loadpaths(celeba_path, split='test')
    test_y = test_y[:, attrnames == attr].ravel()
    # generators
    batch_size = n_train
    train_gen = BatchGenerator(train_x, train_y, 2*batch_size, input_reader)
    val_gen = BatchGenerator(val_x, val_y, batch_size, input_reader, roundf=ceil)
    test_gen = BatchGenerator(test_x, test_y, batch_size, input_reader, roundf=ceil)
    return train_gen, val_gen, test_gen
