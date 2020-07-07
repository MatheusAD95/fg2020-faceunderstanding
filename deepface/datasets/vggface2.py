import os

import numpy as np

def loadpaths_vggface2(vggface2_path, split='train'):
    bbfile = open(os.path.join(vggface2_path, 'bb_landmark', 'loose_bb_{}.csv'.format(split)), 'r')
    fnames, y, bboxes = [], [], {}
    for line in bbfile.read().splitlines()[1:]:
        name, x0, y0, w, h = line.strip().split(',')
        x0, y0, w, h = [int(k) for k in [x0, y0, w, h]]
        name = name.strip('\'\"') + '.jpg'
        y.append(os.path.split(name)[0])
        name = os.path.join(vggface2_path, split, name)
        bboxes[name] = [x0, y0, x0 + w, y0 + h]
        fnames.append(name)
    return np.array(fnames), np.unique(y, return_inverse=True)[1], bboxes
