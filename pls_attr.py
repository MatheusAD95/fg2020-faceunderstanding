import os
import sys
import json
import hashlib

import numpy as np
import tensorflow as tf

import deepface as df
from deepface.datasets.augmentation_policies import test_transform

N_TRAIN = 1024
N_VAL = 3072
COMPONENTS = 8

def set_seed(s):
    hex_seed = hashlib.sha1(s.encode()).hexdigest()
    int32_seed = int(hex_seed, 16)%(2**32 - 1)
    np.random.seed(int32_seed)
    tf.set_random_seed(int32_seed)

def attribute_analysis(celeba_path, attr, model, layers):
    logs = {}
    train, val, test = df.datasets.celeba.create_generators(celeba_path, attr,
                                                            test_transform, N_TRAIN, N_VAL)
    pls_pred = df.attribute_prediction.PLSAttributeClassifier(model, COMPONENTS, layers)
    pls_pred.train(train, logs=logs.setdefault('train', {}))
    layer, neuron, channel = pls_pred.validate(val, logs=logs.setdefault('val', {}))
    d = {}
    for lname, predf in [layer, neuron, channel]:
        d.setdefault(lname, []).append(predf)
    test_results = pls_pred.test(d, test)
    for key, (lname, predf) in zip(['layer', 'neuron', 'channel'], [layer, neuron, channel]):
        logs.setdefault('test', {})[key] = test_results[lname].pop(0)
    return logs

if __name__ == '__main__':
    args = json.load(open(sys.argv[1], 'r'))
    os.makedirs(args['outdir'], exist_ok=True)
    layers = ['activation_{:d}'.format(i) for i in range(1, 50)] + ['avg_pool']
    # layers = ['activation_{:d}'.format(i) for i in range(49, 50)] + ['avg_pool']
    model = df.models.Resnet50(args['weights'], output_layers=layers)
    for attr in open(args['attrs_file'], 'r').read().splitlines():
        print(attr)
        set_seed(args['seed']) # seed is reset for each attribute-experiment
        logs = attribute_analysis(args['celeba_path'], attr, model, layers)
        logsf = os.path.join(args['outdir'], '{}.json'.format(attr))
        json.dump(logs, open(logsf, 'w'))
        print('\n')
