import os

import numpy as np

from ..pls import train_pls_qda, train_neuron_qda, vip

class PLSAttributeClassifier:
    def __init__(self, feat_extractor, n_components, layernames, rand_samples=50):
        self.feat_extractor = feat_extractor
        self.n_components = n_components
        self.layernames = layernames
        self.rand_samples = rand_samples
        self.layer_classifiers = {}
        self.top_neurons_classifiers = {}
        self.top_channels_classifiers = {}
        self.channels_classifiers = {}
        self.rand_neurons_classifiers = {}

    def _neuronwise_training(self, x, y, vip_score, logs={}):
        ordered_idx = np.argsort(vip_score)[::-1]
        topk_idx = ordered_idx[:self.n_components]
        neuron_predictors = [train_neuron_qda(x, y, [idx])[-1] for idx in topk_idx]
        joint_predictor = train_neuron_qda(x, y, topk_idx)[-1]
        rand_scores = np.random.uniform(min(vip_score), max(vip_score), self.rand_samples)
        dist = np.abs(vip_score[None] - rand_scores[:, None])
        rand_idx = np.argmin(dist, axis=1)
        rand_predictors = [train_neuron_qda(x, y, [idx])[-1] for idx in rand_idx]
        logs['topk'] = topk_idx.tolist()
        logs['rand'] = rand_idx.tolist()
        return neuron_predictors + [joint_predictor], rand_predictors

    def _channelwise_training(self, x, y, vip_score, logs={}):
        if len(x.shape) < 4:
            return [], []
        channel_score = vip_score.reshape(-1, x.shape[-1]).mean(axis=0)
        ordered_idx = np.argsort(channel_score)[::-1]
        topk_idx = ordered_idx[:self.n_components]
        c = self.n_components
        channel_predictors = [train_pls_qda(x, y, c, [idx])[-1] for idx in range(x.shape[-1])]
        top_channel_predictors = [channel_predictors[idx] for idx in topk_idx]
        joint_predictor = train_pls_qda(x, y, c, topk_idx)[-1]
        logs['topk'] = topk_idx.tolist()
        return top_channel_predictors + [joint_predictor], channel_predictors

    def train(self, train_loader, logs={}):
        batch_size = train_loader.batch_size
        train_size = len(train_loader)*batch_size
        x = [np.zeros((train_size, *s[1:])) for s in self.feat_extractor.output_shape]
        y = np.zeros(train_size)
        print('Extracting features', flush=True)
        # load X and Y to memory for PLS training
        for i, (x_i, y_i) in enumerate(train_loader):
            feats = self.feat_extractor.predict(x_i)
            for j, feat in enumerate(feats):
                x[j][i*batch_size:(i + 1)*batch_size] = feat
            y[i*batch_size:(i + 1)*batch_size] = y_i
        print('Training PLS models', flush=True)
        for x_i, layername in zip(x, self.layernames):
            print(layername, flush=True)
            log_i = logs.setdefault(layername, {})
            pls, qda, predictor = train_pls_qda(x_i, y, self.n_components)
            vip_score = vip(pls)
            npred = self._neuronwise_training(x_i, y, vip_score, log_i.setdefault('neuron', {}))
            cpred = self._channelwise_training(x_i, y, vip_score, log_i.setdefault('channel', {}))
            self.layer_classifiers[layername] = [predictor]
            self.top_neurons_classifiers[layername] = npred[0]
            self.rand_neurons_classifiers[layername] = npred[1]
            self.top_channels_classifiers[layername] = cpred[0]
            self.channels_classifiers[layername] = cpred[1]
            log_i['vip_score'] = vip_score.tolist()

    def validate(self, val_loader, logs={}):
        predictors = {}
        pred_options = [self.layer_classifiers,
                        self.top_neurons_classifiers, self.rand_neurons_classifiers,
                        self.top_channels_classifiers, self.channels_classifiers]
        lnames = self.layernames
        for lname in lnames:
            predictors[lname] = [p for dict_list in pred_options for p in dict_list[lname]]
        results = self.test(predictors, val_loader)
        print('Validating PLS models')
        for lname in lnames:
            split_idx = np.cumsum([len(c[lname]) for c in pred_options])[:-1]
            layer, top_n, rand_n, top_c, all_c = np.split(results[lname], split_idx)
            logs.setdefault('layer', []).append(layer.tolist())
            logs.setdefault('top_neurons', []).append(top_n.tolist())
            logs.setdefault('rand_neurons', []).append(rand_n.tolist())
            if len(top_c):
                logs.setdefault('top_channels', []).append(top_c.tolist())
                logs.setdefault('channels', []).append(all_c.tolist())
        lidx = np.argmax(logs['layer'])
        lname = lnames[lidx]
        best_layer = (lname, self.layer_classifiers[lname][0])
        # the last column of top_neurons/channels contains the joint predictor of the topk units
        idx = np.argmax(np.array(logs['top_neurons'])[:, :-1])
        lidx, nidx = np.unravel_index(idx, np.array(logs['top_neurons'])[:, :-1].shape)
        lname = lnames[lidx]
        best_neuron = (lname, self.top_neurons_classifiers[lname][nidx])
        idx = np.argmax(np.array(logs['top_channels'])[:, :-1])
        lidx, cidx = np.unravel_index(idx, np.array(logs['top_channels'])[:, :-1].shape)
        lname = lnames[lidx]
        best_channel = (lname, self.top_channels_classifiers[lname][cidx])
        return best_layer, best_neuron, best_channel

    def test(self, classifiers, test_loader, logs={}):
        for layername in self.layernames:
            logs[layername] = [[] for _ in range(len(classifiers.get(layername, [])))]
        y = []
        for raw_x, y_i in test_loader:
            x = self.feat_extractor.predict(raw_x)
            y.append(y_i)
            for x_i, layername in zip(x, self.layernames):
                for i, classifier in enumerate(classifiers.get(layername, [])):
                    logs[layername][i].append(classifier(x_i))
        y = np.hstack(y)
        for layername in self.layernames:
            for i in range(len(logs.get(layername, []))):
                logs[layername][i] = (np.hstack(logs[layername][i]) == y).mean()
        return logs
