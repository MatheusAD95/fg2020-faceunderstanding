import numpy as np
import sklearn.cross_decomposition
import sklearn.discriminant_analysis
import warnings; warnings.filterwarnings('ignore')

def vip(pls_model):
    # Adapted from https://github.com/scikit-learn/scikit-learn/issues/7050
    t = pls_model.x_scores_       # (n_samples, n_components)
    w = pls_model.x_weights_      # (p, n_components)
    q = pls_model.y_loadings_     # (q, n_components)
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    w_norm = np.linalg.norm(w, axis=0)
    weights = (w/np.expand_dims(w_norm, axis=0))**2
    return np.sqrt(p*(weights @ s).ravel()/np.sum(s))

def train_pls_qda(x, y, n_components, c=None):
    x_dim = len(x.shape)
    if x_dim == 4: # (B, H, W, C)
        x = x[:, :, :, c] # channel selection
    x = x.reshape(x.shape[0], -1)
    n, p = x.shape
    pls = sklearn.cross_decomposition.PLSRegression(n_components)
    px, _ = pls.fit_transform(x, y)
    qda = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
    qda.fit(px, y)
    if x_dim < 4:
        return pls, qda, lambda x_test: qda.predict(pls.transform(x_test))
    return pls, qda, lambda x_test: qda.predict(pls.transform(x_test[:, :, :, c].reshape(-1, p)))

def train_neuron_qda(x, y, idx):
    x = x.reshape(x.shape[0], -1)
    n, p = x.shape
    qda = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
    qda.fit(x[:, idx], y)
    return qda, lambda x_test: qda.predict(x_test.reshape(-1, p)[:, idx])
