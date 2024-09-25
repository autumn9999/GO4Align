import numpy as np
from numpy.linalg import eigh
from .sdp import sdp_kmeans


def sdp_kmeans_embedding(X, n_clusters, target_dim, ret_sdp=False,
                         method='cvx'):
    D, Q = sdp_kmeans(X, n_clusters, method=method)
    Y = spectral_embedding(Q, target_dim=target_dim, discard_first=True)
    if ret_sdp:
        return Y, D, Q
    else:
        return Y


def spectral_embedding(mat, target_dim, gramian=True, discard_first=True):
    if discard_first:
        last = -1
        first = target_dim - last
    else:
        first = target_dim
        last = None
    if not gramian:
        mat = mat - mat.mean(axis=0)
        mat = mat.dot(mat.T)
    eigvals, eigvecs = eigh(mat)

    sl = slice(-first, last)
    eigvecs = eigvecs[:, sl]
    eigvals_crop = eigvals[sl]
    Y = eigvecs.dot(np.diag(np.sqrt(eigvals_crop)))
    Y = Y[:, ::-1]

    variance_explaned(eigvals, eigvals_crop)
    return Y


def variance_explaned(eigvals, eigvals_crop):
    eigvals_crop[eigvals_crop < 0] = 0
    eigvals[eigvals < 0] = 0
    var = np.sum(eigvals_crop) / np.sum(eigvals)
    print('Variance explained:', var)
