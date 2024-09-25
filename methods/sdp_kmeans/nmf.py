from __future__ import print_function, division, absolute_import
import numpy as np


def symnmf_admm(A, k, H=None, maxiter=1e3, tol=1e-5, sigma=1):
    """
    A is a symmetric matrix
    Solves || A - W.dot(W.T) ||_F^2 s.t. W >= 0
    """
    A = A.copy()
    A[A < 0] = 0

    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError('A must be a symmetric matrix!')

    if H is None:
        H = np.sqrt(A.mean() / k) * np.random.randn(n, k)
        np.abs(H, H)

    Gamma = np.zeros((n, k))
    id_k = np.identity(k)
    step = 1

    error = []
    for i in range(int(maxiter)):
        temp = np.linalg.inv(H.T.dot(H) + sigma * id_k)
        W = (A.dot(H) + sigma * H - Gamma).dot(temp)
        W = np.maximum(W, 0)
        temp = np.linalg.inv(W.T.dot(W) + sigma * id_k)
        H = (A.dot(W) + sigma * W + Gamma).dot(temp)
        H = np.maximum(H, 0)
        Gamma += step * sigma * (W - H)

        error.append(np.linalg.norm(W - H) / np.linalg.norm(W))
        if i > 0 and np.abs(error[-1]) < tol:
            break

    return W


def symnmf_gram_admm(A, k, H=None, maxiter=1e3, tol=1e-5, sigma=1):
    """
    Solves || A.dot(A.T) - W.dot(W.T) ||_F^2 s.t. W >= 0
    """
    A = A.copy()
    A[A < 0] = 0

    if H is None:
        n = A.shape[0]
        H = np.sqrt(A.mean() / k) * np.random.randn(n, k)
        np.abs(H, H)

    Gamma = np.zeros((n, k))
    id_k = np.identity(k)
    step = 1

    error = []
    for i in range(int(maxiter)):
        temp = np.linalg.inv(H.T.dot(H) + sigma * id_k)
        W = (A.dot(A.T.dot(H)) + sigma * H - Gamma).dot(temp)
        W = np.maximum(W, 0)
        temp = np.linalg.inv(W.T.dot(W) + sigma * id_k)
        H = (A.dot(A.T.dot(W)) + sigma * W + Gamma).dot(temp)
        H = np.maximum(H, 0)
        Gamma += step * sigma * (W - H)

        error.append(np.linalg.norm(W - H) / np.linalg.norm(W))
        if i > 0 and np.abs(error[-1]) < tol:
            break

    return W
