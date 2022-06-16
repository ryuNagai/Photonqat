
"""
`WignerFunc` module implements calculation of Wigner function.
This module is internally used.
"""

import numpy as np
from scipy.special import factorial as fact


def reduceState(fockState, mode):
    modeNum = fockState.ndim
    cutoff = fockState.shape[-1] - 1
    fockState = np.swapaxes(fockState, mode, -1)
    fockState = fockState.flatten()
    rho = np.outer(np.conj(fockState), fockState)
    for i in range(modeNum - 1):
        rho = partialTrace(rho, cutoff)
    return rho

def partialTrace(rho, cutoff):
    dim1 = np.int(cutoff + 1)
    dim2 = np.int(rho.shape[0] / dim1)
    rho_ = np.zeros([dim2, dim2]) + 0j
    for j in range(dim1):
        rho_ += rho[(j * dim2):(j * dim2 + dim2), (j * dim2):(j * dim2 + dim2)]
    return rho_

def FockWigner(xmat, pmat, fockState, mode, tol=1e-10):
    rho = reduceState(fockState, mode)
    W = _Wigner_clenshaw(rho, xmat, pmat, tol)
    return W
    
def _Wigner_clenshaw(rho, xmat, pmat, tol, hbar = 1):
    g = np.sqrt(2 / hbar)
    M = rho.shape[0]
    A2 = g * (xmat + 1.0j * pmat)    
    B = np.abs(A2)
    B *= B
    w0 = (2*rho[0, -1])*np.ones_like(A2)
    L = M-1
    rho = rho * (2*np.ones((M,M)) - np.diag(np.ones(M)))
    while L > 0:
        L -= 1
        w0 = _Wigner_laguerre(L, B, np.diag(rho, L)) + w0 * A2 * (L+1)**-0.5
    W = w0 * np.exp(-B * 0.5) * (g ** 2 * 0.5 / np.pi)
    W = np.real(W)
    return W

def _Wigner_laguerre(L, x, c):
    
    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0,    y1 = c[-i] - y1 * (float((k - 1)*(L + k - 1))/((L+k)*k))**0.5, \
            y0 - y1 * ((L + 2*k -1) - x) * ((L+k)*k)**-0.5
            
    return y0 - y1 * ((L + 1) - x) * (L + 1)**-0.5