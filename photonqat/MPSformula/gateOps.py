

"""
`gateOps` module implements calculation for quantum gate operations.
"""


import numpy as np
from .tnOps import *
from scipy.special import eval_hermite as hermite
from scipy.linalg import expm


def Displacement(mps, mode, alpha, modeNum, cutoff):
    mat1 = exp_annihiration(mps, -np.conj(alpha), cutoff = cutoff)
    mat2 = exp_creation(mps, alpha, cutoff = cutoff)
    _mps = one_qumode_gate(mps, mode, mat1)
    _mps = one_qumode_gate(_mps, mode, mat2)
    _mps[0][mode] = _mps[0][mode] * np.exp(-np.abs(alpha)**2 / 2)
    return _mps

def Squeeze(mps, mode, r, phi, modeNum, cutoff):
    G = np.exp(2 * 1j * phi) * np.tanh(r)
    mat1 = exp_annihiration(mps, np.conj(G) / 2, order = 2, cutoff = cutoff)
    mat2 = exp_photonNum(mps, -np.log(np.cosh(r)), cutoff = cutoff)
    mat3 = exp_creation(mps, -G / 2, order = 2, cutoff = cutoff)
    _mps = one_qumode_gate(mps, mode, mat1)
    _mps = one_qumode_gate(_mps, mode, mat2)
    _mps = one_qumode_gate(_mps, mode, mat3)
    _mps[0][mode] = _mps[0][mode] / np.sqrt(np.cosh(r))
    return _mps

def Rotation(mps, mode, theta, modeNum, cutoff):
    mat = exp_Aa(mps, 1j * theta, cutoff = cutoff)
    _mps = one_qumode_gate(mps, mode, mat)
    return _mps

def MZ(mps, mode1, mode2, theta, phi, modeNum, bond_dim, cutoff):
    _mps = Rotation(mps, mode1, phi, modeNum, cutoff)
    _mps = BS(_mps, mode1, mode2, theta, bond_dim)
    return _mps

def Beamsplitter(mps, mode1, mode2, theta, bond_dim):
    n_cutoff = mps[0][0].shape[1]
    mat = exp_BS(theta, n_cutoff)
    _mps = two_qumode_gate(mps, mode1, mode2, mat, bond_dim)
    return _mps

def MachZehnder(mps, mode1, mode2, theta, phi, modeNum, bond_dim, cutoff):
    if np.abs(phi) > 1e-10:
        mps = Rotation(mps, mode1, phi, modeNum, cutoff)
    if np.abs(theta) > 1e-10:
        mps = Beamsplitter(mps, mode1, mode2, theta, bond_dim)
    return mps

def MachZehnderInv(mps, mode1, mode2, theta, phi, modeNum, bond_dim, cutoff):
    if np.abs(theta) > 1e-10:
        mps = Beamsplitter(mps, mode1, mode2, -theta, bond_dim)
    if np.abs(phi) > 1e-10:
        mps = Rotation(mps, mode1, -phi, modeNum, cutoff)
    return mps