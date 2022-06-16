
"""
`gate` module implements quantum gate operations.
This module is internally used.
"""

import numpy as np
from .gateOps import *

class GATE():
    """Quantum gate class."""
    def __init__(self, obj):
        self.obj = obj

class Dgate(GATE):
    """
    Displacement gate.
    """
    def __init__(self, obj, mode, alpha):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.N = self.obj.N
        self.mode = mode
        self.alpha = alpha
        super().__init__(obj)

    def run(self, mps):
        self.alpha = _paramCheck(self.alpha)
        return Displacement(mps, self.mode, self.alpha, self.N, self.cutoff)

class Sgate(GATE):
    """
    Squeezing gate.
    """
    def __init__(self, obj, mode, r, phi = 0):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.N = self.obj.N
        self.mode = mode
        self.r = r
        self.phi = phi
        super().__init__(obj)

    def run(self, mps):
        self.r = _paramCheck(self.r)
        self.phi = _paramCheck(self.phi)
        return Squeeze(mps, self.mode, self.r, self.phi, self.N, self.cutoff)

class Rgate(GATE):
    """
    Rotation gate.
    """
    def __init__(self, obj, mode, theta):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.N = self.obj.N
        self.mode = mode
        self.theta = theta
        super().__init__(obj)

    def run(self, mps):
        self.theta = _paramCheck(self.theta)
        return Rotation(mps, self.mode, self.theta, self.N, self.cutoff)

class BSgate(GATE):
    """
    Beamsplitter gate.
    """    
    def __init__(self, obj, mode1, mode2, theta = np.pi / 4):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.bond_dim = self.obj.bond_dim
        self.N = self.obj.N
        self.mode1 = mode1
        self.mode2 = mode2
        self.theta = theta
        super().__init__(obj)

    def run(self, mps):
        self.theta = _paramCheck(self.theta)
        return Beamsplitter(mps, self.mode1, self.mode2, self.theta, self.bond_dim)

class MZgate(GATE):
    """
    Mach–Zehnder gate.
    """    
    def __init__(self, obj, mode1, mode2, theta, phi):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.bond_dim = self.obj.bond_dim
        self.N = self.obj.N
        self.mode1 = mode1
        self.mode2 = mode2
        self.theta = theta
        self.phi = phi
        super().__init__(obj)

    def run(self, mps):
        self.theta = _paramCheck(self.theta)
        self.phi = _paramCheck(self.phi)
        return MachZehnder(mps, self.mode1, self.mode2, self.theta, self.phi, self.N, self.bond_dim, self.cutoff)

class MZinvgate(GATE):
    """
    Inversed Mach–Zehnder gate.
    """    
    def __init__(self, obj, mode1, mode2, theta, phi):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.bond_dim = self.obj.bond_dim
        self.N = self.obj.N
        self.mode1 = mode1
        self.mode2 = mode2
        self.theta = theta
        self.phi = phi
        super().__init__(obj)

    def run(self, mps):
        self.theta = _paramCheck(self.theta)
        self.phi = _paramCheck(self.phi)
        return MachZehnderInv(mps, self.mode1, self.mode2, self.theta, self.phi, self.N, self.bond_dim, self.cutoff)

def _paramCheck(param):
    if isinstance(param, CregReader):
        return param.read()
    else:
         return param

class CregReader():
    """
    Class for reading classical register.
    """
    def __init__(self, reg, idx, var, scale = 1):
        self.reg = reg
        self.idx = idx
        self.var = var
        self.scale = scale

    def read(self):
        if self.var == "x":
            v = 0
        elif self.var == "p":
            v = 1
        elif self.var == "n":
            v = 2
        else:
            raise ValueError('Creg keeps measurement results of "x" or "p" or "n".')
        return self.reg[self.idx][v] * self.scale