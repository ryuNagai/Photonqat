import numpy as np
import matplotlib.pyplot as plt
from .MPSformula.gates import *
from .MPSformula.tnOps import *
from .MPSformula.WignerFunc import *

GATE_SET = {
    "D": Dgate,
    "BS": BSgate,
    "S": Sgate,
    "R": Rgate,
    "MZ": MZgate,
    "MZinv": MZinvgate
}

class MPS():
    """
    Class for continuous variable quantum compting with Matrix Product State.
    """
    def __init__(self, N, cutoff = 10, bond_dim = 4):
        self.N = N
        self.cutoff = cutoff
        self.bond_dim = bond_dim
        self.mps = create_MPS(self.N, self.cutoff, bond_dim = 1)
        self.state = None
        self.ops = []
        self.creg = [[None, None, None] for i in range(self.N)] # [x, p, n]

    def  __getattr__(self, name):

        if name in GATE_SET:
            self.ops.append(GATE_SET[name])
            # self.initStateFlag = True
            return self._setGateParam

        else:
            raise AttributeError('The method does not exist')

    def _setGateParam(self, *args, **kwargs):
        self.ops[-1] = self.ops[-1](self, *args, **kwargs)
        return self

    def Creg(self, idx, var, scale = 1):
        """
        Access to classical register.
        """
        return CregReader(self.creg, idx, var, scale)

    def run(self):
        """
        Run the circuit.
        """
        for op in self.ops:
            if isinstance(op, GATE):
                self.mps = op.run(self.mps)
                #sum_of_prob = np.sum(np.abs(self.state)**2)

        return self

    def mps_size(self):
        mps = self.mps
        rank = len(mps)
        size = 0
        for i in range(rank):
            size += np.prod(mps[i].shape)
        return size

    def mps_info(self):
        mps = self.mps
        rank = len(mps[0])
        for i in range(rank):
            print(mps[0][i].shape)
        for i in range(rank-1):
            print(mps[1][i].shape)

    def entangle_entropy(self):
        D = self.mps[1]
        EE = []
        for S in D:
            Z = np.sum(S**2)
            lamb = S**2 / Z
            ee = -np.sum(lamb * np.log(lamb))
            EE.append(ee)
        return EE

    def prob(self, output):
        components = np.array(output, dtype = np.int64)
        prob = np.abs(contract(self.mps, components))**2
        return prob

    def to_statevec(self):
        MPS = self.mps
        A = MPS[0]
        rank = len(A)
        n_cutoff = A[0].shape[1]
        bond_dim = A[0].shape[2]
        components_list = permutationWithRepetitionListRecursive(np.arange(n_cutoff), rank)
        state = []
        for component in components_list:
            state.append(contract(MPS, component))
        self.state = np.array(state) / np.sqrt(np.sum(np.abs(state)**2))
        return self.state

    def Wigner(self, mode, method = 'clenshaw', plot = 'y', xrange = 5.0, prange = 5.0):
        """
        Calculate the Wigner function of a selected mode.
        
        Args:
            mode (int): Selecting a optical mode.
            method: "clenshaw" (default) or "moyal".
            plot: If 'y', the plot of wigner function is output using matplotlib.\
                 If 'n', only the meshed values are returned.
            x(p)range: The range in phase space for calculateing Wigner function.
        """
        if self.state is None:
            self.state = self.to_statevec()
        x = np.arange(-xrange, xrange, xrange / 50)
        p = np.arange(-prange, prange, prange / 50)
        m = len(x)
        xx, pp = np.meshgrid(x, -p)
        W = FockWigner(xx, pp, self.state, mode, method)
        if plot == 'y':
            h = plt.contourf(x, p, W)
            plt.show()
        return (x, p, W)

    def Interferometer(self, U):
        num = self.N
        n = U.shape[0]
        BSang = [0.]*(int(n*(n-1)/2))
        rot = [0.]*(int(n*(n-1)/2))
        tol = 1e-10
        for i in range (1, n):
            if ((i+2)%2==1):
                for j in range (i): 
                    T = np.identity(n, dtype = complex)
                    if U[n-j-1][i-j-1]==0:
                        theta = 0
                        alpha = 0
                    elif U[n-j-1][i-j]==0:
                        theta = 0
                        alpha = np.pi/2
                    else:
                        theta = np.angle(U[n-j-1][i-1-j])-np.angle(U[n-j-1][i-j])
                        alpha = np.arctan(np.absolute(U[n-j-1][i-1-j])/np.absolute(U[n-j-1][i-j]))
                    i_ = np.int64(np.sum(range(1, i-1, 2)))
                    if np.abs(alpha) < tol:
                        alpha = 0.
                    if np.abs(theta) < tol:
                        theta = 0.
                    BSang[i_ + j] = alpha
                    rot[i_ + j] = theta
                    e = np.cos(-theta) + np.sin(-theta)*1j
                    T[i-1-j][i-1-j] = e*(np.cos(alpha)+0*1j)
                    T[i-1-j][i-j] = -np.sin(alpha)+0*1j
                    T[i-j][i-1-j] = e*(np.sin(alpha)+0*1j)
                    T[i-j][i-j] = np.cos(alpha)+0*1j
                    U = U @ np.transpose(T)
            else:
                for j in range (i):
                    T = np.identity(n, dtype = complex)
                    if U[n-i+j][j]==0:
                        theta = 0
                        alpha = 0
                    elif U[n-i+j-1][j]==0:
                        theta = 0
                        alpha = np.pi/2
                    else:
                        u1 = U[n + j - i - 1, j]
                        u2 = U[n + j - i, j]
                        a = np.real(u1)
                        b = np.imag(u1)
                        c = np.real(u2)
                        d = np.imag(u2)
                        theta = np.arctan((a*d - b*c) / (a*c + b*d))
                        alpha = np.arctan(c / (b*np.sin(theta) - a*np.cos(theta)))
                    if np.abs(alpha) < tol:
                        alpha = 0.
                    if np.abs(theta) < tol:
                        theta = 0.
                    i_ = np.sum(np.arange(0, i, 2))
                    BSang[-(i_+1+j)] = alpha
                    rot[-(i_+1+j)] = theta
                    e = np.cos(theta) + np.sin(theta)*1j
                    T[n-i+j-1][n-i+j-1] = e*(np.cos(alpha)+0*1j)
                    T[n-i+j-1][n-i+j] = -np.sin(alpha)+0*1j
                    T[n-i+j][n-i+j-1] = e*(np.sin(alpha)+0*1j)
                    T[n-i+j][n-i+j] = np.cos(alpha)+0*1j
                    U = T @ U
        U_d = np.diag(U)
        if np.max(np.abs(U - np.diag(U_d))) > 1e-10:
            raise ValueError("U is not nulled.")

        counter = 0
        for i in range(1, n, 2):
            for k in range(i):
                # self.ops.append(GATE_SET['R'])
                # self.ops[-1] = self.ops[-1](self, i-1-k, rot[counter])
                self.ops.append(GATE_SET['MZ'])
                self.ops[-1] = self.ops[-1](self, i-1-k, i-1-k+1, BSang[counter], rot[counter])
                counter += 1
        params = np.angle(U_d)
        for i in range(n):
            self.ops.append(GATE_SET['R'])
            self.ops[-1] = self.ops[-1](self, i, 2*params[i])
        for i in reversed(range(2, n-2 if (n-1)%2==0 else n-1, 2)):
            for k in range(i):
                # self.ops.append(GATE_SET['R'])
                # self.ops[-1] = self.ops[-1](self, n-2-k, rot[counter])
                self.ops.append(GATE_SET['MZinv'])
                self.ops[-1] = self.ops[-1](self, n-2-k, n-1-k, BSang[counter], rot[counter])
                counter += 1