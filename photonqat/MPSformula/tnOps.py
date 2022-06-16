import tensornetwork as tn
import numpy as np
from scipy.linalg import expm
import itertools as itr
import copy

def block(*dimensions):
    # Construct a new matrix for the MPS with random numbers from 0 to 1
    size = tuple([x for x in dimensions])
    bl = np.zeros(size, dtype=np.complex)
    bl[0, 0, 0] = 1.
    return bl

def create_MPS(rank, dim, bond_dim = 1):
    # Build the MPS tensor
    A = [block(1, dim, bond_dim)] + \
    [block(bond_dim, dim, bond_dim) for _ in range(rank-2)] + \
    [block(bond_dim, dim, 1)]
    d = [np.ones(bond_dim) for _ in range(rank-1)]
    mps = [A, d]
    return mps

def _upMat(dim, order):
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(0, dim - order):
            A[i + order, i] = np.prod(np.sqrt(np.arange(i + 1, i + 1 + order)))
        return A
    
def _downMat(dim, order):    
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(order, dim):
            A[i - order, i] = np.prod(np.sqrt(np.arange(i, i - order, -1)))
        return A

    
def _nMat(dim, order):
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.diag(np.arange(dim) ** order)
        return A
    
def exp_creation(mps, alpha, order = 1, cutoff = 10):
    mat = _upMat(mps[0][0].shape[1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    return mat_

def exp_annihiration(mps, alpha, order = 1, cutoff = 10):
    mat = _downMat(mps[0][0].shape[1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    return mat_

def exp_Aa(mps, alpha, cutoff = 10):
    matd = _downMat(mps[0][0].shape[1], 1)
    matu = _upMat(mps[0][0].shape[1], 1)
    mat = np.dot(matu, matd)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    return mat_

def exp_photonNum(mps, alpha, order = 1, cutoff = 10):
    mat = _nMat(mps[0][0].shape[1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    return mat_

def one_qumode_gate(mps, mode, mat):
    gate_node = tn.Node(mat)
    target_node = tn.Node(mps[0][mode])
    after_node = tn.contract(gate_node.edges[1] ^ target_node.edges[1])
    after = np.swapaxes(after_node.tensor, 0, 1)
    mps[0][mode] = after
    return mps

def _mat_for_mode1(mat):
    l = mat.shape[0]
    mat_ = np.zeros(np.array(mat.shape)**2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            mat_[i*l:i*l+l, j*l:j*l+l] = np.eye(l) * mat[i, j]
    return mat_

def _mat_for_mode2(mat):
    l = mat.shape[0]
    mat_ = np.zeros(np.array(mat.shape)**2)
    for i in range(mat.shape[0]):
        mat_[i*l:i*l+l, i*l:i*l+l] = mat
    return mat_

def exp_BS(alpha, cutoff):
    down = _downMat(cutoff, 1)
    up = _upMat(cutoff, 1)
    mat1_ = np.dot(_mat_for_mode1(up), _mat_for_mode2(down))
    mat2_ = np.dot(_mat_for_mode1(down), _mat_for_mode2(up))
    mat_ = mat1_ - mat2_
    emat_ = expm(alpha * mat_)
    return emat_

def mat_to_tensor(mat, n_cutoff):
    vstack_list = []
    for j in range(n_cutoff):
        hstack_list = []
        for i in range(n_cutoff):
            hstack_list.append(np.reshape(mat[j*n_cutoff + i, :], (n_cutoff, n_cutoff)))
        vstack_list.append(np.hstack(hstack_list))
    tensor = np.vstack(vstack_list)
    return tensor

def two_qumode_gate(mps, mode1, mode2, mat, bond_dim):
    n_cutoff = mps[0][0].shape[1]
    tensor = mat_to_tensor(mat, n_cutoff)
    U, D, V = np.linalg.svd(tensor)
    Us = np.vsplit(U, n_cutoff)
    Vs = np.hsplit(np.dot(np.diag(D), V), n_cutoff)
    
    A1 = mps[0][mode1]
    if mode1 > 0:
        A1 = np.tensordot(np.diag(mps[1][mode1-1]), A1, (1, 0))

    A1 = np.tensordot(A1, np.diag(np.sqrt(mps[1][mode1])), (2, 0))
    A2 = mps[0][mode2]
    A2 = np.tensordot(np.diag(np.sqrt(mps[1][mode1])), A2, (1, 0))

    if mode2 < len(mps[0])-1:
        A2 = np.tensordot(A2, np.diag(mps[1][mode2]), (2, 0))

    after_mode1 = np.sum([np.kron(Us[i], A1[:, i, :]) for i in range(n_cutoff)], axis = 0)
    after_mode2 = np.sum([np.kron(Vs[i], A2[:, i, :]) for i in range(n_cutoff)], axis = 0)
    two_site_mat = np.dot(after_mode1, after_mode2)
    u, d, v = svd_bond_dimension_cutoff(two_site_mat, bond_dim)
    after_mode1_ = np.reshape(u, (mps[0][mode1].shape[0], mps[0][mode1].shape[1], -1), order = 'F')
    after_mode2_ = np.reshape(v, (-1, mps[0][mode2].shape[1], mps[0][mode2].shape[2]))
    _mps = copy.copy(mps)

    if mode1 > 0:
        after_mode1_ = np.tensordot(np.diag(1 / mps[1][mode1-1]), after_mode1_, (1, 0))

    if mode2 < len(mps[0])-1:
        after_mode2_ = np.tensordot(after_mode2_, np.diag(1 / mps[1][mode2]), (2, 0))

    _mps[0][mode1] = after_mode1_
    _mps[0][mode2] = after_mode2_
    _mps[1][mode1] = d
    return _mps

### Before Change
# def two_qumode_gate(mps, mode1, mode2, mat, bond_dim):
#     n_cutoff = mps[0][0].shape[1]
#     tensor = mat_to_tensor(mat, n_cutoff)
#     U, D, V = np.linalg.svd(tensor)
#     Us = np.vsplit(U, n_cutoff)
#     Vs = np.hsplit(np.dot(np.diag(D), V), n_cutoff)
    
#     A1 = mps[0][mode1]
#     A1 = np.tensordot(A1, np.diag(np.sqrt(mps[1][mode1])), (2, 0))
#     A2 = mps[0][mode2]
#     A2 = np.tensordot(np.diag(np.sqrt(mps[1][mode1])), A2, (1, 0))

#     after_mode1 = np.sum([np.kron(Us[i], A1[:, i, :]) for i in range(n_cutoff)], axis = 0)
#     after_mode2 = np.sum([np.kron(Vs[i], A2[:, i, :]) for i in range(n_cutoff)], axis = 0)
#     two_site_mat = np.dot(after_mode1, after_mode2)
#     u, d, v = svd_bond_dimension_cutoff(two_site_mat, bond_dim)
#     after_mode1_ = np.reshape(u, (mps[0][mode1].shape[0], mps[0][mode1].shape[1], -1), order = 'F')
#     after_mode2_ = np.reshape(v, (-1, mps[0][mode2].shape[1], mps[0][mode2].shape[2]))
#     _mps = copy.copy(mps)
#     _mps[0][mode1] = after_mode1_
#     _mps[0][mode2] = after_mode2_
#     _mps[1][mode1] = d
#     return _mps

def svd_bond_dimension_cutoff(A, bond_dim):
    u, d, v = np.linalg.svd(A, full_matrices=False)
    if bond_dim < 0:
        return u, d, v
    if len(d) > bond_dim:
        u = u[:, :bond_dim]
        d = d[:bond_dim]
        v = v[:bond_dim, :]
    return u, d, v

def permutationWithRepetitionListRecursive(data, r):
    if r <= 0:
        return []
    result = []
    _permutationWithRepetitionListRecursive(data, r, [], result)
    return result

def _permutationWithRepetitionListRecursive(data, r, progress, result):
    if r == 0:
        result.append(progress)
        return
    for i in range(len(data)):
        _permutationWithRepetitionListRecursive(data, r - 1, progress + [data[i]], result)

# def contract(MPS, component):
#     rank = len(MPS)
#     mps = [MPS[i][:, component[i], :] for i in range(rank)]
#     mps_nodes = [tn.Node(tensor) for i, tensor in enumerate(mps)]
#     mps_connected_bonds = [mps_nodes[k].edges[1] ^ mps_nodes[k+1].edges[0] for k in range(-1,rank-1)]
#     for x in mps_connected_bonds:
#         mps_contracted_node = tn.contract(x) # update for each contracted bond
#     return mps_contracted_node.tensor.item()

# def contract(MPS, component):
#     rank = len(MPS)
#     mps = [MPS[i][:, component[i], :] for i in range(rank)]
#     # eps = 1e-20
#     # A = mps[0]
#     # for i in range(1, rank):
#     #     th = np.max(A) * eps
#     #     A[np.abs(A) < th] = 0
#     #     A = np.dot(A, mps[i])
#     # res = np.squeeze(A)
#     res = np.squeeze(np.linalg.multi_dot(mps))
#     return res

def contract(MPS, component):
    A = MPS[0]
    D = MPS[1]
    rank = len(A)
    A_ = [A[i][:, component[i], :] for i in range(rank)]
    D_ = [np.diag(D[i]) for i in range(rank-1)]
    M = []
    for i in range(rank-1):
        M.append(A_[i])
        M.append(D_[i])
    M.append(A_[-1])
    res = np.linalg.multi_dot(M)
    return res[0][0]