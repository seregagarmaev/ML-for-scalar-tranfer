import numpy as np
from numpy.linalg import multi_dot
from scipy.ndimage import gaussian_filter1d


def read_field(field_name, step, path):
    '''This function reads the fields in the OpenFOAM format.'''

    # read file
    file_path = f'{path}/{step}/{field_name}'
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    # determine the file parameters
    for i, line in enumerate(lines):
        if 'internalField' in line:
            field_type = line[line.find('<')+1 : line.find('>')]
            n_cells = int(lines[i+1])
            start_line = i+3
            end_line = start_line + n_cells
            break

    # format field as numpy array
    field = np.loadtxt([line.strip('()') for line in lines[start_line:end_line]])
    if field_type == 'tensor':
        field = field.reshape((-1, 3, 3))
    if field_type == 'symmTensor':
        template = np.zeros((n_cells, 3, 3))
        for i in range(n_cells):
            template[i,:,:] = np.array([
                [field[i, 0], field[i, 1], field[i, 2]],
                [field[i, 1], field[i, 3], field[i, 4]],
                [field[i, 2], field[i, 4], field[i, 5]],
            ])
        field = template

    return field

def dns_to_R(data):
    '''This function creates Rij field using DNS data.'''
    n_points = data.shape[0]
    R = np.zeros((n_points, 3, 3))
    for i in range(n_points):
        uu = data.loc[i, 'uu+']
        vv = data.loc[i, 'vv+']
        ww = data.loc[i, 'ww+']
        uv = data.loc[i, 'uv+']
        r = np.array([[uu, uv, 0],
                      [uv, vv, 0],
                      [0, 0, ww]])
        R[i, :, :] = r
    return R

def generate_S_R(grad_U, omega):
    '''Generate S and R tensors using U gradients and specific dissipation rate'''
    n_cells = grad_U.shape[0]
    S = np.zeros((n_cells, 3, 3))
    R = np.zeros((n_cells, 3, 3))
    for i in range(n_cells):
        S[i, :, :] = 0.5 / omega[i] * (grad_U[i, :, :] + np.transpose(grad_U[i, :, :]))
        R[i, :, :] = 0.5 / omega[i] * (grad_U[i, :, :] - np.transpose(grad_U[i, :, :]))
    return S, R

def calc_invariants(S, R, grad_c, Re_d, nut, nu, num_invariants=15):
    '''Generates 15 invariants based on strain and rotation rate tensors.'''
    n_cells = S.shape[0]
    invariants = np.zeros((n_cells, num_invariants))
    for i in range(n_cells):
        invariants[i, 0] = np.trace(np.dot(S[i, :, :], S[i, :, :]))
        invariants[i, 1] = np.trace(multi_dot([S[i, :, :], S[i, :, :], S[i, :, :]]))
        invariants[i, 2] = np.trace(np.dot(R[i, :, :], R[i, :, :]))
        invariants[i, 3] = np.trace(multi_dot([S[i, :, :], R[i, :, :], R[i, :, :]]))
        invariants[i, 4] = np.trace(multi_dot([S[i, :, :], S[i, :, :], R[i, :, :], R[i, :, :]]))
        invariants[i, 5] = np.trace(multi_dot([S[i, :, :], S[i, :, :], R[i, :, :], R[i, :, :], S[i, :, :], R[i, :, :]]))
        invariants[i, 6] = np.dot(grad_c[i, :].T, grad_c[i, :])
        invariants[i, 7] = multi_dot([grad_c[i, :].T, S[i, :, :], grad_c[i, :]])
        invariants[i, 8] = multi_dot([grad_c[i, :].T, S[i, :, :], S[i, :, :], grad_c[i, :]])
        invariants[i, 9] = multi_dot([grad_c[i, :].T, R[i, :, :], R[i, :, :], grad_c[i, :]])
        invariants[i, 10] = multi_dot([grad_c[i, :].T, S[i, :, :], R[i, :, :], grad_c[i, :]])
        invariants[i, 11] = multi_dot([grad_c[i, :].T, S[i, :, :], S[i, :, :], R[i, :, :], grad_c[i, :]])
        invariants[i, 12] = multi_dot([grad_c[i, :].T, R[i, :, :], S[i, :, :], R[i, :, :], R[i, :, :], grad_c[i, :]])
        invariants[i, 13] = Re_d[i]
        invariants[i, 14] = nut[i] / nu
    return invariants

def calc_tensor_basis(S, R):
    '''Calculate 6 basis tensors using strain and rotation rate tensors.'''
    n_cells = S.shape[0]
    T = np.zeros((n_cells, 6, 3, 3))
    for i in range(n_cells):
        s = S[i, :, :]
        r = R[i, :, :]

        T[i, 0, :, :] = np.eye(3)
        T[i, 1, :, :] = s
        T[i, 2, :, :] = r
        T[i, 3, :, :] = np.dot(s, s)
        T[i, 4, :, :] = np.dot(r, r)
        T[i, 5, :, :] = np.dot(s, r) + np.dot(r, s)
    return T

def R_to_b(array):
    n_points = array.shape[0]
    b = np.zeros((n_points, 3, 3))
    for i in range(n_points):
        k = array[i].trace() / 2
        if k != 0:
            b[i, :, :] = array[i] / 2 / k - 1 / 3 * np.eye(3)
    return b

def R_to_a(array):
    n_points = array.shape[0]
    a = np.zeros((n_points, 3, 3))
    for i in range(n_points):
        k = array[i].trace() / 2
        if k != 0:
            a[i, :, :] = array[i] - 2 / 3 * k * np.eye(3)
    return a

def wbRe(k_array, cy_array, nu):
    '''Calculate wall distance based Reynolds number.'''
    result = []
    for k, cy in zip(k_array, cy_array):
        wb_Re = np.sqrt(k) * min(cy,2-cy) / 50 / nu
        # wb_Re = min(cy, 2 - cy)
        result.append(wb_Re)
    return np.array(result)

def wallDistance(cy_array):
    '''Calculate wall distance.'''
    result = []
    for cy in cy_array:
        dist = min(cy, 2 - cy)
        result.append(dist)
    return np.array(result)

def gauss_filter(b, sigma=10):
    b_extended = extend(b)
    b_filtered = gaussian_filter1d(b_extended, sigma, axis=0)
    half_index = int(b_filtered.shape[0] / 2)
    b_filtered = b_filtered[:half_index]
    return b_filtered

def extend(b):
    b_mirrored = np.flip(b, axis=0)
    b_mirrored = b_mirrored * np.array([[1, -1, -1],
                                        [-1, 1, -1],
                                        [-1, -1, 1]])
    b_extended = np.concatenate((b, b_mirrored))
    return b_extended
