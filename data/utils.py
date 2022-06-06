import numpy as np


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