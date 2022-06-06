import pandas as pd
import numpy as np
from utils import read_field
from scipy.interpolate import interp1d

# dns parameters
nu = 0.00043859
u_tau = 0.065789

df = pd.read_csv('DNS/Re_tau_150/Re_tau_150_Pr_071.csv')

df['y'] = df['y+'] * nu / u_tau
u_mean = df['u_mean'] * u_tau
cy = df['y']

c_rans = read_field('C', 10000, 'RANS/Re_tau_150_Pr_071/')
u_rans = read_field('U', 10000, 'RANS/Re_tau_150_Pr_071/')

u_new = u_rans.copy()
# u_new = np.zeros_like(u_rans)

c1 = (c_rans[:, 0] == 0.025)
c2 = (c_rans[:, 2] == 0.025)
c3 = (c_rans[:, 1] <= 1)
c_rans = c_rans[c1 & c2 & c3]
u_rans = u_rans[c1 & c2 & c3]

u_int = interp1d(cy, u_mean, axis=0, kind='cubic')
u_intd = u_int(c_rans[:, 1])

c_new = np.concatenate([c_rans[:, 1], np.flip(2 - c_rans[:, 1])])
u_intd = np.concatenate([u_intd, np.flip(u_intd)])

c_rans = read_field('C', 10000, 'RANS/Re_tau_150_Pr_071/')
u_new[(c_rans[:, 0] == 0.025) & (c_rans[:, 2] == 0.025), 0] = u_intd
u_new[(c_rans[:, 0] == 0.025) & (c_rans[:, 2] == 0.075), 0] = u_intd
u_new[(c_rans[:, 0] == 0.075) & (c_rans[:, 2] == 0.025), 0] = u_intd
u_new[(c_rans[:, 0] == 0.075) & (c_rans[:, 2] == 0.075), 0] = u_intd


# prepare and save b_ij field in OpenFOAM format
u_new_field = ''
for line in u_new:
    print(line)
    u_new_field += f'({} {} {})\n'

for i in range(2):
    for j in range(int(b_GB_filtered_intd.shape[0]/2)):
        x = b_GB_filtered_intd[j]
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'

for i in range(2):
    for j in range(int(b_GB_filtered_intd.shape[0]/2), int(b_GB_filtered_intd.shape[0])):
        x = b_GB_filtered_intd[j]
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'

file_template = f'''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2012                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volSymmTensorField;
    location    "0";
    object      bij;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];


internalField   nonuniform List<symmTensor>
{b_GB_filtered_intd.shape[0]*4}
(
{b_ij_field}
)
;
boundaryField
{{
    walls
    {{
        type            fixedValue;
        value           uniform (0 0 0 0 0 0);
    }}
    inlet
    {{
        type            cyclic;
    }}
    outlet
    {{
        type            cyclic;
    }}
    sides
    {{
        type            empty;
    }}
}}
// ************************************************************************* //'''

# Saving b_ij as a text file
with open('bij', 'w') as file:
    file.write(file_template)