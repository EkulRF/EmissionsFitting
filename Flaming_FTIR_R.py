import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy

from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum

from Toolbox_Processing import *
from Toolbox_Reading import *
from Toolbox_Inversion import *


path = "EmFit_private/spectra/test_series"

Compounds = getCompounds('EmFit_private/Compounds.pickle')

remove = ['SiH', 'CaF', 'SiS', 'BeH', 'HF', 'HCl', 'NH', 'SiH2', 'AlF', 'SH', 'CH', 'AlH', 'TiH', 'CaH', 'LiF', 'MgH', 'ClO']

for r in remove:
    Compounds.pop(r)

ref_spec, obs_spec, Compounds = generateData(Compounds, path)

#Lasso Inversion - removes compounds not present
ref_spec, Compounds, A, Lasso_Evaluation = lasso_inversion(ref_spec, obs_spec, Compounds)
# Should have option to say whether we want lasso regression stats???

#Tikhonov Regularisation
x_sol, sigma, C = temporally_regularised_inversion(ref_spec, obs_spec, 0.00000)

#sigma = np.sqrt(sigma.diagonal())

##### How well does it fit??? Residual shit here!!!!!

np.save('sol.npy', x_sol)
np.save('sig.npy', sigma)
np.save('comp.npy', Compounds)
np.save('ref.npy', ref_spec)
np.save('obs.npy', obs_spec)
np.save('C.npy', C)

# Converting estimated parameters to PPM concentrations
x_sol, sigma = convert2PPM_new_new(Compounds, x_sol, sigma, obs_spec.shape[0], 500)










# np.save('sol.npy', x_sol)
# np.save('sig.npy', sigma)
# np.save('comp.npy', Compounds)
# np.save('ref.npy', ref_spec)
# np.save('obs.npy', obs_spec)

### make below into func
compound_list = list(Compounds.keys())
Nt = obs_spec.shape[0]

num_rows = len(compound_list) // 2  # Calculate the number of rows needed

fig, axs = plt.subplots(num_rows+1, 2, figsize=(10, 6))

for i, spc in enumerate(compound_list):
    row, col = divmod(i, 2)
    axs[row,col].plot(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt], color = 'red')
    axs[row,col].fill_between(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt] - 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
                            x_sol[i*Nt:(i+1)*Nt] + 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
                            color= "0.8")
    axs[row, col].set_title(spc)

plt.savefig('result.jpg')