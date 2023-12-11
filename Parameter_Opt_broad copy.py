# Import necessary libraries and modules
import os
import numpy as np
import random
from Toolbox.Toolbox_Processing import *
from Toolbox.Toolbox_Reading import *
from Toolbox.Toolbox_Inversion import *
from Toolbox.Toolbox_Display import *

# Define the path to the spectra data
#path = "/home/luke/lukeflamingradis/EmFit_private/spectra/test_series"
base_path = "/home/luke/data/MATRIX_data/"
dataset = "Peat6"

os.makedirs('/home/luke/data/Model/results_param/'+dataset+'/', exist_ok=True)

P, T = getPT(dataset)

# Load chemical compound information from a pickle file
Compounds = getCompounds('/home/luke/lukeflamingradis/EmFit_private/Compounds.pickle')

# List of compounds to be removed from the Compounds dictionary
remove = ['SiH', 'CaF', 'SiS', 'BeH', 'HF', 'NH', 'SiH2', 'AlF', 'SH', 'CH', 'AlH', 'TiH', 'CaH', 'LiF', 'MgH', 'ClO']
remove.append('H2S')
remove.append('CH3Br')

# Remove specified compounds from the Compounds dictionary
for r in remove:
    Compounds.pop(r)

# Define broadening and regularization constants
regularisation_constant = 10**(-3)

broad_array = np.logspace(-6,-1, 60)

ref_spec_base, obs_spec, wv_obs = generateData_optimisation(Compounds, base_path + dataset, 0, T, P, dataset)

ref_spec_base = np.nan_to_num(ref_spec_base)
obs_spec = np.nan_to_num(obs_spec)

obs_spec = obs_spec[:, ~np.all(ref_spec_base == 0, axis=0)]
ref_spec_base = ref_spec_base[:, ~np.all(ref_spec_base == 0, axis=0)]

ref_spec, Compounds, Lass = lasso_inversion_opt(ref_spec_base, obs_spec, Compounds)

t_steps = [random.randint(0, obs_spec.shape[0]) for _ in range(10)]

obs_spec = obs_spec[t_steps,:]

for i, key in enumerate(Compounds):

    print("Finding optimal spectra for ", key)
    
    mol_arr = getReferenceMatrix_opt(Compounds[key], T, P, wv_obs, broad_array, key)

    Lasso_eval = []

    for arr in mol_arr:

        reference_spectra = ref_spec_base

        reference_spectra[i] = np.nan_to_num(arr[~np.all(ref_spec_base == 0, axis=0)])

        S = np.array([s[~np.all(reference_spectra == 0, axis=0)] for s in obs_spec])
        reference_spectra = np.array(reference_spectra[:, ~np.all(reference_spectra == 0, axis=0)])

        Lasso_eval.append(lasso_inversion_opt2(reference_spectra, S, Compounds))

    # Find minima
    ind = np.where([x['RMSE'] for x in Lasso_eval] == [x['RMSE'] for x in Lasso_eval].min())
    print(key, ind, mol_arr[ind])
    print([x['RMSE'] for x in Lasso_eval])
