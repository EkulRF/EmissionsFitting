# Import necessary libraries and modules
import os
import numpy as np
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
Compounds = getCompounds('/home/luke/lukeflamingradis/EmFit_private/EmissionsSpeciesInfo.pickle')

# List of compounds to be removed from the Compounds dictionary
remove = ['SiH', 'CaF', 'SiS', 'BeH', 'HF', 'NH', 'SiH2', 'AlF', 'SH', 'CH', 'AlH', 'TiH', 'CaH', 'LiF', 'MgH', 'ClO', 'CH3Br', 'H2S']

# Remove specified compounds from the Compounds dictionary
for r in remove:
    Compounds.pop(r)

regularisation_constant = 10**(-3)

ref_spec, obs_spec, full_ref_spec, Compounds = generateData(Compounds, base_path + dataset, 0, 273, 1.01325, dataset)

ref_spec, Compounds, Lasso_Evaluation, full_ref_spec = lasso_inversion(ref_spec, full_ref_spec, obs_spec, Compounds)

x_sol, sigma, C = temporally_regularised_inversion(ref_spec, obs_spec, regularisation_constant, dataset, list(Compounds.keys()))

(Ns, Nl), Nt = ref_spec.shape, obs_spec.shape[0]

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

wv_obs = np.load('/home/luke/data/Model/results/'+ dataset + '/W.npy')
T_guess = np.linspace(273, 673, 200)
P_guess = np.linspace(0.9, 10, 200)
initial_params = np.concatenate([T_guess, P_guess])
# Define bounds for the parameters if needed
bounds = [(0, None)] * len(initial_params)  # Assuming non-negative values for simplicity

Full_Ref_Spec = []
Ref_Spec = []

for i, spc in enumerate(list(Compounds.keys())):

    print(spc)

    species_arr = x_sol[i * Nt:(i + 1) * Nt]
    ind = np.argmax(species_arr)
    obs_selection = obs_spec[ind]

    loc = generateSingleRef(Compounds[spc], spc, wv_obs, 273, 1.01)[1]

    obs_selection = [element for start, end in loc for element in obs_selection[start:end + 1]]
    wv_selection = [element for start, end in loc for element in wv_obs[start:end + 1]]

    theoretical_spectra = [[element for start, end in loc for element in generateSingleRef(Compounds[spc], spc, wv_obs, T, P)[0][start:end + 1]]
 for T, P in zip(T_guess, P_guess)]
    theoretical_spectra = [((np.nan_to_num(spectrum) * LinearRegression().fit(np.nan_to_num(spectrum).reshape(-1, 1), np.nan_to_num(obs_selection)).coef_[0]) + LinearRegression().fit(np.nan_to_num(spectrum).reshape(-1, 1), np.nan_to_num(obs_selection)).intercept_) for spectrum in theoretical_spectra]

    np.save('/home/luke/data/Model/results_param/'+dataset+'/theoretical_spectra_' + spc + '.npy', theoretical_spectra)
    np.save('/home/luke/data/Model/results_param/'+dataset+'/wv_selection_' + spc + '.npy', wv_selection)
    np.save('/home/luke/data/Model/results_param/'+dataset+'/obs_selection_' + spc + '.npy', obs_selection)
    
    rmse_values, min_rmse_index = find_min_rmse(obs_selection, theoretical_spectra)

    plt.figure()
    plt.plot(T_guess, rmse_values)
    plt.savefig(spc + '.png')
    plt.show()

    reference_spec =  generateSingleRef(Compounds[spc], spc, wv_obs, T_guess[min_rmse_index], P_guess[min_rmse_index])[0]
    full_ref_spec =  generateSingleFullRef(Compounds[spc], spc, wv_obs, T_guess[min_rmse_index], P_guess[min_rmse_index])[0]
    Ref_Spec.append(reference_spec)
    Full_Ref_Spec.append(full_ref_spec)

Full_Ref_Spec = np.array(Full_Ref_Spec)
Ref_Spec = np.array(Ref_Spec)
Ref_Spec[np.isnan(Ref_Spec)] = 0 

Obs = np.array([s[~np.all(Ref_Spec == 0, axis=0)] for s in obs_spec])
Ref = np.array(Ref_Spec[:, ~np.all(Ref_Spec == 0, axis=0)])
W = wv_obs[~np.all(Ref_Spec == 0, axis=0)]

# Perform Tikhonov regularization
x_sol, sigma, C = temporally_regularised_inversion(Ref, Obs, regularisation_constant, dataset, list(Compounds.keys()))

# Calculate modeled and observed residuals
y_model, y, y_model_err = inversion_residual(Ref, Obs, x_sol, np.sqrt(sigma))

# Squeeze and extract information from the residuals
y_model_wv_squeezed, y_model_time_squeezed = squeeze_Residuals(y_model, y, Ref.shape[1])


# Saving Results
np.save('/home/luke/data/Model/results/'+ dataset + '/sol.npy', x_sol)
np.save('/home/luke/data/Model/results/'+ dataset + '/sig.npy', sigma)
np.save('/home/luke/data/Model/results/'+ dataset + '/comp.npy', Compounds)
np.save('/home/luke/data/Model/results/'+ dataset + '/ref.npy', Ref)
try:
    np.save('/home/luke/data/Model/results/'+ dataset + '/full_ref.npy', Full_Ref_Spec)
except:
    print("Failed to Save!")
np.save('/home/luke/data/Model/results/'+ dataset + '/obs.npy', Obs)
with open('/home/luke/data/Model/results/'+ dataset + '/C.pickle', 'wb') as handle:
    pkl.dump(C, handle, protocol=pkl.HIGHEST_PROTOCOL)
np.save('/home/luke/data/Model/results/'+ dataset + '/y_model_wv_squeezed.npy', y_model_wv_squeezed)
np.save('/home/luke/data/Model/results/'+ dataset + '/y_model_time_squeezed.npy', y_model_time_squeezed)
np.save('/home/luke/data/Model/results/'+ dataset + '/lasso_evaluation.npy', Lasso_Evaluation)


# Plot time series of concentration and emissions ratios
PlotTimeSeries('PPM_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), Obs.shape[0], dataset)
PlotER_TimeSeries('ER_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), Obs.shape[0], 'CO2', dataset)
PlotOPUS_Results('OPUS_result', dataset)

PlotSpectralResiduals(Full_Ref_Spec, np.load('/home/luke/data/Model/results/'+ dataset + '/full_resid_spectra.npy'), np.load('/home/luke/data/Model/results/'+ dataset + '/W_full.npy'), x_sol, sigma, Compounds, dataset)