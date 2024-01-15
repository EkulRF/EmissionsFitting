# Import necessary libraries and modules
import numpy as np
from Toolbox.Toolbox_Processing import *
from Toolbox.Toolbox_Reading import *
from Toolbox.Toolbox_Inversion import *
from Toolbox.Toolbox_Display import *

# Define the path to the spectra data
base_path = "/home/luke/data/MATRIX_data/"
dataset = "Peat9"

makeDirs(dataset)

P, T = getPT(dataset)
print(f"Pressure: {P}, Temperature: {T}")
P *= 10
T = 250
T, P = 693, 0.101325

# Load chemical compound information from a pickle file
Compounds = getCompounds('/home/luke/lukeflamingradis/EmFit_private/EmissionsSpeciesInfo.pickle')

#Compounds['CO2']['bounds'] = [Compounds['CO2']['bounds'][0]]
Compounds['CO']['bounds'] = [Compounds['CO']['bounds'][0]]

# List of compounds to be removed from the Compounds dictionary
remove = ['SiH', 'CaF', 'SiS', 'BeH', 'HF', 'NH', 'SiH2', 'AlF', 'SH', 'CH', 'AlH', 'TiH', 'CaH', 'LiF', 'MgH', 'ClO']
remove.append('H2S')
remove.append('CH3Br')

# Remove specified compounds from the Compounds dictionary
for r in remove:
    Compounds.pop(r)

# Define broadening and regularization constants
broadening_constant = 0.241
regularisation_constant = 10**(-5)

# Generate reference and observed spectra, and update the Compounds dictionary
ref_spec, obs_spec, full_ref_spec, Compounds = generateData(Compounds, base_path + dataset, broadening_constant, T, P, dataset)

# Perform Lasso inversion and remove compounds not present
ref_spec, Compounds, Lasso_Evaluation, full_ref_spec = lasso_inversion(ref_spec, full_ref_spec, obs_spec, Compounds)

# Perform Tikhonov regularization
x_sol, sigma, C = temporally_regularised_inversion(ref_spec, obs_spec, regularisation_constant, dataset, list(Compounds.keys()))

# Calculate modeled and observed residuals
y_model, y, y_model_err = inversion_residual(ref_spec, obs_spec, x_sol, np.sqrt(sigma))

# Squeeze and extract information from the residuals
y_model_wv_squeezed, y_model_time_squeezed = squeeze_Residuals(y_model, y, ref_spec.shape[1])


# Saving Results
np.save('/home/luke/data/Model/results/'+ dataset + '/sol.npy', x_sol)
np.save('/home/luke/data/Model/results/'+ dataset + '/sig.npy', sigma)
np.save('/home/luke/data/Model/results/'+ dataset + '/comp.npy', Compounds)
np.save('/home/luke/data/Model/results/'+ dataset + '/ref.npy', ref_spec)
np.save('/home/luke/data/Model/results/'+ dataset + '/full_ref.npy', full_ref_spec)
np.save('/home/luke/data/Model/results/'+ dataset + '/obs.npy', obs_spec)
with open('/home/luke/data/Model/results/'+ dataset + '/C.pickle', 'wb') as handle:
    pkl.dump(C, handle, protocol=pkl.HIGHEST_PROTOCOL)
np.save('/home/luke/data/Model/results/'+ dataset + '/y_model_wv_squeezed.npy', y_model_wv_squeezed)
np.save('/home/luke/data/Model/results/'+ dataset + '/y_model_time_squeezed.npy', y_model_time_squeezed)
np.save('/home/luke/data/Model/results/'+ dataset + '/lasso_evaluation.npy', Lasso_Evaluation)


# Plot time series of concentration and emissions ratios
PlotTimeSeries('PPM_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), obs_spec.shape[0], dataset)
PlotER_TimeSeries('ER_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), obs_spec.shape[0], 'CO2', dataset)
PlotOPUS_Results('OPUS_result', dataset)

PlotSpectralResiduals(full_ref_spec, np.load('/home/luke/data/Model/results/'+ dataset + '/full_resid_spectra.npy'), np.load('/home/luke/data/Model/results/'+ dataset + '/W_full.npy'), x_sol, sigma, Compounds, dataset)
# Plot residuals both in time and across wavenumbers
#PlotResiduals(y_model_wv_squeezed, y_model_time_squeezed, dataset)

print(Lasso_Evaluation)