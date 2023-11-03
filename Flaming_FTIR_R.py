# Import necessary libraries and modules
import numpy as np
from Toolbox_Processing import *
from Toolbox_Reading import *
from Toolbox_Inversion import *
from Toolbox_Display import *

# Define the path to the spectra data
path = "EmFit_private/spectra/test_series"

# Load chemical compound information from a pickle file
Compounds = getCompounds('EmFit_private/Compounds.pickle')

# List of compounds to be removed from the Compounds dictionary
remove = ['SiH', 'CaF', 'SiS', 'BeH', 'HF', 'NH', 'SiH2', 'AlF', 'SH', 'CH', 'AlH', 'TiH', 'CaH', 'LiF', 'MgH', 'ClO']
remove.append('H2S')
remove.append('CH3Br')

# Remove specified compounds from the Compounds dictionary
for r in remove:
    Compounds.pop(r)

# Define broadening and regularization constants
broadening_constant = 0.4
regularisation_constant = 10**(-3)

# Generate reference and observed spectra, and update the Compounds dictionary
ref_spec, obs_spec, Compounds = generateData(Compounds, path, broadening_constant)

# Perform Lasso inversion and remove compounds not present
ref_spec, Compounds, Lasso_Evaluation = lasso_inversion(ref_spec, obs_spec, Compounds)

# Perform Tikhonov regularization
x_sol, sigma, C = temporally_regularised_inversion(ref_spec, obs_spec, regularisation_constant)

# Calculate modeled and observed residuals
y_model, y = inversion_residual(ref_spec, obs_spec, x_sol)

# Squeeze and extract information from the residuals
y_model_wv_squeezed, y_model_time_squeezed = squeeze_Residuals(y_model, y, ref_spec.shape[1])


# Saving Results
np.save('EmFit_private/results/sol.npy', x_sol)
np.save('EmFit_private/results/sig.npy', sigma)
np.save('EmFit_private/results/comp.npy', Compounds)
np.save('EmFit_private/results/ref.npy', ref_spec)
np.save('EmFit_private/results/obs.npy', obs_spec)
with open('EmFit_private/results/C.pickle', 'wb') as handle:
    pkl.dump(C, handle, protocol=pkl.HIGHEST_PROTOCOL)
np.save('EmFit_private/results/y_model_wv_squeezed.npy', y_model_wv_squeezed)
np.save('EmFit_private/results/y_model_time_squeezed.npy', y_model_time_squeezed)
np.save('EmFit_private/results/lasso_evaluation.npy', Lasso_Evaluation)


# Plot time series of concentration and emissions ratios
PlotTimeSeries('PPM_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), obs_spec.shape[0])
PlotER_TimeSeries('ER_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), obs_spec.shape[0], 'CO2')

# Plot residuals both in time and across wavenumbers
PlotResiduals(y_model_wv_squeezed, y_model_time_squeezed)

print(Lasso_Evaluation)