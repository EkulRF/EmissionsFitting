import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import pickle as pkl
# import scipy

from Toolbox_Processing import *
from Toolbox_Reading import *
from Toolbox_Inversion import *
from Toolbox_Display import *


path = "EmFit_private/spectra/test_series"

Compounds = getCompounds('EmFit_private/Compounds.pickle')

remove = ['SiH', 'CaF', 'SiS', 'BeH', 'HF',  'NH', 'SiH2', 'AlF', 'SH', 'CH', 'AlH', 'TiH', 'CaH', 'LiF', 'MgH', 'ClO']

remove.append('H2S')
remove.append('CH3Br')

for r in remove:
    Compounds.pop(r)

broadening_constant = 0.4
regularisation_constant = 10**(-3)

ref_spec, obs_spec, Compounds = generateData(Compounds, path, broadening_constant)

#Lasso Inversion - removes compounds not present
ref_spec, Compounds, A, Lasso_Evaluation = lasso_inversion(ref_spec, obs_spec, Compounds)
# Should have option to say whether we want lasso regression stats???

#Tikhonov Regularisation
x_sol, sigma, C = temporally_regularised_inversion(ref_spec, obs_spec, regularisation_constant)

y_model, y = inversion_residual(ref_spec, obs_spec, x_sol)
y_model_wv_squeezed, y_model_time_squeezed = squeeze_Residuals(y_model, y, ref_spec.shape[1])

PlotTimeSeries('PPM_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), obs_spec.shape[0])
PlotER_TimeSeries('ER_TimeSeries', list(Compounds.keys()), x_sol, np.sqrt(sigma), obs_spec.shape[0], 'CO2')
PlotResiduals(y_model_wv_squeezed, y_model_time_squeezed)




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

print(Lasso_Evaluation)