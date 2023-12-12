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

def generateSingleRef(comp, c, W_obs, T, P):

    output = []
    loc = []
    norm_constant = 1 / (np.sqrt(2 * np.pi) * sigma)

    bank = comp['Source']
    tmp = np.zeros_like(W_obs)

    for i in range(len(comp['bounds'])):
        bound = comp['bounds'][i]
        try:
            s = calc_spectrum(
                bound[0], bound[1],  # cm-1
                molecule=c,
                isotope='1',
                pressure=P,  # bar
                Tgas=T,  # K
                mole_fraction=10**(-6),
                path_length=500,  # cm
                warnings={'AccuracyError':'ignore'},
            )
        except Exception as error:
            print("An exception occurred:", error)
            continue

        s.apply_slit(0.241, 'cm-1', shape="gaussian")  # Simulate an experimental slit
        w, A = s.get('absorbance', wunit='cm-1')

        iloc, jloc = np.argmin(np.abs(w.min() - W_obs)), np.argmin(np.abs(w.max() - W_obs))
        s.resample(W_obs[iloc:jloc], energy_threshold=2)

        w, A = s.get('absorbance', wunit='cm-1')

        tmp[iloc:jloc] = A
        loc.append([iloc,jloc])

    ref_mat = np.array(tmp)

    return ref_mat, loc

def cost_function(params, observed_spectra, theoretical_spectra):
    T_values = params[:len(params)//2]
    P_values = params[len(params)//2:]

    # Calculate the sum of squared differences between observed and theoretical spectra
    cost = np.sum((observed_spectra - theoretical_spectra)**2)
    return cost

wv_obs = np.load('/home/luke/data/Model/results/'+ dataset + '/W.npy')
T_guess = np.linspace(273, 473, 100)
P_guess = np.linspace(0.9, 10, 100)
initial_params = np.concatenate([T_guess, P_guess])
# Define bounds for the parameters if needed
bounds = [(0, None)] * len(initial_params)  # Assuming non-negative values for simplicity

for i, spc in enumerate(list(Compounds.keys())):

    print(spc)

    species_arr = x_sol[i * Nt:(i + 1) * Nt]
    ind = np.argmax(species_arr)
    obs_selection = obs_spec[ind]

    loc = generateSingleRef(Compounds[spc], spc, wv_obs, 273, 1.01)[1]

    obs_selection = [element for start, end in loc for element in obs_selection[start:end + 1]]

    #theoretical_spectra = [generateSingleRef(Compounds[spc], spc, wv_obs, T, P)[0] for T, P in zip(T_guess, P_guess)]
    theoretical_spectra = [[element for start, end in loc for element in generateSingleRef(Compounds[spc], spc, wv_obs, T, P)[0][start:end + 1]]
 for T, P in zip(T_guess, P_guess)]
    theoretical_spectra = [LinearRegression().fit(np.array(obs_selection).reshape(-1, 1), spectrum).coef_[0, 0] * spectrum for spectrum in theoretical_spectra]

    # Optimize the cost function using scipy's minimize function
    result = minimize(cost_function, initial_params, args=(obs_selection,theoretical_spectra), bounds=[(273, 473)] * len(T_guess) + [(0.9, 10)] * len(P_guess))
    optimized_params = result.x
    optimized_T = optimized_params[:len(T_guess)]
    optimized_P = optimized_params[len(T_guess):]

    print("Optimized T:", optimized_T)
    print("Optimized P:", optimized_P)

    best_T = np.mean(optimized_T, axis=0)
    best_P = np.mean(best_P, axis=0)
    print("T, P = ", best_T, best_P)