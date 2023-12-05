# Import necessary libraries and modules
import os
import numpy as np
from Toolbox_Processing import *
from Toolbox_Reading import *
from Toolbox_Inversion import *
from Toolbox_Display import *

# Define the path to the spectra data
#path = "/home/luke/lukeflamingradis/EmFit_private/spectra/test_series"
base_path = "/home/luke/data/MATRIX_data/"
dataset = "Peat6"

os.makedirs('/home/luke/data/Model/results_param/'+dataset+'/', exist_ok=True)

P, T = getPT(dataset)
#P=1.01

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
#broadening_constant = 1
regularisation_constant = 10**(-3)

broad_array = np.logspace(-2,2, 50)

for broadening_constant in broad_array:

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
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/sol_' + str(broadening_constant) + '.npy', x_sol)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/sig_' + str(broadening_constant) + '.npy', sigma)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/comp_' + str(broadening_constant) + '.npy', Compounds)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/ref_' + str(broadening_constant) + '.npy', ref_spec)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/full_ref_' + str(broadening_constant) + '.npy', full_ref_spec)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/obs_' + str(broadening_constant) + '.npy', obs_spec)
    with open('/home/luke/data/Model/results_param/'+ dataset + '/C_' + str(broadening_constant) + '.pickle', 'wb') as handle:
        pkl.dump(C, handle, protocol=pkl.HIGHEST_PROTOCOL)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/y_model_wv_squeezed_' + str(broadening_constant) + '.npy', y_model_wv_squeezed)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/y_model_time_squeezed_' + str(broadening_constant) + '.npy', y_model_time_squeezed)
    np.save('/home/luke/data/Model/results_param/'+ dataset + '/lasso_evaluation_' + str(broadening_constant) + '.npy', Lasso_Evaluation)

# Create a dictionary to store files based on their numbers
files_by_number = {}

for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        match = re.search(r'(\d+(\.\d+)?)\.npy', filename)
        if match:
            number = match.group(1)
            files_by_number.setdefault(number, []).append(filename)


constant = []
L2 = []
RSS = []
RMSE = []

for number, files in files_by_number.items():
    print(f"Files with number {number}:")
    constant.append(float(number))
    for file in files:
        if file.startswith("ref"):
            ref = np.load(directory + file)
        elif file.startswith("sol"):
            x_sol = np.load(directory + file)
        elif file.startswith("sig"):
            sig = np.load(directory + file)
        elif file.startswith("y_model_t"):
            res = np.load(directory + file)
    

    #y_model, y, y_model_err = inversion_residual(ref, obs_spec[:len(ref)], x_sol, np.sqrt(sigma))

    RSS.append(np.sum((res.flatten())**2))
    RMSE.append(np.sqrt(np.sum((res.flatten())**2)/len(res.flatten())))
    L2.append(np.sqrt(np.sum(x_sol**2)))

#Taking the log of the 'constant' array for log scale
log_constant = np.log10(constant)

# Plotting
plt.scatter(L2, RSS, c=log_constant, cmap='viridis')  # You can choose a different colormap
plt.colorbar(label='log(Constant)')  # Add colorbar with label

# Add labels and title
plt.xlabel('L2')
plt.ylabel('RSS')
plt.title('Broadening Constant, $\sigma_{\\text{b}}$, L-curve')

plt.savefig('/home/luke/data/Model/plots/' + dataset + '/Broadening_L_curve.png')

plt.show()

plt.figure()

plt.scatter(constant, RSS, color = 'purple')
plt.xscale('log')
plt.ylabel('RSS')
plt.xlabel('Broadening Constant, $\sigma_{\\text{b}}$')


plt.savefig('/home/luke/data/Model/plots/' + dataset + '/Broadening_L_curve2.png') 

plt.figure()

plt.scatter(constant, RMSE, color = 'purple')
plt.xscale('log')
plt.ylabel('RMSE')
plt.xlabel('Broadening Constant, $\sigma_{\\text{b}}$')


plt.savefig('/home/luke/data/Model/plots/' + dataset + '/Broadening_L_curve3.png') 