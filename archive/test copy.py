import numpy as np
import math
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
from Toolbox_Display import *

x_sol = np.load('EmFit_private/results/sol.npy')
sigma = np.load('EmFit_private/results/sig.npy')
Compounds = np.load('EmFit_private/results/comp.npy', allow_pickle=True).item()
ref_spec = np.load('EmFit_private/results/ref.npy')
full_ref_spec = np.load('EmFit_private/results/full_ref.npy')
obs_spec = np.load('EmFit_private/results/obs.npy')
full_obs_spec = np.load('EmFit_private/results/full_resid_spectra.npy')
W = np.load('EmFit_private/results/W.npy')
W_full = np.load('EmFit_private/results/W_full.npy')

y_model_wv_squeezed = np.load('EmFit_private/results/y_model_wv_squeezed.npy')
y_model_time_squeezed = np.load('EmFit_private/results/y_model_time_squeezed.npy')
C = np.load('EmFit_private/results/C.pickle', allow_pickle=True)


y_model, y, y_model_err = inversion_residual(full_ref_spec, full_obs_spec, x_sol, np.sqrt(sigma))

t = 200
Nl = full_ref_spec.shape[1]


y_model_sel, y_sel, y_model_err_sel = y_model[t*Nl:(t+1)*Nl], y[t*Nl:(t+1)*Nl], y_model_err[t*Nl:(t+1)*Nl]

for key in Compounds:

    num = len(Compounds[key]['bounds'])
    fig, axs = plt.subplots(num, 1, figsize=(8, 4*num))
    if num == 1:
        axs = [axs]

    fig.suptitle(key, x=0.05, y=0.95, ha='left', fontsize=16)  # Add figure text at the top left
    fig.text(0.02, 0.5, 'Absorbance', va='center', ha='left', rotation='vertical', fontsize=14)

    for i, bound in enumerate(Compounds[key]['bounds']):
        
        indices_within_limits = []

        for b in Compounds[key]['bounds']:
            indices_within_limits.append(np.max(y_model_sel[np.where((W_full >= b[0]) & (W_full <= b[1]))]))

        axs[i].plot(W_full, y_sel, color='red', linewidth=4, label='Observed Spectra')
        axs[i].plot(W_full, y_model_sel, '-.', color='k', label='Modelled Spectra')
        axs[i].fill_between(W_full, y_model_sel - y_model_err_sel, y_model_sel + y_model_err_sel, color='gray', alpha=0.5)
        axs[i].set_xlim(bound[0], bound[1])
        axs[i].set_ylim(-0.01, np.max(indices_within_limits)+0.01)
        axs[i].tick_params(axis='both', labelsize=12)

        # Add custom title on the right side
        axs[i].text(0.75, 1.05, str(bound[0])+' - '+str(bound[1])+' cm$^{-1}$', transform=axs[i].transAxes, va='center', ha='left', fontsize=14)

    axs[len(Compounds[key]['bounds'])-1].set_xlabel('Wavenumber / cm$^{-1}$', fontsize=14)
    axs[0].legend(fontsize=14)

    # Adjust layout to prevent clipping of titles
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])

    # Show the plot
    plt.savefig('EmFit_private/plot/Resid/' + key + '.png')
    plt.show()




#PlotResiduals(y_model_wv_squeezed, y_model_time_squeezed)

# def read_hitran_xsc(file_path):
#     data = np.loadtxt(file_path, skiprows=3)  # Skip the first 3 lines containing header information
#     wavenumbers = data[:, 0]
#     absorption = data[:, 1]
#     return wavenumbers, absorption

# # Replace 'your_hitran_xsc_file.xsc' with the path to your HITRAN XSC file
# hitran_xsc_file_path = 'EmFit_private/Manual_Spectra/Acetone/CH3COCH3_297.8_700.0_700.0-1780.0_13.xsc'

# wavenumbers, absorption = read_hitran_xsc(hitran_xsc_file_path)

# plt.plot(wavenumbers, absorption)
# plt.savefig('Ace.png')

# import re 
# import os

# filepath = 'EmFit_private/Manual_Spectra/Acetone/CH3COCH3_297.8_700.0_700.0-1780.0_13.xsc'

# def generateReferenceFromFile(file_path):

#     filename = os.path.basename(file_path)

#     float_range = re.search(r'_(\d+\.\d+)-(\d+\.\d+)_', filename)
#     wv_start, wv_end = float(float_range.group(1)), float(float_range.group(2))

#     float_string = filename[-8:-4] 
#     wv_interval = float(float_string[:1] + "." + float_string[2:])

#     print(len(np.arange(wv_start, wv_end, wv_interval)))

#     data = np.loadtxt(file_path, skiprows=1)

#     return np.linspace(wv_start,wv_end, num = len(data.flatten())), data.flatten()

# w, A = generateReferenceFromFile(filepath)

# s = Spectrum.from_array(w, A, 'absorbance', wunit='cm-1', unit='')



# c_inv = np.linalg.inv(C.toarray())
# std_devs = np.sqrt(np.diag(c_inv))

# # Use broadcasting to calculate the correlation matrix
# inv_corr = c_inv/ np.outer(std_devs, std_devs)

# plot_inv = np.where(inv_corr > 0, np.log10(inv_corr), -np.log10(-inv_corr))
# plt.imshow(plot_inv, cmap='gnuplot')

# n = 15
# row_labels = list(Compounds.keys())
# plt.xticks(np.linspace(0, inv_corr.shape[0], n), row_labels, rotation = 45)
# plt.yticks(np.linspace(0, inv_corr.shape[1], n), row_labels, rotation = 0)

# plt.colorbar()

# plt.savefig('EmFit_private/plot/Correlation_Matrix.jpg')
# plt.close()