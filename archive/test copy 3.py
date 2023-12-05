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

x_sol = np.load('/home/luke/data/Model/results/sol.npy')
sigma = np.load('/home/luke/data/Model/results/sig.npy')
Compounds = np.load('/home/luke/data/Model/results/comp.npy', allow_pickle=True).item()
ref_spec = np.load('/home/luke/data/Model/results/ref.npy')
full_ref_spec = np.load('/home/luke/data/Model/results/full_ref.npy')
obs_spec = np.load('/home/luke/data/Model/results/obs.npy')
full_obs_spec = np.load('/home/luke/data/Model/results/full_resid_spectra.npy')
W = np.load('/home/luke/data/Model/results/W.npy')
W = np.load('/home/luke/data/Model/results/W.npy')

y_model_wv_squeezed = np.load('/home/luke/data/Model/results/y_model_wv_squeezed.npy')
y_model_time_squeezed = np.load('/home/luke/data/Model/results/y_model_time_squeezed.npy')
C = np.load('/home/luke/data/Model/results/C.pickle', allow_pickle=True)


y_model, y, y_model_err = inversion_residual(ref_spec, obs_spec, x_sol, np.sqrt(sigma))

t = 250
Nl = ref_spec.shape[1]


y_model_sel, y_sel, y_model_err_sel = y_model[t*Nl:(t+1)*Nl], y[t*Nl:(t+1)*Nl], y_model_err[t*Nl:(t+1)*Nl]

for key in Compounds:

    num = len(Compounds[key]['bounds'])
    fig, axs = plt.subplots(num, 2, figsize=(12, 4*num), gridspec_kw={'width_ratios': [3, 1]})
    if num == 1:
        axs = [axs]

    fig.suptitle(key, x=0.05, y=0.95, ha='left', fontsize=16)  # Add figure text at the top left
    fig.text(0.02, 0.5, 'Absorbance', va='center', ha='left', rotation='vertical', fontsize=14)
    fig.text(0.6955, 0.5, 'Observed - Modelled Spectra', va='center', ha='left', rotation='vertical', fontsize=14)

    for i, bound in enumerate(Compounds[key]['bounds']):
        
        indices_within_limits = []

        #for b in Compounds[key]['bounds']:
            #indices_within_limits.append(np.max(y_model_sel[np.where((W >= b[0]) & (W <= b[1]))]))

        # Plot on the left side
        axs[i][0].plot(W, y_sel, color='red', linewidth=4, label='Observed Spectra')
        axs[i][0].plot(W, y_model_sel, '-.', color='k', label='Modelled Spectra')
        axs[i][0].fill_between(W, y_model_sel - y_model_err_sel, y_model_sel + y_model_err_sel, color='gray', alpha=0.5)
        axs[i][0].set_xlim(bound[0], bound[1])
        #axs[i][0].set_ylim(-0.01, np.max(indices_within_limits))
        axs[i][0].tick_params(axis='both', labelsize=12)

        # Add custom title on the right side
        axs[i][0].text(0.75, 1.05, str(bound[0])+' - '+str(bound[1])+' cm$^{-1}$', transform=axs[i][0].transAxes, va='center', ha='left', fontsize=14, rotation='horizontal', fontstyle='italic')

        # Plot histogram on the right side
        diff = y_sel[np.where((W >= bound[0]) & (W <= bound[1]))] - y_model_sel[np.where((W >= bound[0]) & (W <= bound[1]))]
        axs[i][1].hist(diff, bins=20,  alpha=0.7, color='#3498db', edgecolor='black', orientation='horizontal')
        axs[i][1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axs[i][1].grid(axis='x', linestyle='--', alpha=0.7)
        axs[i][1].tick_params(axis='both', labelsize=12)

    axs[len(Compounds[key]['bounds'])-1][0].set_xlabel('Wavenumber / cm$^{-1}$', fontsize=14)
    axs[len(Compounds[key]['bounds'])-1][1].set_xlabel('Frequency', fontsize=14)
    axs[0][0].legend(fontsize=14)

    # Adjust layout to prevent clipping of titles
    plt.tight_layout(rect=[0.03, 0, 0.98, 0.95])
    plt.subplots_adjust(wspace=0.25)

    # Show the plot
    plt.savefig('/home/luke/data/Model/plots/Residuals/' + key + '.png')
    plt.show()
