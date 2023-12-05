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


x_sol = np.load('/home/luke/lukeflamingradis/EmFit_private/results/sol.npy')
sigma = np.load('/home/luke/lukeflamingradis/EmFit_private/results/sig.npy')
Compounds = np.load('/home/luke/lukeflamingradis/EmFit_private/results/comp.npy', allow_pickle=True).item()
ref_spec = np.load('/home/luke/lukeflamingradis/EmFit_private/results/ref.npy')
obs_spec = np.load('/home/luke/lukeflamingradis/EmFit_private/results/obs.npy')
W = np.load('/home/luke/lukeflamingradis/EmFit_private/results/W.npy')


with open('/home/luke/data/Model/results/C.pickle', 'rb') as handle:
    C = pkl.load(handle)


t_step = 250
PlotSpectralResiduals(t_step, np.load('/home/luke/data/Model/results/full_ref.npy'), np.load('/home/luke/data/Model/results/full_resid_spectra.npy'), np.load('/home/luke/data/Model/results/W_full.npy'), np.load('/home/luke/data/Model/results/sol.npy'), np.load('/home/luke/data/Model/results/sig.npy'), Compounds)
# Plot residuals both in time and across wavenumbers

import os
import re
from sklearn.metrics import mean_squared_error

directory = "/home/luke/data/Model/results_param_T/"

# Create a dictionary to store files based on their numbers
files_by_number = {}

for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        match = re.search(r'(lasso_)?(\d+(\.\d+)?)\.npy', filename)
        if match:
            prefix = match.group(1) or ''  # Use empty string if the prefix is not present
            number = match.group(2)
            key = f"{prefix}{number}"
            files_by_number.setdefault(key, []).append(filename)

T = []
RMSE = []

# Print or process the files with the same numbers
for key, files in files_by_number.items():
    print(f"Files with key {key}:")
    T.append(float(key))
    for file in files:
        if file.startswith('y'):
            resid = np.load(directory + file).flatten()
    RMSE.append(np.sqrt(np.sum(resid**2)/len(resid)))


    # You can perform further operations with 'key' and 'files' if needed
    print("\n")

plt.scatter(T, RMSE)
plt.savefig('T.png')