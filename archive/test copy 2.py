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

# x_sol = np.load('/home/luke/data/Model/results/sol.npy')
# sigma = np.load('/home/luke/data/Model/results/sig.npy')
# Compounds = np.load('/home/luke/data/Model/results/comp.npy', allow_pickle=True).item()
# ref_spec = np.load('/home/luke/data/Model/results/ref.npy')
# full_ref_spec = np.load('/home/luke/data/Model/results/full_ref.npy')
# obs_spec = np.load('/home/luke/data/Model/results/obs.npy')
# full_obs_spec = np.load('/home/luke/data/Model/results/full_resid_spectra.npy')
# W = np.load('/home/luke/data/Model/results/W.npy')
# W_full = np.load('/home/luke/data/Model/results/W_full.npy')

# y_model_wv_squeezed = np.load('/home/luke/data/Model/results/y_model_wv_squeezed.npy')
# y_model_time_squeezed = np.load('/home/luke/data/Model/results/y_model_time_squeezed.npy')
# C = np.load('/home/luke/data/Model/results/C.pickle', allow_pickle=True)

f = open("/home/luke/data/MATRIX_data/Peat3/(2016_03_04_13_24_37_450)_Run3_progression_ResultSeries.txt", "r")

print(f.read())

import pandas as pd

# Read the data into a DataFrame
df = pd.read_csv("/home/luke/data/MATRIX_data/Peat3/(2016_03_04_13_24_37_450)_Run3_progression_ResultSeries.txt", delim_whitespace=True, skiprows=[0])

# Rename columns
df.columns = [col.replace(".", "_") for col in df.columns]

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.drop(['Date', 'Time'], axis=1)

num_rows = (len(df.columns)-2) // 2  # Calculate the number of rows needed

fig, axs = plt.subplots(num_rows + 1, 2, figsize=(10, 6))

fig.text(0.5, 0.04, 'Time Step', ha='center')
fig.text(0.07, 0.5, 'Concentration / ppm', va='center', rotation='vertical')

plt.subplots_adjust(hspace=0.5)

for i, spc in enumerate(df.columns):

    if spc == 'DateTime':
        continue

    row, col = divmod(i, 2)
    axs[row, col].plot(df['DateTime'], df[spc], color='red')
    axs[row, col].set_title(spc, loc='right')
    axs[row, col].grid()

fig.subplots_adjust(top=0.95)

#plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/' + name + 'OPUS.png')
plt.savefig('OPUS.png')