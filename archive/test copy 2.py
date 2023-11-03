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

x_sol = np.load('sol.npy')
sigma = np.load('sig.npy')
Compounds = np.load('comp.npy', allow_pickle=True).item()
ref_spec = np.load('ref.npy')
obs_spec = np.load('obs.npy')

path = "EmFit_private/spectra/test_series"

Compounds = getCompounds('EmFit_private/Compounds.pickle')

print(Compounds['CO']['bounds'])




Compounds['H2S']['bounds'] = [[4900, 5300], [6000, 6500]]
Compounds['CO']['bounds'] = [[1900, 2300]]
Compounds['CO2']['bounds']= [[2260, 2400]]

for c in Compounds:

    if c =='CO':   
        print(c) 
        bank = Compounds[c]['Source']

        T = 700
        P = 1.01
        for i in range(len(Compounds[c]['bounds'])):
                bound = Compounds[c]['bounds'][i]
                print(c, bound, bank)
                s = calc_spectrum(bound[0], bound[1],         # cm-1
                        molecule=c,
                        isotope='1',
                        pressure=P,   # bar
                        Tgas=T,           # K
                        mole_fraction=10**(-6),
                        path_length=500      # cm
                        )
    

# sig = 0.4
# gam = 10**(-3)

# ref_spec, obs_spec, Compounds = generateData(Compounds, path, sig)

# for i, a in enumerate(list(Compounds.keys())):
#     if a == 'H2S':
#         plt.plot(ref_spec[i])

# plt.savefig('H2S.jpg')