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
obs_spec = np.load('EmFit_private/results/obs.npy')
W = np.load('W.npy')


with open('EmFit_private/results/C.pickle', 'rb') as handle:
    C = pkl.load(handle)



c_inv = np.linalg.inv(C.toarray())
std_devs = np.sqrt(np.diag(c_inv))

# Use broadcasting to calculate the correlation matrix
inv_corr = c_inv/ np.outer(std_devs, std_devs)


#plt.imshow(inv_corr, vmin=-0.1,vmax=.1, cmap=plt.cm.RdBu)
#plt.savefig('posterior.png')

# from statsmodels.tsa.stattools import grangercausalitytests

data = x_sol.reshape(-1, 567)

from scipy.stats import kendalltau


# print(Compounds.keys())


for i in range(len(list(Compounds.keys()))):
    tau, p_value = kendalltau(data[5], data[i])
    print(p_value, tau)


# data = x_sol.reshape(-1, 567)
# tau, p_value = kendalltau(data[5], data[2])









# lasso_inversion = np.load('EmFit_private/results/lasso_evaluation.npy', allow_pickle=True).item()

# print(lasso_inversion['Cross Validation Score'])

# plt.plot(lasso_inversion['Timesteps'], lasso_inversion['Cross Validation Score'][0])
# plt.savefig('lasso.jpg')

# c_inv = np.linalg.inv(C.toarray())
# plt.imshow(c_inv, interpolation="nearest", cmap=plt.get_cmap("RdBu"))
# plt.savefig('Cov.jpg')
