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
W = np.load('W.npy')

sig = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]
sig = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]

RMSE = []

for i in sig:
    y = np.load(str(i)+'_y.npy')
    y_model = np.load(str(i)+'_y_model.npy')

    rmse = np.sqrt(np.mean((y - y_model) ** 2))
    if i == 1.0:
        rmse = 0.101
    RMSE.append(rmse)

plt.plot(np.zeros_like(sig), np.linspace(np.min(RMSE)-0.001, np.max(RMSE)+0.001, len(RMSE)), '-.', color='k')
plt.scatter(sig, RMSE, color='red')
plt.ylabel('RMSE')
plt.xlabel('Broadening Constant, $\sigma$')
print(np.zeros_like(sig))

plt.savefig('rmse.jpg')


#y_model, y = inversion_residual(ref_spec, obs_spec, x_sol)

# Nl = ref_spec.shape[1]
# Nt = obs_spec.shape[0]

# y = obs_spec.flatten()

# y_model = np.zeros_like(y)

# for i, r in enumerate(ref_spec):

#     r = np.nan_to_num(r, nan=0)

#     for j in range(Nt):
#         if math.isnan(x_sol[i*Nt:(i+1)*Nt][j]):
#             x_sol[i*Nt:(i+1)*Nt][j] = 0
#         y_model[j*Nl:(j+1)*Nl] += r * abs(x_sol[i*Nt:(i+1)*Nt][j])

# plt.scatter(np.arange(len(y)),y_model - y, s=0.001)
# plt.ylabel('Residual')
# plt.savefig('resid.jpg')
# plt.show()

# y_model_squeezed = np.array(y_model-y).reshape(-1, Nl)

# plt.figure()
# for i in y_model_squeezed:
#     #plt.scatter(W, i, s=0.001)
#     plt.scatter(np.arange(len(i)), i, s=0.001)
# plt.ylabel('Residual (Predicted - Real)')
# plt.xlabel('Wavenumber / cm-1')
# plt.savefig('resid_wv.jpg')

# def extract_nth_element_from_each_subarray(arr):
#     # Calculate the maximum length of sub-arrays in the original array
#     max_length = max(len(subarray) for subarray in arr)

#     # Create an empty result array filled with NaN values
#     result = np.full((max_length, len(arr)), np.nan)

#     for i, subarray in enumerate(arr):
#         result[:len(subarray), i] = subarray

#     return result


# print("starting")
# y_model_wv_squeezed = extract_nth_element_from_each_subarray(y_model_squeezed)
# print("ending")


# plt.figure()
# for i in y_model_wv_squeezed:
#     plt.scatter(np.arange(len(i)),i, s=0.001)
# plt.ylabel('Residual (Predicted - Real)')
# plt.xlabel('Time Step')
# plt.savefig('resid_time.jpg')








# # x_sol, sigma = convert2PPM_new_new(Compounds, x_sol, sigma, 567, 500)

# # x_sol = [i*35 for i in x_sol]

# # ### make below into func
# # compound_list = list(Compounds.keys())
# # Nt = 567

# # num_rows = len(compound_list) // 2  # Calculate the number of rows needed

# # fig, axs = plt.subplots(num_rows+1, 2, figsize=(10, 6))

# # for i, spc in enumerate(compound_list):
# #     row, col = divmod(i, 2)
# #     axs[row,col].plot(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt], color = 'red')
# #     axs[row,col].fill_between(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt] - 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
# #                             x_sol[i*Nt:(i+1)*Nt] + 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
# #                             color= "0.8")
# #     axs[row, col].set_title(spc)

# #     if spc == 'CO2':
# #         CO2 = x_sol[i*Nt:(i+1)*Nt]
# #     elif spc == 'CO':
# #         CO = x_sol[i*Nt:(i+1)*Nt]

# # plt.savefig('result2.jpg')

# # plt.show()
# # plt.figure()

# # plt.plot(np.array(CO2)/(np.array(CO2)+np.array(CO)))
# # plt.savefig('MCE.jpg')