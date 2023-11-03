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

#print(Compounds['CO2'])


#x_sol = [abs(i) * 10**6 for i in x_sol]
x_sol = [abs(i) for i in x_sol]

### make below into func
compound_list = list(Compounds.keys())
Nt = obs_spec.shape[0]

num_rows = len(compound_list) // 2  # Calculate the number of rows needed

fig, axs = plt.subplots(num_rows, 2, figsize=(10, 6))

fig.text(0.5, 0.04, 'Time Step', ha='center')
fig.text(0.04, 0.5, 'Concentration (Definitely!) / ppm', va='center', rotation='vertical')


for i, spc in enumerate(compound_list):

    if spc =='H2S':
         continue
    if spc == 'CH3Br':
        continue

    row, col = divmod(i, 2)
    axs[row,col].plot(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt], color = 'red')
    axs[row,col].fill_between(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt] - 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
                            x_sol[i*Nt:(i+1)*Nt] + 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
                            color= "0.8")
    axs[row, col].set_title(spc)
    axs[row, col].grid()

    if spc == 'CO2':
        CO2 = np.array(x_sol[i*Nt:(i+1)*Nt])
    elif spc =='CO':
        CO = np.array(x_sol[i*Nt:(i+1)*Nt])
    elif spc =='CH4':
        CH4 = np.array(x_sol[i*Nt:(i+1)*Nt])
    elif spc =='H2O':
        H2O = np.array(x_sol[i*Nt:(i+1)*Nt])

# CO2 = np.array([i+800 for i in CO2])
# H2O = np.array([i+8000 for i in H2O])
#print(CO2)

plt.savefig('resultbfb.jpg')

# plt.show()

# plt.figure()

# plt.plot(CO/CO2)

# plt.savefig('ER.jpg')

# plt.show()

# plt.figure()

# plt.plot(CH4/CO2)

# plt.savefig('ER2.jpg')

# plt.show()

# plt.figure()

# plt.plot(H2O/CO2)

# plt.savefig('ER3.jpg')

# plt.show()
























# plt.figure()

# plt.plot((CO2)/(CO2+CO))
# plt.savefig('mce.jpg')

# CO2 /= 44
# CO /= 28


# plt.show()

# plt.figure()

# plt.plot((CO2)/(CO2+CO))
# plt.savefig('mce2.jpg')


# plt.show()

# plt.figure()

# plt.plot(CO2/10**4)
# plt.savefig('co2_conc.jpg')

# plt.show()

# plt.figure()

# plt.plot(CO*28/10**5)
# plt.savefig('co_conc.jpg')