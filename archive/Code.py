
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy
from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum

from EmFit_private.Inversion_Toolbox import *

def isNaN(num):
    return num != num

def generateData():

    bounds = {
    "CO2": (6250, 6600),
    "CH4": [9100, 9700],
    "H2O": [5500, 5900],
    "CO": [2500, 4000],
}
    selected_spectra = list(bounds.keys())

    (spectra_obs, absorption_spectra, species_names, wv_obs) = read_data(
        selected_spectra,
        "EmFit_private/spectra/test_series", # Location of the series we want to invert.
        "EmFit_private/spectra/templates/" # Location of the individual species "template" absorption
    )

    T, P = 300, 1.01

    with open('EmFit_private/compoundsCH3OH.pickle', 'rb') as handle:
        Compounds = pkl.load(handle)

    new = {}

    for c in ['NO2', 'CO', 'NO', 'CO2', 'CH4', 'H2O']:

        new[c] = Compounds[c]

    Compounds = new

    storage_mtx = getReferenceMatrix(Compounds, T, P, wv_obs)

    for i in range(len(storage_mtx)):
        for j in range(len(storage_mtx[i])):
            if isNaN(storage_mtx[i][j]):
                storage_mtx[i][j] = 0

    S = np.array([s[~np.all(storage_mtx == 0, axis=0)] for s in spectra_obs])

    S = np.array([-np.log(np.abs(x)) for x in S])
    W = wv_obs[~np.all(storage_mtx == 0, axis=0)]
    residual_spectra = remove_background(S, {}, W)

    #residual_spectra = np.array([-np.log(x) for x in residual_spectra])

    storage_mtx = storage_mtx[:,~np.all(storage_mtx == 0, axis=0)]
    reference_spectra = storage_mtx[:, ~np.all(storage_mtx == 0, axis=0)]

    deselect = np.full(len(W), True)

    for i in range(len(residual_spectra)):
        arr = [index for (index, item) in enumerate(residual_spectra[i]) if item != item]
        for i in arr:
            deselect[i] = False
    return reference_spectra, residual_spectra

def getReferenceMatrix(Compounds, T, P, W_obs):
    
    output = []
    
    for c in Compounds:
        
        bank = Compounds[c]['Source']
        
        tmp = np.zeros_like(W_obs)
        
        for i in range(len(Compounds[c]['bounds'])):
            bound = Compounds[c]['bounds'][i]
            try:
                print(c, bound, bank)
                s = calc_spectrum(bound[0], bound[1],         # cm-1
                          molecule=c,
                          isotope='1',
                          pressure=P,   # bar
                          Tgas=T,           # K
                          mole_fraction=10**(-4),
                          path_length=500,      # cm
                          databank=bank,  # or 'hitemp', 'geisa', 'exomol'
                          )
            except:
                print("BAD", c)
                continue
            s.apply_slit(0.241, 'cm-1', shape="gaussian")       # simulate an experimental slit
            w, A = s.get('absorbance', wunit='cm-1')
           
            iloc, jloc = np.argmin(np.abs(w.min() - W_obs)), np.argmin(np.abs(w.max() - W_obs))
            s.resample(W_obs[iloc:jloc], energy_threshold=2)
            
            w, A = s.get('absorbance', wunit='cm-1')

            tmp[iloc:jloc] = A
            
        output.append(tmp)
    
    ref_mat = np.array(output)
    
    return ref_mat
print("Gen Data")
reference_spectra, residual_spectra = generateData()
print("Done That!")
for i in range(len(reference_spectra)):
    print("Plotting Ref")
    plt.plot(reference_spectra[i])
    #plt.savefig(str(i)+'.png')
print("Calculation Ongoing")
x_sol,sigma, C = temporally_regularised_inversion(reference_spectra, residual_spectra,
                                                  0.00001)

print("HAPPY DAYS!")

print(x_sol)

#####
# Converts to ppm

nt = residual_spectra.shape[0]

with open('EmFit_private/compoundsCH3OH.pickle', 'rb') as handle:         
   Compounds = pkl.load(handle)

for i, spc in enumerate(list(Compounds.keys())):
    f_d = Compounds[spc]['PPM_RelativeAbsorbance']
    f_d[0].insert(0,0)
    f_d[1].insert(0,0)
    f =  scipy.interpolate.interp1d(f_d[1], f_d[0])
    for j in range(len(x_sol[i*nt:(i+1)*nt])):
        sigma[i*nt:(i+1)*nt][j] *= (f(abs(x_sol[i*nt:(i+1)*nt][j]))/abs(x_sol[i*nt:(i+1)*nt][j]))
        x_sol[i*nt:(i+1)*nt][j] = f(abs(x_sol[i*nt:(i+1)*nt][j]))

#####

fig, axs = plt.subplots(nrows=13, ncols=2, figsize= (30,15),sharex=True)
axs = axs.flatten()
nt = residual_spectra.shape[0]
#colours = getDistinctColors(len(labels_x))

with open('EmFit_private/compoundsCH3OH.pickle', 'rb') as handle:
    Compounds = pkl.load(handle)

new = {}

for c in ['NO2', 'CO', 'NO', 'CO2', 'CH4', 'H2O']:

    new[c] = Compounds[c]

Compounds = new

for i, spc in enumerate(list(Compounds.keys())):
    print(i,spc)
    try:
        axs[i].plot(np.arange(nt), x_sol[i*nt:(i+1)*nt], color = 'red')
        axs[i].fill_between(np.arange(nt), x_sol[i*nt:(i+1)*nt] - 0.5*np.sqrt(sigma[i*nt:(i+1)*nt]),
                            x_sol[i*nt:(i+1)*nt] + 0.5*np.sqrt(sigma[i*nt:(i+1)*nt]),
                            color= "0.8")
        axs[i].set_title(spc, fontsize=20)
        axs[i].tick_params(labelsize=20)
    except:
        print(spc)
        continue
    
fig.text(0.5, -0.001, 'Time Step', ha='center', fontsize=20)
fig.text(-0.003, 0.5, '$\sim$ Concentration (ish)', va='center', rotation='vertical', fontsize=20)

fig.tight_layout()
plt.savefig('ex2.png')
plt.show()