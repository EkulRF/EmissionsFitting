import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy

from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum

from Inversion_Toolbox import *

from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import random

with open('EmFit_private/compoundsCH3OH.pickle', 'rb') as handle:
        Compounds = pkl.load(handle)



reference_spectra, residual_spectra = generateData(Compounds)


Ns, Nl = reference_spectra.shape
Nt = residual_spectra.shape[0]



print("Building A Matrix...")
A = build_A_matrix(reference_spectra, Ns, Nl, 1)





print("Lassoing some coefficients...")

lasso = LassoCV(cv=5, fit_intercept=False)

A_csr = sp.csr_matrix(A)
A_dense = np.array(A_csr.todense())

result = [[] for _ in range(Ns)]

rand_timesteps = [random.randint(0, Nt) for _ in range(int(round(Nt/5)))]

R2 = []
RMSE = []

for i in rand_timesteps:
    # Fit the model
    lasso.fit(A_dense, residual_spectra[i])  # Reshape residual_spectra to 1D

    # Get the coefficients (the fitted sample spectra)
    for i, a in enumerate(lasso.coef_):
        result[i].append(a)
    #fitted_sample_spectra = abs(lasso.coef_.reshape(Ns, Nt))

    scores = cross_val_score(lasso, A_dense, residual_spectra[i], cv=5, scoring='scoring_metric')
    mean_score = scores.mean()
    std_score = scores.std()

    y_pred = lasso.predict(A_dense)
    r_squared = r2_score(residual_spectra[i], y_pred)
    R2.append(abs(r_squared))
    mse = mean_squared_error(residual_spectra[i], y_pred)
    rmse = np.sqrt(mse)
    RMSE.append(rmse)
    print("R2 = ", r_squared)

plt.scatter(rand_timesteps, R2)
plt.ylim(0,1)
plt.savefig('r2.png')
plt.show()

plt.scatter(rand_timesteps, RMSE)
plt.savefig('RMSE.png')
plt.show()


present_compounds = []

for i in range(Ns):
     if sum(result[i]) != 0:
        present_compounds.append(list(Compounds.keys())[i])

new_Compounds = {key: Compounds[key] for key in present_compounds}
compound_list = list(new_Compounds.keys())

reference_spectra = np.array([list(a) for i, a in enumerate(reference_spectra) if sum(result[i])!=0])



x_sol,sigma, C = temporally_regularised_inversion(reference_spectra, residual_spectra, 0.00001)

##### To PPM

for i, spc in enumerate(compound_list):
    f_d = new_Compounds[spc]['PPM_RelativeAbsorbance']
    f_d[0].insert(0,0)
    f_d[1].insert(0,0)
    f =  scipy.interpolate.interp1d(f_d[1], f_d[0])
    for j in range(len(x_sol[i*Nt:(i+1)*Nt])):
        sigma[i*Nt:(i+1)*Nt][j] *= (f(abs(x_sol[i*Nt:(i+1)*Nt][j]))/abs(x_sol[i*Nt:(i+1)*Nt][j]))
        x_sol[i*Nt:(i+1)*Nt][j] = f(abs(x_sol[i*Nt:(i+1)*Nt][j]))

np.save('x_sol.npy', x_sol)
np.save('sigma.npy', sigma)

##### Plotting

num_rows = len(compound_list) // 2  # Calculate the number of rows needed

fig, axs = plt.subplots(num_rows+1, 2, figsize=(10, 6))

for i, spc in enumerate(compound_list):
    row, col = divmod(i, 2)
    axs[row,col].plot(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt], color = 'red')
    axs[row,col].fill_between(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt] - 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
                            x_sol[i*Nt:(i+1)*Nt] + 0.5*np.sqrt(sigma[i*Nt:(i+1)*Nt]),
                            color= "0.8")
    axs[row, col].set_title(spc)

plt.savefig('result.jpg')
