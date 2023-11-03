import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pathlib import Path
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy

from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import random

from Toolbox_Processing import *

def temporally_regularised_inversion(
    reference_spectra,
    residual_spectra,
    lambda_,
    post_cov = True,
    do_spilu = True,
):
    """Temporally regularised inversion using selected bassis functions.

    Args:
        absorption_spectra (np.ndarray): The absorption spectra, shape (Ns, Nl)
        residual_spectra (np.ndarray): The residuals of the transmiatted spectra,
            shape (Nt, Nl)
        lambda_ (float): The amount of regularisation. 0.005 seems to work?
        post_cov (boolean, optional): Return inverse posterior covariance matrix.
            Defaults to True.
        do_spilu (boolean, optional): Solve the system using and ILU factorisation.
            Seems faster and more memory efficient, with an error around 0.5-1%

    Returns:
        Maximum a poseteriori estimate, and variance. Optionally, also
        the posterior inverse covariance matrix.
    """
    Ns, Nl = reference_spectra.shape
    Nt = residual_spectra.shape[0]
    assert Ns != residual_spectra.shape[1], "Wrong spectral sampling!!"
    # Create the "hat" matrix
    A_mat = build_A_matrix(reference_spectra, Ns, Nl, Nt)
    #print(np.linalg.cond(A_mat.todense()))
    # Regulariser
    D_mat = sp.lil_matrix(sp.kron(sp.eye(Ns), create_smoother(Nt)))
    # Squeeze observations
    y = residual_spectra.flatten()
    # Solve
    C = sp.csc_array(A_mat.T @ A_mat + lambda_ * D_mat)
    

    c_inv = np.linalg.inv(C.toarray())

    sigma = c_inv.diagonal()

    cobj = spl.spilu(C)
    x_sol = cobj.solve(A_mat.T @ y) if do_spilu else spl.spsolve(C, A_mat.T @ y)


    #s = np.sqrt(s.diagonal())
    #print("Plotting covariance")
    #plt.imshow(s, interpolation="nearest", cmap=plt.get_cmap("RdBu"))
    #plt.savefig('Cov.jpg')


    return (x_sol, sigma, C) if post_cov else (x_sol, sigma)

def inversion_residual(ref_spec, obs_spec, x_sol):

    Nl = ref_spec.shape[1]
    Nt = obs_spec.shape[0]

    y = obs_spec.flatten()

    y_model = np.zeros_like(y)

    for i, r in enumerate(ref_spec):
        for j in range(Nt):
            y_model[j*Nl:(j+1)*Nl] += r * x_sol[i*Nt:(i+1)*Nt][j]

    return y_model, y


def lasso_inversion(
    reference_spectra,
    residual_spectra,
    Compounds
    ):
    
    (Ns, Nl), Nt = (reference_spectra.shape), residual_spectra.shape[0]

    A = build_A_matrix(reference_spectra, Ns, Nl, 1)
    
    lasso = LassoCV(cv=5, fit_intercept=False)

    A_csr = sp.csr_matrix(A)
    A_dense = np.array(A_csr.todense())

    result = [[] for _ in range(Ns)]

    rand_timesteps = [random.randint(0, Nt-1) for _ in range(int(round(Nt/10)))]

    Cross_Val_Score = [[] for _ in range(3)]
    R2, RMSE = [], []

    for i in rand_timesteps:
        lasso.fit(A_dense, residual_spectra[i])

        for i, a in enumerate(lasso.coef_):
            result[i].append(a)

        # Analysing regression
        scores = cross_val_score(lasso, A_dense, residual_spectra[i], cv=5, scoring='neg_mean_absolute_error')

        Cross_Val_Score[0].append(scores.mean())
        Cross_Val_Score[1].append(scores.std())
        Cross_Val_Score[2].append(scores)


        y_pred = lasso.predict(A_dense)

        R2.append(abs(r2_score(residual_spectra[i], y_pred)))
        RMSE.append(np.sqrt(mean_squared_error(residual_spectra[i], y_pred)))
        #####

    Lasso_Evaluation = {'Timesteps': rand_timesteps, 'Cross Validation Score': Cross_Val_Score, 'R2': R2, 'RMSE': RMSE}

    present_compounds = []

    for i in range(Ns):
        if sum(result[i]) != 0:
            present_compounds.append(list(Compounds.keys())[i])

    new_Compounds = {key: Compounds[key] for key in present_compounds}
    compound_list = list(new_Compounds.keys())

    reference_spectra = np.array([list(a) for i, a in enumerate(reference_spectra) if sum(result[i])!=0])

    (Ns, Nl), Nt = (reference_spectra.shape), residual_spectra.shape[0]

    A = build_A_matrix(reference_spectra, Ns, Nl, 1)
        
    return reference_spectra, new_Compounds, A, Lasso_Evaluation