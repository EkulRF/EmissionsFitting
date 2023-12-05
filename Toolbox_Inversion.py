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
    dataset,
    compound_list,
    post_cov = True,
    do_spilu = True,
    ):
    """
    Temporally regularised inversion using selected bassis functions.

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
    print('Performing Tikhonov Regularisation')

    Ns, Nl = reference_spectra.shape
    Nt = residual_spectra.shape[0]
    assert Ns != residual_spectra.shape[1], "Wrong spectral sampling!!"
    # Create the "hat" matrix
    A_mat = build_A_matrix(reference_spectra, Ns, Nl, Nt)
    # Regulariser
    D_mat = sp.lil_matrix(sp.kron(sp.eye(Ns), create_smoother(Nt)))
    # Squeeze observations
    y = residual_spectra.flatten()
    # Solve
    C = sp.csc_array(A_mat.T @ A_mat + lambda_ * D_mat)
    

    c_inv = np.linalg.inv(C.toarray())
    std_devs = np.sqrt(np.diag(c_inv))

    # Use broadcasting to calculate the correlation matrix
    inv_corr = c_inv/ np.outer(std_devs, std_devs)

    plot_inv = np.where(inv_corr > 0, np.log10(inv_corr), -np.log10(-inv_corr))
    plt.imshow(plot_inv, cmap='gnuplot')

    ticks = np.linspace(inv_corr.shape[0]/(2*len(compound_list)), inv_corr.shape[0] * (1 - (1/(2*len(compound_list)))), len(compound_list))
    plt.xticks(ticks, compound_list, rotation=45)
    plt.yticks(ticks, compound_list)

    plt.colorbar()

    plt.tight_layout()

    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/Correlation_Matrix.png')
    plt.close()

    sigma = c_inv.diagonal()

    cobj = spl.spilu(C)
    x_sol = cobj.solve(A_mat.T @ y) if do_spilu else spl.spsolve(C, A_mat.T @ y)

    return (x_sol, sigma, C) if post_cov else (x_sol, sigma)

def inversion_residual(ref_spec, obs_spec, x_sol, x_err):
    """
    Derive 1D array versions of model and observed results.

    Args:
        ref_spec (np.ndarray): Reference spectra matrix.
        obs_spec (np.ndarray): Observed spectra matrix.
        x_sol (np.ndarray): Solution data for the compounds over time.
        x_err (np.ndarray): Standard error for each parameter at each timestep.

    Returns:
        tuple: A tuple containing model results (y_model), observed results (y), and model errors (y_model_err).

    This function computes the model results (y_model), observed results (y), and model errors (y_model_err) based on the
    reference spectra, observed spectra, solution data for the compounds over time, and the standard error for each parameter
    at each timestep. The function returns these results as a tuple.

    """
    print('Calculating Residuals')

    Nl = ref_spec.shape[1]
    Nt = obs_spec.shape[0]

    y = obs_spec.flatten()
    y_model = np.zeros_like(y)
    y_model_err = np.zeros_like(y)

    ref_spec_reshaped = ref_spec.reshape(-1, 1, Nl)
    x_sol_reshaped = x_sol.reshape(-1, Nt, 1)
    x_err_reshaped = x_err.reshape(-1, Nt, 1)

    # Calculate the culmulative model output.
    y_model = np.sum(ref_spec_reshaped * x_sol_reshaped, axis=0).reshape(-1)

    # Calculate y_model_err
    y_model_err = np.sqrt(np.sum((ref_spec_reshaped * x_sol_reshaped * x_err_reshaped)**2, axis=0)).reshape(-1)

    return y_model, y, y_model_err

def lasso_inversion(
    reference_spectra: np.ndarray,
    full_reference_spectra: np.ndarray,
    residual_spectra: np.ndarray,
    Compounds: dict
    ):
    """
    Perform Lasso inversion on the observed results.

    Args:
        reference_spectra (np.ndarray): Reference spectra matrix.
        residual_spectra (np.ndarray): Observed residual spectra.
        Compounds (dict): Dictionary containing information about chemical species.

    Returns:
        tuple: A tuple containing the updated reference spectra, updated Compounds dictionary,
        reference A matrix, and Lasso inversion evaluation results.

    This function performs Lasso inversion on the observed residual spectra using reference spectra.
    It evaluates whether each species is present in the inversion. If a species is not present, it is
    removed from the Compounds dictionary, and an updated version of Compounds is returned. The function
    also returns the updated reference spectra, reference A matrix, and the Lasso inversion evaluation results.

    """
    print('Performing Lasso Inversion')

    # Get the dimensions of the reference and observed spectra
    (Ns, Nl), Nt = reference_spectra.shape, residual_spectra.shape[0]

    # Build the reference A matrix
    A = build_A_matrix(reference_spectra, Ns, Nl, 1)

    # Initialize a Lasso regression model with cross-validation
    lasso = LassoCV(cv=5, fit_intercept=False)

    # Convert the A matrix to a sparse matrix and then to a dense array
    A_csr = sp.csr_matrix(A)
    A_dense = np.array(A_csr.todense())

    # Initialize a list to store the coefficients for each time step
    result = [[] for _ in range(Ns)]
    # Initialize lists to store cross-validation scores, R2, and RMSE
    Cross_Val_Score = [[] for _ in range(3)]
    R2, RMSE = [], []

    # Randomly select time steps for analysis
    rand_timesteps = [random.randint(0, Nt-1) for _ in range(int(round(Nt/10)))]

    for i in rand_timesteps:
        # Fit Lasso regression for each selected time step
        lasso.fit(A_dense, residual_spectra[i])

        [result[i].append(a) for i, a in enumerate(lasso.coef_)]

        # Analyzing regression
        scores = cross_val_score(lasso, A_dense, residual_spectra[i], cv=5, scoring='neg_mean_absolute_error')

        Cross_Val_Score[0].append(scores.mean())
        Cross_Val_Score[1].append(scores.std())
        Cross_Val_Score[2].append(scores)

        y_pred = lasso.predict(A_dense)

        R2.append(abs(r2_score(residual_spectra[i], y_pred)))
        RMSE.append(np.sqrt(mean_squared_error(residual_spectra[i], y_pred)))

    # Create a dictionary to store Lasso evaluation results
    Lasso_Evaluation = {'Timesteps': rand_timesteps, 'Cross Validation Score': Cross_Val_Score, 'R2': R2, 'RMSE': RMSE}

    # Determine which compounds are present in the inversion
    present_compounds = [key for key, values in zip(Compounds.keys(), result) if sum(values) != 0]

    # Create an updated Compounds dictionary with only present compounds
    new_Compounds = {key: Compounds[key] for key in present_compounds}
    compound_list = list(new_Compounds.keys())

    # Filter the reference spectra to keep only those with present compounds
    reference_spectra = reference_spectra[[sum(result[i]) != 0 for i in range(Ns)]]
    full_reference_spectra = full_reference_spectra[[sum(result[i]) != 0 for i in range(Ns)]]
    #np.save('EmFit_private/results/full_ref.npy', full_reference_spectra)

    # Get the dimensions of the updated reference spectra
    (Ns, Nl), Nt = reference_spectra.shape, residual_spectra.shape[0]

    return reference_spectra, new_Compounds,  Lasso_Evaluation, full_reference_spectra