import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy

from tqdm import tqdm

from pathlib import Path

from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum

from sklearn.linear_model import LassoCV

import random

from Toolbox_Processing import *

def read_spectrum(fname):
    """Read spectrum file

    Args:
        fname (str | Path) : The filename

    Returns:
        np.ndarray: The measured spectrum
    """
    return np.loadtxt(fname, usecols=[1])

def interpolate_spectrum(
    wv: np.ndarray, y: np.ndarray, wv_out: np.ndarray
) -> np.ndarray:
    """Interpolate spectrum to reference wavelengths

    Args:
        wv (np.ndarray): input signal wavelengths
        y (np.ndarray): input signal
        wv_out (np.ndarray): interpolate to these wavelengths

    Returns:
        np.ndarray: The (linearly) interpolated spectrum. Same size as `wv_out`.

    """

    return np.interp(wv_out, wv, y)

def read_data(
    spectral_data,
    cutoff= 800,
):
    """Read spectral data.

    Args:
        spectral_data (str | Path): Location of spectral files. Assumed all
            prn files in folder should be read and ordered according to file
            name.
        cutoff (int, optional): Wavelength cut-off. Defaults to 800.

    Returns:
        tuple : spectra, absorption_spectra, species_names, wv
    """
    if isinstance(spectral_data, str):
        spectral_data = Path(spectral_data)


    # Next bit loads up the different recorded spectra
    files = sorted([f for f in spectral_data.glob("*prn")])
    # Read the wavenumbers for the observations.
    # All files have same spectral range
    wv = np.loadtxt(files[0], usecols=[0])
    # Now read in all the actual spectra
    spectra = np.array([read_spectrum(f) for f in tqdm(files)])


    # Remove all wavenumbers before the cutoff. (Useful as the MATRIX spits out jibberish for the first ~800 entries or so)
    wvc = wv[wv > cutoff]
    spectra = spectra[:, wv > cutoff]
    wv = wvc * 1

    return spectra, wv

def generateData(Compounds: dict, path: str, sigma: float):
    """
    Generate data from the Compounds dictionary, path, and broadening constant.

    Args:
        Compounds (dict): A dictionary containing information about chemical species.
        path (str): A string representing the path variable.
        sigma (float): The broadening constant.

    Returns:
        tuple: A tuple containing the reference spectra matrix, residual spectra, and an updated Compounds dictionary.

    This function generates reference data based on the 'Compounds' dictionary and broadening constant 'sigma, the residual spectra
    based on files found in the specified 'path'. It uses various functions to read data, generate a reference
    matrix, and process the spectra. The resulting data is returned as a tuple, and the 'Compounds'
    dictionary may contain additional information about the generated data.

    """
    # Read observed spectra and wavenumber array from the specified path
    spectra_obs, wv_obs = read_data(path)

    T, P = 300, 1.01

    # Generate the reference matrix using the 'getReferenceMatrix' function
    storage_mtx = getReferenceMatrix(Compounds, T, P, wv_obs, sigma)
    print("Reference Matrix Generated!!")

    # Replace NaN values in the reference matrix with zeros
    for i in range(len(storage_mtx)):
        for j in range(len(storage_mtx[i])):
            if np.isnan(storage_mtx[i][j]):
                storage_mtx[i][j] = 0

    # Process observed spectra and wavenumber array to match the reference matrix
    S = np.array([s[~np.all(storage_mtx == 0, axis=0)] for s in spectra_obs])
    W = wv_obs[~np.all(storage_mtx == 0, axis=0)]
    np.save('EmFit_private/results/W.npy', W)

    # Remove background from the processed spectra
    residual_spectra = remove_background(S, W)
    reference_spectra = storage_mtx[:, ~np.all(storage_mtx == 0, axis=0)]

    # Deselect data points with NaN values
    deselect = np.full(len(W), True)

    for i in range(len(residual_spectra)):
        arr = [index for (index, item) in enumerate(residual_spectra[i]) if np.isnan(item)]
        for i in arr:
            deselect[i] = False

    return reference_spectra, residual_spectra, Compounds

def getCompounds(file: str) -> dict:
    """
    Load and return a dictionary of compounds from a binary file.

    Args:
        file (str): The path to the binary file containing the compounds data.

    Returns:
        dict: A dictionary containing information about chemical species.
    """

    with open(file, 'rb') as handle:
        Compounds = pkl.load(handle)
    return Compounds