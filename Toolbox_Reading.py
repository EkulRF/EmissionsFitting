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
    selected_spectra,
    spectral_data,
    template_data,
    cutoff= 800,
):
    """Read spectra and template (e.g. absorption spectra) data.

    Args:
        selected_spectra (list): List of species to read in
        spectral_data (str | Path): Location of spectral files. Assumed all
            prn files in folder should be read and ordered according to file
            name.
        template_data (str | Path): Location of the spectral template
            absorption files. prn extension (as above)
        cutoff (int, optional): Wavelength cut-off. Defaults to 800.

    Returns:
        tuple : spectra, absorption_spectra, species_names, wv
    """
    if isinstance(spectral_data, str):
        spectral_data = Path(spectral_data)
    if isinstance(template_data, str):
        template_data = Path(template_data)

    # Next bit loads up the different recorded spectra
    # over some ~ 550 time steps or so.

    files = sorted([f for f in spectral_data.glob("*prn")])
    print(spectral_data)
    print(len([f for f in spectral_data.glob("*prn")]))
    print(files)
    # Let's read the wavelengths for one spectrum
    # All files have same spectral range
    wv = np.loadtxt(files[0], usecols=[0])
    # Now read in all the actual spectra
    spectra = np.array([read_spectrum(f) for f in tqdm(files)])

    # Luke removes the first 800 samples
    # Dunno why, but I do the same

    wvc = wv[wv > cutoff]
    spectra = spectra[:, wv > cutoff]
    wv = wvc * 1

    # Now, let's go and read in the absorption
    # characteristics of our target species

    files = sorted(
        [
            f
            for f in template_data.glob("*prn")
            if f.name.split("_")[1] in selected_spectra
        ]
    )
    # Again, read in the wavelenth columns
    wvc = np.loadtxt(files[0], usecols=[0])
    # Retrieve species names from filename
    species_names = [f.name.split("_")[1] for f in files]
    # Read & interpolate the the actual spectra
    absorption_spectra = np.array(
        [interpolate_spectrum(wvc, read_spectrum(f), wv) for f in tqdm(files)]
    )
    return spectra, absorption_spectra, species_names, wv

def generateData(Compounds, path, sigma):

    bounds = {
    "CO2": (6250, 6600),
    "CH4": [9100, 9700],
    "H2O": [5500, 5900],
    "CO": [2500, 4000],
}
    selected_spectra = list(bounds.keys())

    (spectra_obs, absorption_spectra, species_names, wv_obs) = read_data(
        selected_spectra,
        path, # Location of the series we want to invert.
        "EmFit_private/spectra/templates/" # Location of the individual species "template" absorption
    )

    T, P = 300, 1.01

    storage_mtx, storage_coef_mtx = getReferenceMatrix(Compounds, T, P, wv_obs, sigma)
    print("ref generated")

    for i in range(len(storage_mtx)):
        for j in range(len(storage_mtx[i])):
            if isNaN(storage_mtx[i][j]):
                storage_mtx[i][j] = 0

    S = np.array([s[~np.all(storage_mtx == 0, axis=0)] for s in spectra_obs])

    W = wv_obs[~np.all(storage_mtx == 0, axis=0)]

    np.save('EmFit_private/results/W.npy', W)


    residual_spectra = remove_background(S, W)

    reference_spectra_coef = storage_coef_mtx[:, ~np.all(storage_mtx == 0, axis=0)]
    reference_spectra = storage_mtx[:, ~np.all(storage_mtx == 0, axis=0)]

    #Compounds = getPeaks(Compounds, W, reference_spectra, reference_spectra_coef)

    deselect = np.full(len(W), True)

    for i in range(len(residual_spectra)):
        arr = [index for (index, item) in enumerate(residual_spectra[i]) if item != item]
        for i in arr:
            deselect[i] = False
    return reference_spectra, residual_spectra, Compounds

def getCompounds(file):
    with open(file, 'rb') as handle:
        Compounds = pkl.load(handle)
    return Compounds