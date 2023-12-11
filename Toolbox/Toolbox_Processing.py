import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl

import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d

from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum

from sklearn.linear_model import LassoCV

import random

def create_smoother(N):
    """Create a smoother for one variable.
    Note that we assume reflection as the boundary condition.

    Args:
        N (int): Size of the varaible

    Returns:
        np.ndarray: D matrix
    """
    D = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    # Set boundary condition to repeat
    # w_0=w_1 and w_N=w_N+1
    D[0, 0] = 1
    D[N - 1, N - 1] = 1
    return D

def build_A_matrix(spectra, Ns, Nl, Nt):
    """Builds the A matrix from the spectra.

    Args:
        spectra (np.ndarray): Array of reference spectra ("templates")
        Ns (int): Number of species
        Nl (int): Number of wavelength
        Nt (int): Number of time steps

    Returns:
        A sparse Nl*Nt, Ns*Nt matrix
    """
    S = []
    for i in range(Ns):
        a = sp.lil_matrix((Nl * Nt, Nt), dtype=np.float32)
        for j in range(Nt):
            a[(j * Nl) : (j + 1) * Nl, j] = spectra[i, :]
        S.append(a)
    return sp.hstack(S)  # (Nl*Nt, Ns*Nt) matrix


def remove_background(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Remove the background from the input spectra.

    Args:
        S (np.ndarray): An array of spectra where each row represents a spectrum.
        W (np.ndarray): The corresponding wavenumber array.

    Returns:
        np.ndarray: An array of absorbance spectra with the background removed.

    The function removes the background from the input spectra by subtracting the
    baseline absorbance spectrum obtained using the asymmetric least squares (ALS)
    algorithm. The baseline spectrum is calculated from the first spectrum in 'S'.

    """
    # Create a Spectrum object from the input data
    spectrum = Spectrum.from_array(np.array(W), np.array(S[0]), 'transmittance', wunit='cm-1', unit='')

    # Calculate the baseline using the ALS algorithm (Old method)
    sb = Spectrum.get_baseline(spectrum, algorithm='als')
    w, A = sb.get('transmittance', wunit='cm-1')

    # Calculate the baseline of the first spectrum in 'S'
    w, A = spectrum.get('transmittance', wunit='cm-1')
    base_abs = -np.log(A)

    absorbance_spectra = []

    # Process each spectrum in 'S'
    for s in S:
        a = -np.log(s)
        rb = a - base_abs
        rb[np.isnan(rb)] = 0
        absorbance_spectra.append(rb)

    return np.array(absorbance_spectra)

def getReferenceMatrix(Compounds: dict, T: float, P: float, W_obs: np.ndarray, sigma: float, dataset: str) -> np.ndarray:
    """
    Generate a reference matrix based on input compounds and parameters.

    Args:
        Compounds (dict): A dictionary containing information about chemical species.
        T (float): Temperature in Kelvin.
        P (float): Pressure in bar.
        W_obs (np.ndarray): The wavenumber array for observed spectra.
        sigma (float): The broadening constant.

    Returns:
        np.ndarray: A reference matrix containing spectra of the specified compounds.

    This function generates a reference matrix by simulating and processing spectra for
    each compound defined in the 'Compounds' dictionary. It applies broadening, via a convolution with a
    Gaussian, defined by the 'sigma' parameter, and the resulting spectra are stored in the reference matrix.

    """
    output = []
    norm_constant = 1 / (np.sqrt(2 * np.pi) * sigma)
    #norm_constant = 1

    for c in Compounds:

        plt.figure()

        bank = Compounds[c]['Source']
        tmp = np.zeros_like(W_obs)

        for i in range(len(Compounds[c]['bounds'])):
            bound = Compounds[c]['bounds'][i]
            try:
                print(c, bound, bank)
                s = calc_spectrum(
                    bound[0], bound[1],  # cm-1
                    molecule=c,
                    isotope='1',
                    pressure=P,  # bar
                    Tgas=T,  # K
                    mole_fraction=10**(-6),
                    path_length=500,  # cm
                    warnings={'AccuracyError':'ignore'},
                )
            except:
                print("BAD", c)
                continue

            s.apply_slit(0.241, 'cm-1', shape="gaussian")  # Simulate an experimental slit
            w, A = s.get('absorbance', wunit='cm-1')

            iloc, jloc = np.argmin(np.abs(w.min() - W_obs)), np.argmin(np.abs(w.max() - W_obs))
            s.resample(W_obs[iloc:jloc], energy_threshold=2)

            w, A = s.get('absorbance', wunit='cm-1')

            if sigma != 0:

                # A = np.nan_to_num(A, nan=0)
                # broadened_A = convolve_with_peaks(w, A, sigma)
                # broadened_A = np.nan_to_num(broadened_A, nan=0)
                # # broadened_A = convolve(
                # #     A, norm_constant * np.exp(-(w - np.median(w))**2 / (2 * sigma**2)), mode='same')
                # #broadened_A *= np.max(A) / np.max(broadened_A)
                # tmp[iloc:jloc] = broadened_A
                # plt.plot(w, broadened_A)

                selected_centre = weighted_average_center(w,A)
                broadening_function = gaussian_broadening(w, x0=selected_centre, sigma=sigma)
                broadened_A = broaden_spectrum(w, A, broadening_function)
                tmp[iloc:jloc] = broadened_A
                plt.plot(w, broadened_A)

            else:
                tmp[iloc:jloc] = A
                plt.plot(w, A)

        output.append(tmp)

        plt.show()
        plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/spectra_plots/' + str(c) + '.png')

    ref_mat = np.array(output)
    return ref_mat

def squeeze_Residuals(y_model: np.ndarray, y: np.ndarray, Nl: int):
    """
    Squeeze and extract information from residual data.

    Args:
    y_model (np.ndarray): Modeled data.
    y (np.ndarray): Observed data.
    Nl (int): Number of wavenumbers.

    Returns:
    tuple: A tuple containing two NumPy arrays, 'y_model_wv_squeezed' and 'y_model_time_squeezed'.

    This function takes modeled and observed data, computes the difference, and then squeezes the residual data.
    The result is separated into two parts: wavenumber-squeezed and time-squeezed data.

    """

    # Compute the difference between the modeled and observed data and reshape it
    y_model_wv_squeezed = np.array(y_model - y).reshape(-1, Nl)

    # Extract every nth element from each subarray to get time-squeezed data
    y_model_time_squeezed = extract_nth_element_from_each_subarray(y_model_wv_squeezed)

    return y_model_wv_squeezed, y_model_time_squeezed

def extract_nth_element_from_each_subarray(arr):
    # Calculate the maximum length of sub-arrays in the original array
    max_length = max(len(subarray) for subarray in arr)

    # Create an empty result array filled with NaN values
    result = np.full((max_length, len(arr)), np.nan)

    for i, subarray in enumerate(arr):
        result[:len(subarray), i] = subarray

    return result


def getReferenceMatrix2(Compounds: dict, T: float, P: float, W_obs: np.ndarray, sigma: float, dataset: str) -> np.ndarray:
    """
    Generate a reference matrix based on input compounds and parameters.

    Args:
        Compounds (dict): A dictionary containing information about chemical species.
        T (float): Temperature in Kelvin.
        P (float): Pressure in bar.
        W_obs (np.ndarray): The wavenumber array for observed spectra.
        sigma (float): The broadening constant.

    Returns:
        np.ndarray: A reference matrix containing spectra of the specified compounds.

    This function generates a reference matrix by simulating and processing spectra for
    each compound defined in the 'Compounds' dictionary. It applies broadening, via a convolution with a
    Gaussian, defined by the 'sigma' parameter, and the resulting spectra are stored in the reference matrix.

    """
    output = []
    norm_constant = 1 / (np.sqrt(2 * np.pi) * sigma)

    for c in Compounds:
        plt.figure()
        bank = Compounds[c]['Source']
        tmp = np.zeros_like(W_obs)

        s = calc_spectrum(
            800, 6000,  # cm-1
            molecule=c,
            isotope='1',
            pressure=P,  # bar
            Tgas=T,  # K
            mole_fraction=10**(-6),
            path_length=500,  # cm
            warnings={'AccuracyError':'ignore'},
        )
     
        s.apply_slit(0.241, 'cm-1', shape="gaussian")  # Simulate an experimental slit
        w, A = s.get('absorbance', wunit='cm-1')

        iloc, jloc = np.argmin(np.abs(w.min() - W_obs)), np.argmin(np.abs(w.max() - W_obs))
        s.resample(W_obs[iloc:jloc], energy_threshold=2)

        w, A = s.get('absorbance', wunit='cm-1')

        if sigma != 0:
            broadened_A = convolve_with_peaks(w, A, sigma)


            # broadened_A = convolve(
            #     A, norm_constant * np.exp(-(w - np.median(w))**2 / (2 * sigma**2)), mode='same')
            #broadened_A *= np.max(A) / np.max(broadened_A)
            tmp[iloc:jloc] = broadened_A
            plt.plot(w, broadened_A)
        else:
            tmp[iloc:jloc] = A

        output.append(tmp)

        plt.plot(w, A)
        plt.show()
        plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/spectra_plots/' + str(c) + '_full.png')

    ref_mat = np.array(output)
    return ref_mat

def convolve_with_peaks(x, y, sigma):
    delta_functions = np.eye(len(x))
    gaussian_kernel = gaussian_filter1d(delta_functions, sigma, mode='constant', cval=0.0, axis=0)
    convolved_y = np.dot(gaussian_kernel, y)
    
    return convolved_y

def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) / sigma) ** 2 / 2) / (sigma * np.sqrt(2 * np.pi))


def gaussian_broadening(x, x0, sigma):
    """
    Calculate the Gaussian broadening of a peak.

    Parameters:
    - x: array of x values
    - x0: center of the peak
    - sigma: standard deviation (width) of the peak

    Returns:
    - Gaussian broadening function values corresponding to each x value
    """
    return np.exp(-0.5 * ((x - x0) / sigma)**2)

def broaden_spectrum(x, y, broadening_function):
    """
    Broaden a spectrum using a given broadening function.

    Parameters:
    - x: array of x values (original data)
    - y: array of y values (original data)
    - broadening_function: function used for broadening

    Returns:
    - Broadened spectrum (convolution of original spectrum and broadening function)
    """
    broadened_y = convolve(y, broadening_function, mode='same') / sum(broadening_function)
    return broadened_y

def weighted_average_center(x, y):
    """
    Calculate the weighted average center of a spectrum.

    Parameters:
    - x: array of x values (original data)
    - y: array of y values (original data)

    Returns:
    - Weighted average center
    """
    total_intensity = np.sum(y)
    weighted_average = np.sum(x * y) / total_intensity
    return weighted_average


def getReferenceMatrix_opt(c, T, P, wv_obs, broad_array, key) -> np.ndarray:
    """
    Generate a reference matrix based on input compounds and parameters.

    Args:
        Compounds (dict): A dictionary containing information about chemical species.
        T (float): Temperature in Kelvin.
        P (float): Pressure in bar.
        W_obs (np.ndarray): The wavenumber array for observed spectra.
        sigma (float): The broadening constant.

    Returns:
        np.ndarray: A reference matrix containing spectra of the specified compounds.

    This function generates a reference matrix by simulating and processing spectra for
    each compound defined in the 'Compounds' dictionary. It applies broadening, via a convolution with a
    Gaussian, defined by the 'sigma' parameter, and the resulting spectra are stored in the reference matrix.

    """
    output = []

    for mol in broad_array:

        bank = c['Source']
        tmp = np.zeros_like(wv_obs)

        for i in range(len(c['bounds'])):
            bound = c['bounds'][i]
            try:
                s = calc_spectrum(
                    bound[0], bound[1],  # cm-1
                    molecule=key,
                    isotope='1',
                    pressure=P,  # bar
                    Tgas=T,  # K
                    mole_fraction=mol,
                    path_length=500,  # cm
                    warnings={'AccuracyError':'ignore'},
                )
            except:
                print("BAD", c)
                continue

            s.apply_slit(0.241, 'cm-1', shape="gaussian")  # Simulate an experimental slit
            w, A = s.get('absorbance', wunit='cm-1')

            iloc, jloc = np.argmin(np.abs(w.min() - wv_obs)), np.argmin(np.abs(w.max() - wv_obs))
            s.resample(wv_obs[iloc:jloc], energy_threshold=2)

            w, A = s.get('absorbance', wunit='cm-1')

            tmp[iloc:jloc] = A

        output.append(tmp)

    mol_array = np.array(output)

    return mol_array