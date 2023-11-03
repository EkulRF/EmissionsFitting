import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl

import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.signal import convolve

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

def getReferenceMatrix(Compounds: dict, T: float, P: float, W_obs: np.ndarray, sigma: float) -> np.ndarray:
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
                    path_length=500  # cm
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
                broadened_A = convolve(
                    A, norm_constant * np.exp(-(w - np.median(w))**2 / (2 * sigma**2)), mode='same')
                broadened_A *= np.max(A) / np.max(broadened_A)
                tmp[iloc:jloc] = broadened_A
                plt.plot(w, broadened_A)
            else:
                tmp[iloc:jloc] = A

        output.append(tmp)

        plt.show()
        plt.savefig('EmFit_private/Spectra_Plots/' + str(c) + '.jpg')

    ref_mat = np.array(output)
    return ref_mat

# def convert2PPM(Compounds, x_sol, sigma, Nt):

#     compound_list = list(Compounds.keys())

#     for i, spc in enumerate(compound_list):
#         f_d = Compounds[spc]['PPM_RelativeAbsorbance']
#         f_d[0].insert(0,0)
#         f_d[1].insert(0,0)
#         f =  scipy.interpolate.interp1d(f_d[1], f_d[0])
#         for j in range(len(x_sol[i*Nt:(i+1)*Nt])):
#             sigma[i*Nt:(i+1)*Nt][j] *= (f(abs(x_sol[i*Nt:(i+1)*Nt][j]))/abs(x_sol[i*Nt:(i+1)*Nt][j]))
#             x_sol[i*Nt:(i+1)*Nt][j] = f(abs(x_sol[i*Nt:(i+1)*Nt][j]))

#     return x_sol, sigma

# def convert2PPM_new(Compounds, x_sol, sigma, Nt, l):

#     compound_list = list(Compounds.keys())

#     for i, spc in enumerate(compound_list):

#         peak_wv, extinction_coef, peak = Compounds[spc]['Peak_Info'][0], Compounds[spc]['Peak_Info'][1], Compounds[spc]['Peak_Info'][2]
#         molar_m = Compounds[spc]['Molar_Mass']
            
#         for j in range(len(x_sol[i*Nt:(i+1)*Nt])):
#             A = x_sol[i*Nt:(i+1)*Nt][j] * peak
#             c = A / (extinction_coef * l)

#             ppm = abs(c * molar_m * 10**6)
#             #print(spc, ppm, peak, sigma[i*Nt:(i+1)*Nt][j])
#             sigma[i*Nt:(i+1)*Nt][j] *= (ppm/abs(x_sol[i*Nt:(i+1)*Nt][j])) if ppm != 0 else 0
#             x_sol[i*Nt:(i+1)*Nt][j] = ppm

#     return x_sol, sigma

# def getPeaks(Compounds, W, reference_spectra, reference_spectra_coef):

#     for i, spc in enumerate(list(Compounds.keys())):
        
#         peak_wv = Compounds[spc]['Peak_Info'][0]

#         ind = find_closest_index(peak_wv, W)

#         peak = reference_spectra[i][find_largest_peak_in_range(reference_spectra[i], ind)]
#         peak_coef = reference_spectra_coef[i][find_largest_peak_in_range(reference_spectra_coef[i], ind)]

#         Compounds[spc]['Peak_Info'].append(peak)
#         Compounds[spc]['Peak_Info'].append(peak_coef)

#     return Compounds

# def find_closest_index(target, array):
#     # Initialize variables to keep track of the closest value and its index
#     closest_value = array[0]
#     closest_index = 0

#     # Calculate the initial difference between the target and the first element of the array
#     min_difference = abs(target - array[0])

#     # Iterate through the array to find the closest value
#     for i in range(1, len(array)):
#         difference = abs(target - array[i])

#         # Update the closest value and its index if a closer value is found
#         if difference < min_difference:
#             min_difference = difference
#             closest_value = array[i]
#             closest_index = i

#     return closest_index

# def find_closest_peak(arr, index):
#     n = len(arr)

#     # Initialize variables to store the closest peaks in both directions
#     left_peak = None
#     right_peak = None

#     # Search for the closest peak to the left of the given index
#     for i in range(index - 1, -1, -1):
#         if arr[i] > arr[i + 1]:
#             left_peak = i
#             break

#     # Search for the closest peak to the right of the given index
#     for i in range(index + 1, n):
#         if arr[i] > arr[i - 1]:
#             right_peak = i
#             break

#     # Calculate the distances to the left and right peaks
#     left_distance = float('inf') if left_peak is None else index - left_peak
#     right_distance = float('inf') if right_peak is None else right_peak - index

#     # Determine the closest peak
#     if left_distance < right_distance:
#         return left_peak
#     elif right_distance < left_distance:
#         return right_peak
#     else:
#         return left_peak

# def find_largest_peak_in_range(arr, index):
#     n = len(arr)
#     left_limit = max(0, index - 200)
#     right_limit = min(n, index + 201)

#     max_peak = index

#     for i in range(left_limit, right_limit):
#         if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
#             if max_peak is None or arr[i] > arr[max_peak]:
#                 max_peak = i

#     return max_peak

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