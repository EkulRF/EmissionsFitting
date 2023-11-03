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

def isNaN(num):
    return num != num

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

# def remove_background(
#     spectra, bounds, wv, n_steps= 150
# ) -> np.ndarray:
#     # Mean spectrum first n_steps timesteps
#     avg_spectrum = spectra[:150, :].mean(axis=0)

#     # Remove the "background"
#     # and subset the measured spectrum
#     residual_spectra = (spectra - avg_spectrum[None, :])#[:, passer]
#     # Flip spectrum.
#     residual_spectra = -residual_spectra - residual_spectra.min(axis=0)
#     return residual_spectra

def remove_background(S, W):
    
    spectrum = Spectrum.from_array(np.array(W), np.array(S[0]), 'transmittance',
                           wunit='cm-1', unit='')
    
    sb = Spectrum.get_baseline(spectrum, algorithm='als')
    w, A = sb.get('transmittance', wunit='cm-1')
    ####
    w, A = spectrum.get('transmittance', wunit='cm-1')
    ####
    
    base_abs = -np.log(A)
    
    absorbance_spectra = []
    
    for s in S:
        
        a = -np.log(s)
        rb = a - base_abs
        rb[np.isnan(rb)] = 0
        absorbance_spectra.append(rb)
    
    return np.array(absorbance_spectra)

def remove_background_new(S, W):
    
    #This is a cheat solution to get a baseline

    spectrum = Spectrum.from_array(np.array(W), np.array(S[0]), 'transmittance',
                           wunit='cm-1', unit='')
    
    sb = Spectrum.get_baseline(spectrum, algorithm='als')
    w, A = sb.get('transmittance', wunit='cm-1')
    ####
    w, A = spectrum.get('transmittance', wunit='cm-1')
    ####
    
    base_abs = -np.log(A)

    degree = 5
    coefficients = np.polyfit(w[:20000], base_abs[:20000], degree)
    y_pred = np.polyval(coefficients, w[:20000])
    y_pred += -((y_pred.min()-base_abs[:20000].min())/2)
    y_pred = list(y_pred)
    coefficients2 = np.polyfit(w[20000:], base_abs[20000:], degree)
    y_pred2 = np.polyval(coefficients, w[20000:])
    y_pred2 += -((y_pred2.min()-base_abs[20000:].min())/2)
    y_pred2 = list(y_pred2)
    for i in y_pred2:
        y_pred.append(i)

    base_abs = y_pred
    
    absorbance_spectra = []
    
    for s in S:
        
        a = -np.log(s)
        rb = a - base_abs
        rb[np.isnan(rb)] = 0
        absorbance_spectra.append(rb)
    
    return np.array(absorbance_spectra)

def getReferenceMatrix(Compounds, T, P, W_obs, sigma):
    
    output = []
    output_coef = []

    norm_constant = 1 / (np.sqrt(2 * np.pi) * sigma)
    
    for c in Compounds:

        plt.figure()
        
        bank = Compounds[c]['Source']
        
        tmp = np.zeros_like(W_obs)
        tmp_coef = np.zeros_like(W_obs)
        
        for i in range(len(Compounds[c]['bounds'])):
            bound = Compounds[c]['bounds'][i]
            try:
                print(c, bound, bank)
                s = calc_spectrum(bound[0], bound[1],         # cm-1
                          molecule=c,
                          isotope='1',
                          pressure=P,   # bar
                          Tgas=T,           # K
                          mole_fraction=10**(-6),
                          path_length=500      # cm
                          )
            except:
                print("BAD", c)
                continue
            s.apply_slit(0.241, 'cm-1', shape="gaussian")       # simulate an experimental slit
            w, A = s.get('absorbance', wunit='cm-1')
            
           
            iloc, jloc = np.argmin(np.abs(w.min() - W_obs)), np.argmin(np.abs(w.max() - W_obs))
            s.resample(W_obs[iloc:jloc], energy_threshold=2)
            
            w, A = s.get('absorbance', wunit='cm-1')

            if sigma != 0:
                broadened_A = convolve(A, norm_constant * np.exp(-(w-np.median(w))**2 / (2 * sigma**2)), mode='same')
                broadened_A *= np.max(A)/np.max(broadened_A)
                tmp[iloc:jloc] = broadened_A
                plt.plot(w,broadened_A)
            else:
                tmp[iloc:jloc] = A

            w_coef, A_coef = s.get('abscoeff', wunit='cm-1', Iunit='cm-1')

            tmp_coef[iloc:jloc] = A_coef
            
        output.append(tmp)
        output_coef.append(tmp_coef)

        plt.show()
        plt.savefig(str(c)+'.jpg')

    ref_mat = np.array(output)
    ref_mat_coef = np.array(output_coef)
    
    return ref_mat, ref_mat_coef

def convert2PPM(Compounds, x_sol, sigma, Nt):

    compound_list = list(Compounds.keys())

    for i, spc in enumerate(compound_list):
        f_d = Compounds[spc]['PPM_RelativeAbsorbance']
        f_d[0].insert(0,0)
        f_d[1].insert(0,0)
        f =  scipy.interpolate.interp1d(f_d[1], f_d[0])
        for j in range(len(x_sol[i*Nt:(i+1)*Nt])):
            sigma[i*Nt:(i+1)*Nt][j] *= (f(abs(x_sol[i*Nt:(i+1)*Nt][j]))/abs(x_sol[i*Nt:(i+1)*Nt][j]))
            x_sol[i*Nt:(i+1)*Nt][j] = f(abs(x_sol[i*Nt:(i+1)*Nt][j]))

    return x_sol, sigma

def convert2PPM_new(Compounds, x_sol, sigma, Nt, l):

    compound_list = list(Compounds.keys())

    for i, spc in enumerate(compound_list):

        peak_wv, extinction_coef, peak = Compounds[spc]['Peak_Info'][0], Compounds[spc]['Peak_Info'][1], Compounds[spc]['Peak_Info'][2]
        molar_m = Compounds[spc]['Molar_Mass']
            
        for j in range(len(x_sol[i*Nt:(i+1)*Nt])):
            A = x_sol[i*Nt:(i+1)*Nt][j] * peak
            c = A / (extinction_coef * l)

            ppm = abs(c * molar_m * 10**6)
            #print(spc, ppm, peak, sigma[i*Nt:(i+1)*Nt][j])
            sigma[i*Nt:(i+1)*Nt][j] *= (ppm/abs(x_sol[i*Nt:(i+1)*Nt][j])) if ppm != 0 else 0
            x_sol[i*Nt:(i+1)*Nt][j] = ppm

    return x_sol, sigma

def convert2PPM_new_new(Compounds, x_sol, sigma, Nt, l):

    for i, spc in enumerate(list(Compounds.keys())):

        peak = Compounds[spc]['Peak_Info'][2]
        abs_coef = Compounds[spc]['Peak_Info'][3]
        molar_m = Compounds[spc]['Molar_Mass']
            
        for j in range(len(x_sol[i*Nt:(i+1)*Nt])):
            A = x_sol[i*Nt:(i+1)*Nt][j] * peak
            c = A / (abs_coef * l)

            ppm = abs(c * molar_m)

            sigma[i*Nt:(i+1)*Nt][j] *= (ppm/abs(x_sol[i*Nt:(i+1)*Nt][j])) if ppm != 0 else 0
            x_sol[i*Nt:(i+1)*Nt][j] = ppm

    return x_sol, sigma

def getPeaks(Compounds, W, reference_spectra, reference_spectra_coef):

    for i, spc in enumerate(list(Compounds.keys())):
        
        peak_wv = Compounds[spc]['Peak_Info'][0]

        ind = find_closest_index(peak_wv, W)

        peak = reference_spectra[i][find_largest_peak_in_range(reference_spectra[i], ind)]
        peak_coef = reference_spectra_coef[i][find_largest_peak_in_range(reference_spectra_coef[i], ind)]

        Compounds[spc]['Peak_Info'].append(peak)
        Compounds[spc]['Peak_Info'].append(peak_coef)

    return Compounds

def find_closest_index(target, array):
    # Initialize variables to keep track of the closest value and its index
    closest_value = array[0]
    closest_index = 0

    # Calculate the initial difference between the target and the first element of the array
    min_difference = abs(target - array[0])

    # Iterate through the array to find the closest value
    for i in range(1, len(array)):
        difference = abs(target - array[i])

        # Update the closest value and its index if a closer value is found
        if difference < min_difference:
            min_difference = difference
            closest_value = array[i]
            closest_index = i

    return closest_index

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

def find_largest_peak_in_range(arr, index):
    n = len(arr)
    left_limit = max(0, index - 200)
    right_limit = min(n, index + 201)

    max_peak = index

    for i in range(left_limit, right_limit):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if max_peak is None or arr[i] > arr[max_peak]:
                max_peak = i

    return max_peak

def squeeze_Residuals(y_model, y, Nl):

    y_model_wv_squeezed = np.array(y_model-y).reshape(-1, Nl)

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