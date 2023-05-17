from os import listdir
from os.path import isfile, join
import csv
import pandas as pd
from datetime import datetime as dt
import numpy as np
from numpy.polynomial import polynomial as P
from scipy import interpolate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

###########################

# Reads raw emission spectra and produces a wavenumber array (Freq) and a transmission array (Spec)

def getEmissionsSpectra(filename):

    with open(filename, "r", encoding="utf8") as raw_data:
        file = csv.reader(raw_data, delimiter="\t")
        
        Freq = []
        Spec = []
        
        for j, row in enumerate(file):
            Freq.append(float(row[0]))
            Spec.append(float(row[1]))

    return Freq, Spec

###########################

# Fits polynomial BASELINE to raw emissions spectra

def getSpecFit(Freq, Spec, Range, polynomial):

    c, stats = P.polyfit(Freq,Spec,polynomial,full=True)

    return c, stats

# Get a range of data points from data from an input wavenumber range

def getRange(Freq, Spec, Range):
    
    index = [Freq.index(min(Freq, key=lambda x:abs(x-Range[0]))), Freq.index(min(Freq, key=lambda x:abs(x-Range[1])))]

    x = Freq[index[0]:index[1]]
    y = Spec[index[0]:index[1]]
    
    return x, y

# Polynomial fit

def polynomialSum(x, order, exponents):
    
    y = 0
    
    for i in range(order+1):
        y += exponents[i] * x**i
    
    return y

# The core function for deriving the base line

def getBase(Freq, Spec, Range, polynomial):
    
    x, y = getRange(Freq, Spec, Range)
    
    c, stats = getSpecFit(x, y, Range, polynomial)
    
    Base = []
    Diff = []
    
    for j in range(len(x)):
        
        Base.append(polynomialSum(x[j], polynomial, c))
        Diff.append(y[j] - polynomialSum(x[j], polynomial, c))

    Base = [b + max(Diff) for b in Base]

    return x, Base

##################################################

Compounds = ['C2H2',
          'C2H4',
          'C2H6',
          'CH3Cl',
          'CH4',
          'ClONO2',
          'ClO',
          'CO2',
          'COF2',
          'CO',
          'H2CO',
          'H2O2',
          'H2O',
          'HBr',
          'HCl',
          'HCN',
          'HCOOH',
          'HF',
          'HI',
          'HNO3',
          'HO2',
          'HOCl',
          'N2O',
          'N2',
          'NH3',
          'NO+',
          'NO2',
          'NO',
          'O2',
          'O3',
          'OCS',
          'OH',
          'PH3',
          'SF6',
          'SO2']

References_Ranges = [[1120,1230],
              [1190,1240],
              [725,770],
              [900,980],
              [2090,2130],
              [2229,2247],
              [680,710],
              [951,1175],
              [900,1020],
              [2740,2840],
              [1070,1130], 
              [730,760], 
              [1120,885], 
              [1800,1930], 
              [2745,2850], 
              [700,770], 
              [880,915], 
              [3000,3150], 
              [951,1099], 
              [1185,1230], 
              [1140,1270], 
              [1874,1930], 
              [2906,2935], 
              [787,820], 
              [2175,2235], 
              [1043,1070], 
              [1130,1215], 
              [860,970], 
              [1095,1216], 
              [710,760]]

def getRefRange(Comp):

    index = Compounds.index(Comp)
    Range = References_Ranges[index]

    return Range

# Get absorption spectra from file

def getAbsorption(filename):

    with open(filename, "r", encoding="utf8") as raw_data:
        file = csv.reader(raw_data, delimiter="\t")
        
        Freq = []
        Spec = []

        for j, row in enumerate(file):
            Freq.append([float(x) for x in row][0])
            Spec.append([float(x) for x in row][1])

    return Freq, Spec

# Having run the above function for all reference templates, the below function pulls it out of the saved .csv file for the compounds you want.

def getAbsorptionFeatures(Compounds):

    ## Compounds: 'Compounds': array containing string names of the compounds of interest, e.g. ['CO2', 'CO', 'Acetic Acid', 'H2O']. Refer to the above 'References' array.

    Absorp = []

    for c in Compounds:

        ind = Compounds.index(c)

        with open('Absorptions.csv', 'r') as f:
            f = csv.reader(f, delimiter=',')
            for i,line in enumerate(f):
                if i==ind:
                    Absorp.append([float(x) for x in line])
                    
    return Absorp

# Coming from the above function, this function snips the absorption spectra to a given wavelength range.

def getAbsorptionSelection(Absorption, Range):

    Wave = []

    with open('Wave.csv', 'r') as f:
        f = csv.reader(f, delimiter=',')
        for line in f:
            Wave.append(float(line[0]))

    new_wave = []
    new_absorp = []

    for y in Absorption:
        w, a = getRange(Wave, y, Range)

        new_wave.append(w)
        new_absorp.append(a)
        
    return new_wave, new_absorp


#######################################################

# Remove parts of A matrix for species that do not absorb within a given range.

def reduce_A_matrix(a):

    species_present = np.all(a, axis=0)

    empty_compound_index = []

    for i in range(len(a[0])):
        if species_present[i] == False:
            empty_compound_index.append(i)
        
    new_a = []

    for y in range(len(a)-1):
    
        new_a.append([i for j,i in enumerate(a[y]) if j not in empty_compound_index])
    
    Compounds_Present = [i for j,i in enumerate(Compounds) if j not in empty_compound_index]

    return new_a, Compounds_Present

# Remove parts of A matrix, leaving only parts concerned with CO2, CO, CH4, H2O

def reduce_A_matrix_bigSpeciesOnly(a, Compounds_Present):
    
    new_a = []

    keep_indexes = []

    Compounds_Core = []

    for i, j in enumerate(Compounds_Present):
         if j in ['CO2', 'H2O', 'CO', 'CH4']:
            keep_indexes.append(i)
            Compounds_Core.append(j)

    for i in range(len(a)):
        arr = []
        for j in keep_indexes:
            arr.append(a[i][j])
        
        new_a.append(arr)

    return new_a, Compounds_Core

# Identify peaks in reference and create func from them to improve fit (hopefully)

def smooth_signal(a, wave_seg):

    functions = []
    normal_func = []

    for i in range(len(a[0])):
        v = []
    
        for j in range(len(a)):
        
            v.append(a[j][i])
        
        v = np.array(v)
        wave_seg = np.array(wave_seg)
    
        peaks, _ = find_peaks(v, height=0.15 * max(v))
        peaks = np.insert(peaks,0,0)
    
        peak_func = interpolate.interp1d(wave_seg[peaks], v[peaks])
        norm_func = interpolate.interp1d(wave_seg, v)
    
        functions.append(peak_func)
        normal_func.append(norm_func)

    new_a = []    

    for x in range(len(a)):
        arr = []
        for i, func in enumerate(functions):
            try:
                arr.append(float(func(wave_seg[x])))
            except:
                arr.append(float(normal_func[i](wave_seg[x])))
                continue
    
        new_a.append(arr)

    return new_a

# The main meat and potatoes.

def fit_spectra(filename, Range, Base, References, reduce_fit):

    Path = 'spectra/'
    Path_templates = Path + '/templates/'

    # unpack data

    Wave_b = Base[0]
    Spec_b = Base[1]

    f, s = getEmissionsSpectra(filename)
    f, s = np.array(f[800:]), np.array(s[800:])


    # process data to play with

    Absorption = list(Spec_b - s)

    filename_CO2 = Path_templates + '006_CO2_04_448K_ref296s.prn'

    wave_ref, abso_ref = getAbsorption(filename_CO2)

    func = interpolate.interp1d(f, Absorption)

    wave_seg, abso_seg = getRange(wave_ref, func(wave_ref), Range)

    y_obs = (-abso_seg)-min(-abso_seg)
    
    # fun maths fitting time!

    a = np.transpose(References)

    index = [wave_ref.index(min(wave_ref, key=lambda x:abs(x-Range[0]))), wave_ref.index(min(wave_ref, key=lambda x:abs(x-Range[1])))]
    a = a[index[0]:(index[1]+1)]

    # remove species that don't absorb in the range
    a, Compounds_Present = reduce_A_matrix(a)

    # Smooth over absorption peaks. Improves fits in a big way.
    a  = smooth_signal(a, wave_seg)


    # new func to remove all compounds except for CO2, H2O, CO, CH4
    if reduce_fit == True:
        a, Compounds_Present = reduce_A_matrix_bigSpeciesOnly(a, Compounds_Present)

    aT = np.transpose(a)
    aT_a = aT @ a
    inv = np.linalg.inv(aT_a)

    x = np.linalg.solve(inv, aT @ y_obs)

    e = a @ x - y_obs
    rmse = np.sqrt(np.mean(e * e))

    # Plots a simple bar chart of the derived weights

    plt.figure()
    plt.bar(Compounds_Present, x)
    plt.xticks(rotation=90)
    plt.yscale('log')
    plt.show()

    # Error plot (I doubt this is correct)

    plt.plot(e, "o", color='purple')
    plt.axhspan(-1,1,fc="0.8", alpha=0.5)
    plt.show()

    C_obs_inv = np.diag(np.ones_like(y_obs))

    x_opt = np.linalg.solve(aT @ C_obs_inv @ a, aT @ C_obs_inv @ y_obs)

    e_new = a @ x_opt - y_obs

    cov = np.linalg.inv(aT @ C_obs_inv @ a)
    rmse_opt = np.sqrt(np.mean(e_new * e_new))
    uncertainty = np.sqrt(cov.diagonal())

    # Prints uncertainties

    print("Inferred weights:", x_opt)
    print("Uncertainity:",uncertainty)
    print("RMSE:",rmse_opt)

    cond = np.linalg.cond(aT @ C_obs_inv @ a)
    
    print("Conditioning Number:", cond)

    empt = np.zeros(len(aT[0]))

    for i in range(len(aT)):
        empt += aT[i] * x_opt[i]

    # Plots a 'Observed' vs 'Fit' diagram

    plt.plot(wave_seg, empt, label='Fit')
    plt.plot(wave_seg, y_obs, label='Observed')
    plt.legend()
    plt.show()

    return x_opt, e_new, rmse_opt, uncertainty, cond