from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

def read_spectrum(fname, skipper=None):
    return np.loadtxt(fname, usecols=[1])[skipper:] # Skip 3 first wvl's as they look weird

def interpolate_spectrum(wv, y, wv_out):
    return np.interp(wv_out, wv, y)

def spectral_angle_mapper(spec1, spec2):
    dot_product = (spec1 * spec2).sum()
    preds_norm = np.linalg.norm(spec1)
    target_norm  = np.linalg.norm(spec1)
    z = np.clip(dot_product / (preds_norm * target_norm), -1, 1)
    return np.arccos(z)

loc = Path("spectra/test_series")
files = sorted([f for f in loc.glob("*prn")])
wv = np.loadtxt(files[0], usecols=[0])[3:]
spectra = np.array([read_spectrum(f, skipper=3) for f in tqdm(files)])
loc = Path("spectra/templates")
files = sorted([f for f in loc.glob("*prn")])
wvc = np.loadtxt(files[0], usecols=[0])
species_names = [f.name.split("_")[1] for f in files]
absorption_spectra= np.array([interpolate_spectrum(wvc, read_spectrum(f), wv) for f in tqdm(files)])


mu = spectra.mean(axis=0)
std = spectra.std(axis=0)
norm_spectra = (spectra - mu)/std
u, s, v = np.linalg.svd(norm_spectra)
sam = np.zeros_like
for i, compound in enumerate(species_names):

