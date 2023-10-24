import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pathlib import Path
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy
from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum


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


def remove_background(
    spectra, bounds, wv, n_steps= 150
) -> np.ndarray:
    # Mean spectrum first n_steps timesteps
    avg_spectrum = spectra[:150, :].mean(axis=0)

    # Now, select only the spectral regions that we defined
    # above
    # First, a filter function
    #def pass_func(a, b):
     #   return np.logical_and(wv >= wv[a], wv <= wv[b])

    # Apply the filter function to the bounds
    #pass_arrays = np.array([pass_func(*v) for k, v in bounds.items()])
    # Combine the individual regions for each spectra
    #passer = np.logical_or.reduce(pass_arrays, axis=0)

    # Remove the "background"
    # and subset the measured spectrum
    residual_spectra = (spectra - avg_spectrum[None, :])#[:, passer]
    # Flip spectrum.
    residual_spectra = -residual_spectra - residual_spectra.min(axis=0)
    return residual_spectra


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
    cobj = spl.spilu(C)
    x_sol = cobj.solve(A_mat.T @ y) if do_spilu else spl.spsolve(C, A_mat.T @ y)

    s = np.zeros(Ns * Nt)
    t = np.zeros(Ns * Nt)
    for i in range(Ns * Nt):
        a = t * 1.0
        a[i] = 1.0
        s[i] = cobj.solve(a)[i]
    return (x_sol, s, C) if post_cov else (x_sol, s)

def temporally_regularised_inversion2(
    reference_spectra,
    residual_spectra,
    gamma_sel,
    post_cov= True,
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
    G = sp.eye(Ns)
    G.setdiag(gamma_sel)
    D_mat = sp.lil_matrix(sp.kron(G, create_smoother(Nt)))
    # Squeeze observations
    y = residual_spectra.flatten()
    # Solve
    C = sp.csc_array(A_mat.T @ A_mat + D_mat)
    cobj = spl.spilu(C)
    x_sol = cobj.solve(A_mat.T @ y) if do_spilu else spl.spsolve(C, A_mat.T @ y)

    s = np.zeros(Ns * Nt)
    t = np.zeros(Ns * Nt)
    for i in range(Ns * Nt):
        a = t * 1.0
        a[i] = 1.0
        s[i] = cobj.solve(a)[i]
    return (x_sol, s, C) if post_cov else (x_sol, s)

def isNaN(num):
    return num != num

def generateData(Compounds):

    bounds = {
    "CO2": (6250, 6600),
    "CH4": [9100, 9700],
    "H2O": [5500, 5900],
    "CO": [2500, 4000],
}
    selected_spectra = list(bounds.keys())

    (spectra_obs, absorption_spectra, species_names, wv_obs) = read_data(
        selected_spectra,
        "EmFit_private/spectra/test_series", # Location of the series we want to invert.
        "EmFit_private/spectra/templates/" # Location of the individual species "template" absorption
    )

    T, P = 300, 1.01

    storage_mtx = getReferenceMatrix(Compounds, T, P, wv_obs)

    for i in range(len(storage_mtx)):
        for j in range(len(storage_mtx[i])):
            if isNaN(storage_mtx[i][j]):
                storage_mtx[i][j] = 0

    S = np.array([s[~np.all(storage_mtx == 0, axis=0)] for s in spectra_obs])

    S = np.array([-np.log(np.abs(x)) for x in S])
    W = wv_obs[~np.all(storage_mtx == 0, axis=0)]
    residual_spectra = remove_background(S, {}, W)

    storage_mtx = storage_mtx[:,~np.all(storage_mtx == 0, axis=0)]
    reference_spectra = storage_mtx[:, ~np.all(storage_mtx == 0, axis=0)]

    deselect = np.full(len(W), True)

    for i in range(len(residual_spectra)):
        arr = [index for (index, item) in enumerate(residual_spectra[i]) if item != item]
        for i in arr:
            deselect[i] = False
    return reference_spectra, residual_spectra

def getReferenceMatrix(Compounds, T, P, W_obs):
    
    output = []
    
    for c in Compounds:
        
        bank = Compounds[c]['Source']
        
        tmp = np.zeros_like(W_obs)
        
        for i in range(len(Compounds[c]['bounds'])):
            bound = Compounds[c]['bounds'][i]
            try:
                print(c, bound, bank)
                s = calc_spectrum(bound[0], bound[1],         # cm-1
                          molecule=c,
                          isotope='1',
                          pressure=P,   # bar
                          Tgas=T,           # K
                          mole_fraction=10**(-4),
                          path_length=500,      # cm
                          databank=bank,  # or 'hitemp', 'geisa', 'exomol'
                          )
            except:
                print("BAD", c)
                continue
            s.apply_slit(0.241, 'cm-1', shape="gaussian")       # simulate an experimental slit
            w, A = s.get('absorbance', wunit='cm-1')
           
            iloc, jloc = np.argmin(np.abs(w.min() - W_obs)), np.argmin(np.abs(w.max() - W_obs))
            s.resample(W_obs[iloc:jloc], energy_threshold=2)
            
            w, A = s.get('absorbance', wunit='cm-1')

            tmp[iloc:jloc] = A
            
        output.append(tmp)
    
    ref_mat = np.array(output)
    
    return ref_mat


def Regularised_Inversion(
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

    print("Ns,Nl,Nt", Ns, Nl, Nt)

    assert Ns != residual_spectra.shape[1], "Wrong spectral sampling!!"
    # Create the "hat" matrix
    A_mat = build_A_matrix(reference_spectra, Ns, Nl, Nt)

    print("Amat built")
    # Regulariser
    D_mat = sp.lil_matrix(sp.kron(sp.eye(Ns), create_smoother(Nt)))
    # Squeeze observations
    y = residual_spectra.flatten()
    
    # Solve
    C = sp.csc_array(A_mat.T @ A_mat + lambda_ * D_mat)
    cobj = spl.spilu(C)
    #x_sol = cobj.solve(A_mat.T @ y) if do_spilu else spl.spsolve(C, A_mat.T @ y)
    print("Here now")
    A_csr = sp.csr_matrix(A_mat.all())
    A_dense = np.array(A_csr.todense())
    print("densified!")
    model = LassoCV(cv=5, fit_intercept=False) 
    reg = model.fit(A_dense, y)
    print("done!")

    s = np.zeros(Ns * Nt)
    t = np.zeros(Ns * Nt)
    for i in range(Ns * Nt):
        a = t * 1.0
        a[i] = 1.0
        s[i] = cobj.solve(a)[i]
    return (x_sol, s, C) if post_cov else (x_sol, s)