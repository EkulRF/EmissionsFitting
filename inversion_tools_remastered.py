import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pathlib import Path
from tqdm import tqdm


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






#########################################

# Voigt Profile

