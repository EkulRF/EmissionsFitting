import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy

from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum

from Toolbox_Processing import *
from Toolbox_Reading import *
from Toolbox_Inversion import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    obs_spectra = parser.parse_args()

    Compounds = getCompounds('compoundsCH3OH.pickle')

    ref_spec, obs_spec = generateData(Compounds, obs_spectra.path)

    #Lasso Inversion
    ref_spec, Compounds, A = lasso_inversion(ref_spec, obs_spec)
    # Should have option to say whether we want lasso regression stats???

    #Tikhonov Regularisation
    x_sol, sigma, C = temporally_regularised_inversion(ref_spec, obs_spec, 0.00001)

    # PPM
    x_sol, sigma = convert2PPM(Compounds, x_sol, sigma)

    ### Save results to csv.

if __name__ == "__main__":
    main()