import numpy as np
import math
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

x_sol = np.load('EmFit_private/results/sol.npy')
sigma = np.load('EmFit_private/results/sig.npy')
Compounds = np.load('EmFit_private/results/comp.npy', allow_pickle=True).item()
ref_spec = np.load('EmFit_private/results/ref.npy')
obs_spec = np.load('EmFit_private/results/obs.npy')
W = np.load('W.npy')

y_model_wv_squeezed = np.load('EmFit_private/results/y_model_wv_squeezed.npy')
y_model_time_squeezed = np.load('EmFit_private/results/y_model_time_squeezed.npy')

def PlotResiduals2(y_model_wv_squeezed, y_model_time_squeezed):

    W = np.load('W.npy')

    plt.figure()
    for i in y_model_wv_squeezed:
        plt.scatter(W, i, s=0.001)
    plt.ylabel('Residual (Predicted - Real)')
    plt.xlabel('Wavenumber / cm-1')
    plt.ylim(-2.5,2.5)
    plt.savefig('resid_wv.jpg')

    plt.figure()
    # for i in y_model_time_squeezed:
    #     plt.scatter(np.arange(len(i)),i, s=0.001)
    # plt.ylabel('Residual (Predicted - Real)')
    # plt.xlabel('Time Step')
    # plt.savefig('resid_time.jpg')

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # Create the residual plot on the left
    ax1 = fig.add_subplot(gs[0])
    for i in y_model_time_squeezed:
        ax1.scatter(np.arange(len(i)), i, s=0.001)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Residual (Predicted - Real)')
    ax1.set_ylim(-2.5,2.5)

    # Create the histogram on the right, rotated 90 degrees
    ax2 = fig.add_subplot(gs[1])
    ax1.set_ylim(-2.5,2.5)
    ax2.hist(y_model_time_squeezed.flatten(), bins=100, orientation='horizontal', color='skyblue', edgecolor='black')
    ax2.set_xlabel('Frequency')
    ax2.set_title('Histogram')

    # Adjust the space between the two plots
    plt.tight_layout(w_pad=2)
    plt.savefig('resid_time.jpg')
    # Show the combined plot
    plt.show()

    return


PlotResiduals2(y_model_wv_squeezed, y_model_time_squeezed)
