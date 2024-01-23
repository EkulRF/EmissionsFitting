import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from Toolbox.Toolbox_Inversion import *

def PlotTimeSeries(name: str, compound_list: list, x_sol: np.ndarray, x_err: np.ndarray, Nt: int, dataset: str):
    """
    Plot time series data for a list of compounds.

    Args:
        name (str): Name of the plot or figure.
        compound_list (list): List of compound names.
        x_sol (np.ndarray): Solution for species concentrations over time.
        x_err (np.ndarray): Standard error for the derived concentrations over time.
        Nt (int): Number of time steps.

    This function generates a time series plot for the present compounds, where each compound's
    concentration is plotted over time. The resulting plot is saved with the specified 'name'.
    
    """
    print('Plotting Time Series')


    # Getting time data
    directory = "/home/luke/data/MATRIX_data/" + dataset + '/'

    for filename in os.listdir(directory):
        if filename.endswith('ResultSeries.txt'):
            df = pd.read_csv(directory + filename, delim_whitespace=True, skiprows=[0])
    
    df.columns = [col.replace(".", "_") for col in df.columns]

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    ##

    num_rows = len(compound_list) // 2  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows + 1, 2, figsize=(10, 6))

    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.07, 0.5, 'Concentration / ppm', va='center', rotation='vertical')

    plt.subplots_adjust(hspace=0.5)

    for i, spc in enumerate(compound_list):
        row, col = divmod(i, 2)
        try:
            axs[row, col].plot(df['DateTime'], x_sol[i * Nt:(i + 1) * Nt], color='red')
            axs[row, col].fill_between(df['DateTime'], x_sol[i * Nt:(i + 1) * Nt] - 0.5 * x_err[i * Nt:(i + 1) * Nt],
                                    x_sol[i * Nt:(i + 1) * Nt] + 0.5 * x_err[i * Nt:(i + 1) * Nt],
                                    color="0.8")
        except:
            axs[row, col].plot(np.arange(len(x_sol[i * Nt:(i + 1) * Nt])), x_sol[i * Nt:(i + 1) * Nt], color='red')
            axs[row, col].fill_between(np.arange(len(x_sol[i * Nt:(i + 1) * Nt])), x_sol[i * Nt:(i + 1) * Nt] - 0.5 * x_err[i * Nt:(i + 1) * Nt],
                                    x_sol[i * Nt:(i + 1) * Nt] + 0.5 * x_err[i * Nt:(i + 1) * Nt],
                                    color="0.8")
        axs[row, col].set_title(spc, loc='center', pad = -10)
        axs[row, col].grid()

        if row != num_rows:
            axs[row, col].set_xticklabels([])
        else:
            axs[row, col].tick_params(axis='x', rotation=45)
            axs[row, col].xaxis.set_major_formatter(DateFormatter('%H:%M'))

        #axs[row, col].grid(axis='y')

    for i in range(len(compound_list), (num_rows + 1) * 2):
        fig.delaxes(axs.flatten()[i])

    fig.subplots_adjust(top=0.95)

    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/' + name + '.png')

    return

def PlotOPUS_Results(name: str, dataset: str):

    print('Plotting Results from OPUS')

    directory = "/home/luke/data/MATRIX_data/" + dataset + '/'

    for filename in os.listdir(directory):
        if filename.endswith('ResultSeries.txt'):
            df = pd.read_csv(directory + filename, delim_whitespace=True, skiprows=[0])
    
    df.columns = [col.replace(".", "_") for col in df.columns]

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.drop(['Date', 'Time'], axis=1)

    num_rows = (len(df.columns)-2) // 2  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows + 1, 2, figsize=(10, 6))

    fig.text(0.5, 0.04, 'Time Step', ha='center')
    fig.text(0.07, 0.5, 'Concentration / ppm', va='center', rotation='vertical')

    plt.subplots_adjust(hspace=0.5)

    for i, spc in enumerate(df.columns):

        if spc == 'DateTime':
            continue

        row, col = divmod(i, 2)
        axs[row, col].plot(df['DateTime'], df[spc], color='red')
        axs[row, col].set_title(spc, loc='right')
        axs[row, col].grid()

    fig.subplots_adjust(top=0.95)

    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/' + name + '.png')


    return

def PlotResiduals(y_model_wv_squeezed: np.ndarray, y_model_time_squeezed: np.ndarray, dataset: str):
    """
    Plot residuals both across wavenumbers and in time.

    Args:
        y_model_wv_squeezed (np.ndarray): Residuals across wavenumbers.
        y_model_time_squeezed (np.ndarray): Residuals across time steps.

    This function generates two plots: one for residuals across wavenumbers and one for residuals
    in time. The resulting plots are saved in the '/plot' directory.

    """
    print('Plotting Residuals')

    W = np.load('/home/luke/data/Model/results/'+ dataset + '/W.npy')
    std = np.std(y_model_time_squeezed.flatten())


    ## Plotting Residuals across wavenumber range
    plt.figure()

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])

    ax1 = fig.add_subplot(gs[0])
    for i in y_model_wv_squeezed:
        ax1.scatter(W, i, s=0.001)
    ax1.set_ylabel('Residual (Predicted - Real)')
    ax1.set_xlabel('Wavenumber / cm-1')
    ax1.set_ylim(-6*std,6*std)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_ylim(-6*std,6*std)
    ax2.hist(y_model_wv_squeezed.flatten(), bins=200, orientation='horizontal', color='skyblue', edgecolor='black')
    ax2.set_xlabel('Frequency')
    ax2.set_title('Histogram')

    plt.tight_layout(w_pad=2)

    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/resid_wv.png')


    ## Plotting Residuals across time steps
    plt.figure()

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])

    # Create the residual plot on the left
    ax1 = fig.add_subplot(gs[0])
    for i in y_model_time_squeezed:
        ax1.scatter(np.arange(len(i)), i, s=0.001)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Residual (Predicted - Real)')
    ax1.set_ylim(-6*std,6*std)

    # Create the histogram on the right, rotated 90 degrees
    ax2 = fig.add_subplot(gs[1])
    ax2.set_ylim(-6*std,6*std)
    ax2.hist(y_model_time_squeezed.flatten(), bins=200, orientation='horizontal', color='skyblue', edgecolor='black')
    ax2.set_xlabel('Frequency')
    ax2.set_title('Histogram')

    # Adjust the space between the two plots
    plt.tight_layout(w_pad=2)
    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/resid_time.png')
    # Show the combined plot
    plt.show()

    return

def PlotER_TimeSeries(name: str, compound_list: list, x_sol: np.ndarray, x_err: np.ndarray, Nt: int, Norm_Species: str, dataset: str):
    """
    Plot time series data for a list of compounds.

    Args:
        name (str): Name of the plot or figure.
        compound_list (list): List of compound names.
        x_sol (np.ndarray): Solution for species concentrations over time.
        x_err (np.ndarray): Standard error for the derived concentrations over time.
        Nt (int): Number of time steps.

    This function generates a time series plot for the present compounds, where each compound's
    concentration is plotted over time. The resulting plot is saved with the specified 'name'.
    
    """
    print('Plotting ER Time Series')


    # Getting time data
    directory = "/home/luke/data/MATRIX_data/" + dataset + '/'

    for filename in os.listdir(directory):
        if filename.endswith('ResultSeries.txt'):
            df = pd.read_csv(directory + filename, delim_whitespace=True, skiprows=[0])
    
    df.columns = [col.replace(".", "_") for col in df.columns]

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    ##

    num_rows = (len(compound_list)-1) // 2  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows + 1, 2, figsize=(10, 6))

    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.07, 0.5, 'Emission Ratio / conc(x)/conc('+Norm_Species+')', va='center', rotation='vertical')

    plt.subplots_adjust(hspace=0.5)

    conc_Norm = x_sol[compound_list.index(Norm_Species)*Nt:(compound_list.index(Norm_Species)+1)*Nt]
    se_Norm = x_err[compound_list.index(Norm_Species)*Nt:(compound_list.index(Norm_Species)+1)*Nt]

    for i, spc in enumerate(compound_list):

        if spc == Norm_Species:
            fig.delaxes(axs.flatten()[i])
            continue
        # elif spc == 'CO2':
        #     x_sol[i * Nt:(i + 1) * Nt] *= 20
        # elif spc =='CO':
        #     x_sol[i * Nt:(i + 1) * Nt] *= 4

        er = np.divide(x_sol[i * Nt:(i + 1) * Nt],conc_Norm)

        # Compute the standard error of the result
        se_result = np.sqrt(np.divide(x_err[i * Nt:(i + 1) * Nt], x_sol[i * Nt:(i + 1) * Nt])**2 + (np.divide(se_Norm, conc_Norm))**2)


        row, col = divmod(i, 2)
        try:
            axs[row, col].plot(df['DateTime'], er, color='red')
        except:
            axs[row, col].plot(np.arange(len(er)), er, color='red')
        # axs[row, col].fill_between(df['DateTime'], er - 0.5 * se_result,
        #                           er + 0.5 * se_result,
        #                           color="0.8")
        axs[row, col].set_title(spc, loc='center', pad = -10)
        axs[row, col].grid()

        if row != num_rows:
            axs[row, col].set_xticklabels([])
            #axs[row, col].grid(axis='x')
        else:
            axs[row, col].tick_params(axis='x', rotation=20)
            axs[row, col].xaxis.set_major_formatter(DateFormatter('%H:%M'))

        #axs[row, col].grid(axis='y')

    for i in range(len(compound_list), (num_rows + 1) * 2):
        fig.delaxes(axs.flatten()[i])

    fig.subplots_adjust(top=0.95)

    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/' + name + '.png')

    return

def PlotSpectralResiduals(full_ref_spec, full_obs_spec, W_full, x_sol, sigma, Compounds, dataset: str, t = None):

    y_model, y, y_model_err = inversion_residual(full_ref_spec, full_obs_spec, x_sol, np.sqrt(sigma))

    Nl = full_ref_spec.shape[1]
    Nt = full_obs_spec.shape[0]

    for j, key in enumerate(Compounds):

        species_arr = x_sol[j * Nt:(j + 1) * Nt]
        t = np.argmax(species_arr)
        y_model_sel, y_sel, y_model_err_sel = y_model[t*Nl:(t+1)*Nl], y[t*Nl:(t+1)*Nl], y_model_err[t*Nl:(t+1)*Nl]

        num = len(Compounds[key]['bounds'])
        fig, axs = plt.subplots(num, 2, figsize=(12, 4*num), gridspec_kw={'width_ratios': [3, 1]})
        if num == 1:
            axs = [axs]

        fig.suptitle(key, x=0.05, y=0.95, ha='left', fontsize=16)  # Add figure text at the top left
        fig.text(0.02, 0.5, 'Absorbance', va='center', ha='left', rotation='vertical', fontsize=14)
        fig.text(0.695, 0.5, 'Prediction Residuals', va='center', ha='left', rotation='vertical', fontsize=14)

        for i, bound in enumerate(Compounds[key]['bounds']):
            
            max_points = []
            min_points = []

            for b in Compounds[key]['bounds']:
                max_points.append(np.max(y_model_sel[np.where((W_full >= b[0]) & (W_full <= b[1]))]))
                min_points.append(np.min(y_model_sel[np.where((W_full >= b[0]) & (W_full <= b[1]))]))

            # Plot on the left side
            axs[i][0].plot(W_full, y_sel, color='red', linewidth=4, label='Observed Spectrum')
            axs[i][0].plot(W_full, y_model_sel, '-.', color='k', label='Modelled Spectrum')
            axs[i][0].fill_between(W_full, y_model_sel - y_model_err_sel, y_model_sel + y_model_err_sel, color='gray', alpha=0.35)
            axs[i][0].set_xlim(bound[0], bound[1])
            axs[i][0].set_ylim(np.min(min_points)-0.01, np.max(max_points)+0.01)
            axs[i][0].tick_params(axis='both', labelsize=12)

            # Add custom title on the right side
            axs[i][0].text(0.75, 1.05, str(bound[0])+' - '+str(bound[1])+' cm$^{-1}$', transform=axs[i][0].transAxes, va='center', ha='left', fontsize=14, rotation='horizontal', fontstyle='italic')

            # Plot histogram on the right side
            diff = y_sel[np.where((W_full >= bound[0]) & (W_full <= bound[1]))] - y_model_sel[np.where((W_full >= bound[0]) & (W_full <= bound[1]))]
            axs[i][1].hist(diff, bins=20,  alpha=0.7, color='#9034db', edgecolor='black', orientation='horizontal')
            axs[i][1].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axs[i][1].grid(axis='x', linestyle='--', alpha=0.7)
            axs[i][1].tick_params(axis='both', labelsize=12)

        axs[len(Compounds[key]['bounds'])-1][0].set_xlabel('Wavenumber / cm$^{-1}$', fontsize=14)
        axs[len(Compounds[key]['bounds'])-1][1].set_xlabel('Frequency', fontsize=14)
        axs[0][0].legend(fontsize=14)

        # Adjust layout to prevent clipping of titles
        plt.tight_layout(rect=[0.03, 0, 0.98, 0.95])
        plt.subplots_adjust(wspace=0.25)

        # Show the plot
        plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/Residuals/' + key + '.png')
        plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/Residuals/' + key + '.pdf')
        plt.show()

    return

def max_sum_interval(arr, x):
    n = len(arr)

    # Check if x is greater than the array length
    if x > n: 
        raise ValueError("Interval size x is greater than array length.")

    # Initialize the sum of the first interval
    current_sum = sum(arr[:x])
    max_sum = current_sum
    max_index = 1

    # Iterate through the array to find the interval with the maximum sum
    for i in range(int(n/x)):
        current_sum = sum(arr[x*i: (x*i)+ x+1])
        if current_sum > max_sum:
            max_sum = current_sum
            max_index = i + 1

    return max_index