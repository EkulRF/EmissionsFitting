import numpy as np
import matplotlib.pyplot as plt

def PlotTimeSeries(name: str, compound_list: list, x_sol: np.ndarray, x_err: np.ndarray, Nt: int):
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

    num_rows = len(compound_list) // 2  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows + 1, 2, figsize=(10, 6))

    fig.text(0.5, 0.04, 'Time Step', ha='center')
    fig.text(0.07, 0.5, 'Concentration / ppm', va='center', rotation='vertical')

    plt.subplots_adjust(hspace=0.5)

    for i, spc in enumerate(compound_list):
        row, col = divmod(i, 2)
        axs[row, col].plot(np.arange(Nt), x_sol[i * Nt:(i + 1) * Nt], color='red')
        axs[row, col].fill_between(np.arange(Nt), x_sol[i * Nt:(i + 1) * Nt] - 0.5 * x_err[i * Nt:(i + 1) * Nt],
                                  x_sol[i * Nt:(i + 1) * Nt] + 0.5 * x_err[i * Nt:(i + 1) * Nt],
                                  color="0.8")
        axs[row, col].set_title(spc, loc='right')
        axs[row, col].grid()

    fig.subplots_adjust(top=0.95)

    plt.savefig('EmFit_private/plot/' + name + '.jpg')

    return

def PlotResiduals(y_model_wv_squeezed: np.ndarray, y_model_time_squeezed: np.ndarray):
    """
    Plot residuals both across wavenumbers and in time.

    Args:
        y_model_wv_squeezed (np.ndarray): Residuals across wavenumbers.
        y_model_time_squeezed (np.ndarray): Residuals across time steps.

    This function generates two plots: one for residuals across wavenumbers and one for residuals
    in time. The resulting plots are saved in the '/plot' directory.

    """
    print('Plotting Residuals')

    W = np.load('EmFit_private/results/W.npy')

    plt.figure()
    for i in y_model_wv_squeezed:
        plt.scatter(W, i, s=0.001)
    plt.ylabel('Residual (Predicted - Real)')
    plt.xlabel('Wavenumber / cm-1')
    plt.savefig('EmFit_private/plot/resid_wv.jpg')

    plt.figure()

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # Create the residual plot on the left
    ax1 = fig.add_subplot(gs[0])
    for i in y_model_time_squeezed:
        ax1.scatter(np.arange(len(i)), i, s=0.001)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Residual (Predicted - Real)')

    # Create the histogram on the right, rotated 90 degrees
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(y_model_time_squeezed.flatten(), bins=100, orientation='horizontal', color='skyblue', edgecolor='black')
    ax2.set_xlabel('Frequency')
    ax2.set_title('Histogram')

    # Adjust the space between the two plots
    plt.tight_layout(w_pad=2)
    plt.savefig('EmFit_private/plot/resid_time.jpg')
    # Show the combined plot
    plt.show()

    return

def PlotER_TimeSeries(name: str, compound_list: list, x_sol: np.ndarray, x_err: np.ndarray, Nt: int, Norm_Species: str):
    """
    Plot emission ratio time series for the present compounds.

    Args:
        name (str): Name of the plot or figure.
        compound_list (list): List of compound names.
        x_sol (np.ndarray): Solution data for the compounds over time.
        x_err (np.ndarray): Error data for the compounds over time.
        Nt (int): Number of time steps.
        Norm_Species (str): The species from which emissions are normalised.

    """
    print('Plotting ER Time Series')

    conc = x_sol[compound_list.index(Norm_Species)*Nt:(compound_list.index(Norm_Species)+1)*Nt]
    se = x_err[compound_list.index(Norm_Species)*Nt:(compound_list.index(Norm_Species)+1)*Nt]

    num_rows = (len(compound_list)-1) // 2  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows+1, 2, figsize=(10, 6))

    fig.text(0.5, 0.04, 'Time Step', ha='center')
    fig.text(0.07, 0.5, 'Emission Ratios / (X/' + Norm_Species + ')', va='center', rotation='vertical')

    for i, spc in enumerate(compound_list):
        if spc == 'CO2':
            continue

        ER = x_sol[i*Nt:(i+1)*Nt] / conc
        ER_se = [ER[a]*np.sqrt((se[a]/conc[a])**2 + (x_err[i*Nt:(i+1)*Nt][a]/x_sol[i*Nt:(i+1)*Nt][a])**2) for a in range (len(ER))]

        row, col = divmod(i, 2)
        axs[row,col].plot(np.arange(Nt), ER, color = 'red')
        #Should fix uncertainty calculations (too big atm- mistake in calc.)
        # axs[row,col].fill_between(np.arange(Nt), np.array(ER) - 0.5*np.array(ER_se),
        #                          np.array(ER) + 0.5*np.array(ER_se),
        #                          color= "0.8")
        # axs[row, col].set_title(spc, loc='right')
        axs[row, col].grid()

    plt.savefig('EmFit_private/plot/' + name + '.jpg')

    return