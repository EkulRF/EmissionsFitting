import numpy as np
import matplotlib.pyplot as plt

def PlotTimeSeries(name, compound_list, x_sol, sigma, Nt):

    num_rows = len(compound_list) // 2  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows+1, 2, figsize=(10, 6))

    fig.text(0.5, 0.04, 'Time Step', ha='center')
    fig.text(0.07, 0.5, 'Concentration / ppm', va='center', rotation='vertical')

    plt.subplots_adjust(hspace=0.5)

    for i, spc in enumerate(compound_list):
        row, col = divmod(i, 2)
        axs[row,col].plot(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt], color = 'red')
        axs[row,col].fill_between(np.arange(Nt), x_sol[i*Nt:(i+1)*Nt] - 0.5*sigma[i*Nt:(i+1)*Nt],
                                x_sol[i*Nt:(i+1)*Nt] + 0.5*sigma[i*Nt:(i+1)*Nt],
                                color= "0.8")
        axs[row, col].set_title(spc, loc='right')
        axs[row, col].grid()

    fig.subplots_adjust(top=0.95)

    plt.savefig('EmFit_private/plot/' + name + '.jpg')

    return

def PlotResiduals(y_model_wv_squeezed, y_model_time_squeezed):

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

def PlotER_TimeSeries(name, compound_list, x_sol, sigma, Nt, Norm_Species):

    conc = x_sol[compound_list.index(Norm_Species)*Nt:(compound_list.index(Norm_Species)+1)*Nt]
    se = x_sol[compound_list.index(Norm_Species)*Nt:(compound_list.index(Norm_Species)+1)*Nt]

    num_rows = (len(compound_list)-1) // 2  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows+1, 2, figsize=(10, 6))

    fig.text(0.5, 0.04, 'Time Step', ha='center')
    fig.text(0.07, 0.5, 'Emission Ratios / (X/' + Norm_Species + ')', va='center', rotation='vertical')

    for i, spc in enumerate(compound_list):
        if spc == 'CO2':
            continue

        ER = x_sol[i*Nt:(i+1)*Nt] / conc
        ER_se = [ER[a]*np.sqrt((se[a]/conc[a])**2 + (sigma[i*Nt:(i+1)*Nt][a]/x_sol[i*Nt:(i+1)*Nt][a])**2) for a in range (len(ER))]

        row, col = divmod(i, 2)
        axs[row,col].plot(np.arange(Nt), ER, color = 'red')
        # axs[row,col].fill_between(np.arange(Nt), np.array(ER) - 0.5*np.array(ER_se),
        #                         np.array(ER) + 0.5*np.array(ER_se),
        #                         color= "0.8")
        axs[row, col].set_title(spc, loc='right')
        axs[row, col].grid()

    plt.savefig('EmFit_private/plot/' + name + '.jpg')

    return