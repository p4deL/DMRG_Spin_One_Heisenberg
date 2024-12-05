import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec
from matplotlib import rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"

phase_diag = "D_alpha_observables"
#phase_diag = "lambda_alpha_observables"

L1 = 100  # system size
L2 = 110
Ls = (L1, L2)

chi = 300

# for lambda plot
#alpha_selected = [0.36, 0.42, 0.44, 0.78]
# for D plot
alpha_selected = [0.1, 0.26, 0.6, 0.8]

# output filename
output_file = f"../plots/colorplot_{phase_diag}_centralcharge_L1{L1}_L2{L2}.pdf"

# directory and filename
data_dir = f'../data/phase_diagram/{phase_diag}/B-1e0/'
# Use glob to find all csv files that match the pattern
file_pattern1 = os.path.join(f"{data_dir}L{L1}/", f'spinone_heisenberg_obs_chi{chi}_alpha*.csv')
file_pattern2 = os.path.join(f"{data_dir}L{L2}/", f'spinone_heisenberg_obs_chi{chi}_alpha*.csv')
file_patterns = (file_pattern1, file_pattern2)

# font sizes
fs1 = 18
fs2 = 14


def read_data(file_patterns, Ls):

    svn_list = []
    for L, file_pattern in zip(Ls, file_patterns):
        print(L, file_pattern)

        csv_files = glob.glob(file_pattern)

        # Lists to hold the data
        D_values = []
        alpha_values = []
        z_values = []

        # Iterate through all the files
        for file in csv_files:
            # Extract alphaval from the filename
            alpha = float(file.split('_alpha')[-1].split(f'_L{L}.csv')[0])  # Extract alpha from filename
            # print(alpha)



            # Read the CSV file
            data = pd.read_csv(file)

            # Append data to lists
            # values = np.full_like(data["D"].values, alpha)
            # sort by D values
            if phase_diag == "lambda_alpha_observables":
                combined = list(zip(np.reciprocal(data["D"].values), data["SvN"].values))
            else:
                combined = list(zip(data["D"].values, data["SvN"].values))

            sorted_combined = sorted(combined)
            d, z = zip(*sorted_combined)

            D_values.append(d)
            z_values.append(z)

            alpha_values.append(np.full_like(data["D"].values, 1. / alpha))  # Create an array of alphaval for each D

        # sort by alpha values
        #if phase_diag == "lambda_alpha_observables":
        #    D_values = np.reciprocal(D_values)


        # sort colorplot vals
        combined = zip(alpha_values, D_values, z_values)
        sorted_combined = sorted(combined, key=lambda x: x[0][0])
        alpha_values, D_values, z_values = zip(*sorted_combined)

        # Convert lists to arrays for plotting
        D_values = np.array(D_values)
        alpha_values = np.array(alpha_values)
        z_values = np.array(z_values)
        svn_list.append(z_values)

    ceff_values = 6 * (svn_list[0] - svn_list[1])/(np.log(Ls[0])-np.log(Ls[1]))
    # print(D_values)
    # print(alpha_values)
    # print(z_values)

    return D_values, alpha_values, ceff_values


def get_selected_values(alpha_values, D_values, ceff_values, alpha_selected):
    """
    Finds the indices of selected alpha values and returns corresponding D and ceff values.

    Parameters:
    - alpha_values (numpy array): Array of alpha values.
    - D_values (numpy array): Array of D values.
    - ceff_values (numpy array): Array of ceff values.
    - alpha_select (list): List of alpha values to select.

    Returns:
    - selected_D (numpy array): D values corresponding to alpha_select.
    - selected_ceff (numpy array): ceff values corresponding to alpha_select.
    """
    # Convert alpha_select to a numpy array for comparison
    # alpha_selected = np.reciprocal(alpha_selected)
    alpha_selected = np.array(alpha_selected)

    # Find indices where alpha_values match alpha_select
    #indices = np.where(np.isin(np.unique(alpha_values), alpha_selected))[0]
    indices = np.concatenate([np.where(np.isclose(np.unique(alpha_values), alpha, atol=1e-8))[0] for alpha in alpha_selected])

    alphas = [ np.unique(alpha_values)[i] for i in indices ]
    print(alpha_selected)
    print(f"indices: {indices}: {alphas}")

    # Get the corresponding D and ceff values
    #print(ceff_values)
    D_selected = D_values[indices]
    ceff_selected = ceff_values[indices]

    #D_selected = []
    #ceff_selected = []
    #for alpha in alpha_selected:

        # Find indices where alpha_values match alpha_select
     #   indices = np.where(np.isin(alpha_values, alpha))[0]

        # Get the corresponding D and ceff values
    #    D = D_values[indices]
    #    ceff = ceff_values[indices]



    return D_selected, ceff_selected


if __name__ == "__main__":
    D_values, alpha_values, ceff_values = read_data(file_patterns, Ls)
    # Now create a grid of D and alphaval values for contouring
    D_grid, alpha_grid = np.meshgrid(np.unique(D_values), np.unique(alpha_values))
    # Create the contour plot
    #plt.figure(figsize=(8, 6))
    #fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    #ax1, ax2 = axs
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[50, 1], height_ratios=[1, 1])  # Reserve space for the colorbar
    ax1 = fig.add_subplot(gs[0, 0])  # Main plot
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Second plot

    colorplot = ax1.pcolor(D_grid, alpha_grid, ceff_values.reshape(D_grid.shape), cmap='rainbow') #cmap='viridis')
    # plt.scatter([-1.5, 0, 0, 0.5, 1.5], [0, 0, 0.5, 0.5, 0.], color='red', marker='o', s=100)

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[0, 1])  # Main plot
    cbar = fig.colorbar(colorplot, cax=ax_cbar)
    cbar.set_label("$c_{\\rm eff}$", fontsize=fs2)


    D_selected, ceff_selected = get_selected_values(alpha_values, D_values, ceff_values, alpha_selected)
    for Ds, ceff, invalpha in zip(D_selected, ceff_selected, alpha_selected):
        #print(Ds)
        ax2.plot(Ds, ceff, marker='o', linestyle='-', label=f"$\\alpha={1./invalpha:.3f}$")

    # Label axes
    if phase_diag == "D_alpha_observables":
        ax2.set_xlabel(r'$D$', fontsize=fs1)
    else:
        ax2.set_xlabel('$\\lambda$', fontsize=fs1)
    ax1.set_ylabel('$1/\\alpha$', fontsize=fs1)
    ax2.set_ylabel('$c_{\\rm eff}$', fontsize=fs1)
    ax2.set_ylim([-0.1, 3.5])
    #ax2.set_ylim([-0.1, 5.5])
    ax2.legend(loc='upper right', fontsize=fs2)

    # title
    ax1.set_title(f"$L_1={L1}$,~$L_2={L2}$", fontsize=fs2)

    # save fig
    plt.savefig(output_file)

    # Show plot
    plt.show()
