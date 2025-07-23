import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec
from matplotlib import rc, rcParams
from scipy.optimize import curve_fit


#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"

phase_diag = "D_alpha_observables"
#phase_diag = "lambda_alpha_observables"
#phase_diag = "Jz_alpha_observables"

probe_idx = 1
chi = 500

alpha = float("inf")
#alpha = 4.0

D_c = 0.96845

# output filename
output_file = f"../plots/SvN_mins_alpha{alpha}.pdf"

# directory and filename
#data_dir = f'../data/fss/ising_transition/central_charge/'
data_dir = f'../output/svn_min/'
# Use glob to find all csv files that match the pattern
file_pattern_sz0 = os.path.join(f"{data_dir}/alpha{alpha}/Sz0/", f'spinone_heisenberg_obs_chi{chi}_alpha{alpha}_*.csv')
file_pattern_sz1 = os.path.join(f"{data_dir}/alpha{alpha}/Sz1/", f'spinone_heisenberg_obs_chi{chi}_alpha{alpha}_*.csv')

# font sizes
fs1 = 16
fs2 = 12


def read_data(file_pattern, alpha):

    svn_list = []

    csv_files = glob.glob(file_pattern)

    print(file_pattern)

    # Lists to hold the data
    D_values = []
    L_values = []
    z_values = []

    # Iterate through all the files
    for file in csv_files:

        # Read the CSV file
        data = pd.read_csv(file)

        L = float(file.split('_L')[-1].split(f'.csv')[0])  # Extract alpha from filename

        # Append data to lists
        combined = list(zip(data["D"].values, data["SvN"].values))

        sorted_combined = sorted(combined)
        d, z = zip(*sorted_combined)

        D_values.append(d)
        z_values.append(z)
        L_values.append(L)

        # sort by alpha values
        #if phase_diag == "lambda_alpha_observables":
        #    D_values = np.reciprocal(D_values)

    # sort colorplot vals
    combined = zip(L_values, D_values, z_values)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    L_values, D_values, z_values = zip(*sorted_combined)

    # Convert lists to arrays for plotting
    D_values = np.array(D_values)
    L_values = np.array(L_values)
    z_values = np.array(z_values)

    return L_values, D_values, z_values


if __name__ == "__main__":
    L_values_sz0, D_values_sz0, svn_values_sz0 = read_data(file_pattern_sz0, alpha)
    L_values_sz1, D_values_sz1, svn_values_sz1 = read_data(file_pattern_sz1, alpha)
    print(f"Ls: {np.shape(L_values_sz0)}, Ds: {np.shape(D_values_sz0)}, svns: {np.shape(svn_values_sz0)}")
    print(f"Ls: {np.shape(L_values_sz1)}, Ds: {np.shape(D_values_sz1)}, svns: {np.shape(svn_values_sz1)}")

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    ax1, ax2 = axs


    for idx, (Ds, svn) in enumerate(zip(D_values_sz0, svn_values_sz0)):
        ax1.plot(Ds, svn, marker='o', color='#ef476f', alpha=(idx+1)/len(L_values_sz0))

    cmap = plt.cm.winter
    n_lines = len(L_values_sz1)
    colors_sz1 = cmap(np.linspace(0, 1, n_lines))

    for idx, (Ds, svn) in enumerate(zip(D_values_sz1, svn_values_sz1)):
        ax1.plot(Ds, svn, marker='o', color='#118ab2', alpha=(idx+1)/len(L_values_sz1))

    # Create a color map
    #cmap = plt.cm.viridis
    #n_lines = len(L_values_sz0)
    #colors = cmap(np.linspace(0, 1, n_lines))
    svn_values = np.minimum(np.array(svn_values_sz0), np.array(svn_values_sz1))
    for idx, (Ds, svn) in enumerate(zip(D_values_sz0, svn_values)):
        ax2.plot(Ds, svn, color='#7209b7', marker='o', alpha=(idx+1)/len(L_values_sz1))

    ax1.set_xlabel(r'$D$', fontsize=fs1)
    ax2.set_xlabel(r'$D$', fontsize=fs1)

    #ax2.set_xlabel('$\\log(L)$', fontsize=fs1)
    ax1.set_ylabel('$S_{\\rm VN}$', fontsize=fs1)
    ax2.set_ylabel('$\\min(S^{m^z=0}_{\\rm VN},S^{m^z=1}_{\\rm VN})$', fontsize=fs1)
    #ax2.set_ylim([-0.1, 3.5])
    #ax2.set_ylim([-0.1, 5.5])
    ax2.legend(loc='best', fontsize=fs2)

    #specific_tick = -0.315
    specific_tick = D_c
    specific_label = "$D_c$"

    current_ticks = ax1.get_xticks()
    current_labels = [tick.get_text() for tick in ax1.get_xticklabels()]

    # Add the new tick and label
    new_ticks = list(current_ticks) + [specific_tick]
    new_labels = current_labels + [specific_label]

    # Set the updated ticks and labels
    ax1.set_xticks(new_ticks)
    ax1.set_xticklabels(new_labels)
    ax1.axvline(specific_tick, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlim(0.2,1.2)

    ax2.set_xticks(new_ticks)
    ax2.set_xticklabels(new_labels)
    ax2.axvline(specific_tick, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlim(0.2,1.2)


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax1.text(0.1, 0.9, f"$\\alpha={alpha}$", transform=ax1.transAxes, fontsize=12, verticalalignment='center', bbox=props)

    # title
    #ax1.set_title(f"$L_1={L1}$,~$L_2={L2}$", fontsize=fs2)

    # save fig
    plt.tight_layout()
    plt.savefig(output_file)

    # Show plot
    plt.show()
