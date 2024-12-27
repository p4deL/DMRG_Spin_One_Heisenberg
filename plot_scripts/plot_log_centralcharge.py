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

D_selected = [-0.33, -0.315, -0.3, -0.28]
chi = 300

alpha = float("inf")

# output filename
output_file = f"../plots/log_centralcharge_{phase_diag}.pdf"

# directory and filename
data_dir = f'../data/fss/ising_transition/central_charge/'
# Use glob to find all csv files that match the pattern
file_pattern = os.path.join(f"{data_dir}", f'spinone_heisenberg_obs_chi{chi}_alpha{alpha}_*.csv')

# font sizes
fs1 = 16
fs2 = 12


def read_data(file_pattern, alpha):

    svn_list = []

    csv_files = glob.glob(file_pattern)

    #print(file_pattern)

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
        # values = np.full_like(data["D"].values, alpha)
        # sort by D values
        if phase_diag == "lambda_alpha_observables":
            combined = list(zip(np.reciprocal(data["D"].values), data["SvN"].values))
        elif phase_diag == "Gamma_alpha_observables":
            combined = list(zip(data["Gamma"].values, data["SvN"].values))
        elif phase_diag == "Jz_alpha_observables":
            combined = list(zip(data["Jz"].values, data["SvN"].values))
        else:
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


def get_selected_values(D_values, L_values, svn_values, D_selected):
    """
    Finds the indices of selected alpha values and returns corresponding D and ceff values.
    """
    # Convert alpha_select to a numpy array for comparison
    # alpha_selected = np.reciprocal(alpha_selected)
    D_selected = np.array(D_selected)

    # Find indices where alpha_values match alpha_select
    #indices = np.where(np.isin(np.unique(alpha_values), alpha_selected))[0]
    #print(D_values)
    indices = np.concatenate([np.where(np.isclose(D_values[i], D, atol=1e-8))[0] for i, D in enumerate(D_selected)])

    #Ds = [ D_values[i] for i in indices ]
    #print(D_selected)
    print(f"indices: {indices}")

    # Get the corresponding D and ceff values
    D_selected = D_values[:,indices]
    svn_selected = svn_values[:,indices]


    return D_selected, svn_selected


def linear_fit(log_x, a, b):
    return a*log_x + b


def perform_log_fit(x, y):

    log_x = np.log(x)


    # TODO: check parameters
    popt, pcov = curve_fit(linear_fit, log_x, y, p0=[0.5, 1])

    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pcov))
    print(a, "+/-", a_err)
    print(b, "+/-", b_err)

    return a, a_err, b, b_err

if __name__ == "__main__":
    L_values, D_values, svn_values = read_data(file_pattern, alpha)
    print(f"Ls: {np.shape(L_values)}, Ds: {np.shape(D_values)}, svns: {np.shape(svn_values)}")
    D_selected, svn_selected = get_selected_values(D_values, L_values, svn_values, D_selected)
    print(f"Dsel: {np.shape(D_selected)} svnsel: {np.shape(svn_selected)}")

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    ax1, ax2 = axs

    for _, Ds, svn in zip(L_values, D_values, svn_values):
        ax1.plot(Ds, svn)

    print(D_selected)
    for D, svn in zip(D_selected.T, svn_selected.T):
        #print(f"L: {L}, svn: {svn}")
        ax2.scatter(L_values, svn, label=f"$D={D[0]:.3f}$")

    idx_min = 2
    Lfit = L_values[idx_min:]
    svnfit = svn_selected.T[1,idx_min:]
    print(svnfit)
    a, a_err, b, berr = perform_log_fit(Lfit, svnfit)
    ax2.text(0.6, 0.55, "$c_{\\rm eff}=$" + f"${6*a:.3f}\\pm{6*a_err:.3f}$", transform=ax2.transAxes, fontsize=fs2)

    xrange = np.linspace(Lfit[0], Lfit[-1], 100)
    #ax2.plot(xrange, linear_fit(np.log(xrange), 0.2, -0.25))
    ax2.plot(xrange, linear_fit(np.log(xrange), a, b), lw=2.5, c="black")

    # Label axes
    if phase_diag == "lambda_alpha_observables":
        ax1.set_xlabel('$\\lambda$', fontsize=fs1)
    if phase_diag == "Gamma_alpha_observables":
        ax1.set_xlabel('$\\Gamma$', fontsize=fs1)
    if phase_diag == "Jz_alpha_observables":
        ax1.set_xlabel('$J_z$', fontsize=fs1)
    else:
        ax1.set_xlabel(r'$D$', fontsize=fs1)

    ax1.set_xlim([-0.5, -0.2])
    ax2.set_xlim([40, 1000])
    ax2.set_xscale("log")

    ax2.set_xlabel('$\\log(L)$', fontsize=fs1)
    ax1.set_ylabel('$S_{\\rm VN}$', fontsize=fs1)
    ax2.set_ylabel('$S_{\\rm VN}$', fontsize=fs1)
    #ax2.set_ylim([-0.1, 3.5])
    #ax2.set_ylim([-0.1, 5.5])
    ax2.legend(loc='best', fontsize=fs2)

    specific_tick = -0.315
    specific_label = "$D_c$"

    current_ticks = ax1.get_xticks()
    current_labels = [tick.get_text() for tick in ax1.get_xticklabels()]

    # Add the new tick and label
    new_ticks = list(current_ticks) + [specific_tick]
    new_labels = current_labels + [specific_label]

    # Set the updated ticks and labels
    ax1.set_xticks(new_ticks)
    ax1.set_xticklabels(new_labels)
    ax1.axvline(specific_tick, color='red', linestyle='--', alpha=0.5)


    # title
    #ax1.set_title(f"$L_1={L1}$,~$L_2={L2}$", fontsize=fs2)

    # save fig
    plt.tight_layout()
    plt.savefig(output_file)

    # Show plot
    plt.show()
