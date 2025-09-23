import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec
from matplotlib import rc, rcParams
from scipy.optimize import curve_fit
import matplotlib.cm as cm

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"

phase_diag = "D_alpha_observables"
#phase_diag = "lambda_alpha_observables"
#phase_diag = "Jz_alpha_observables"

chi = 500


# output filename
output_file = f"../plots/paper/centralcharge_gaussian.pdf"

# directory and filename
#data_dir = f'../data/fss/ising_transition/central_charge/'
data_dir = f'../data/fss/gaussian_transition/alpha*/central_charge/Sz1/'
# Use glob to find all csv files that match the pattern
file_pattern = os.path.join(f"{data_dir}", f'spinone_heisenberg_obs_chi{chi}_D*_alpha*.csv')

# font sizes
fs1 = 16
fs2 = 8


def read_data(file_pattern):


    csv_files = glob.glob(file_pattern)
    #print(file_pattern)
    #print(csv_files)

    # Lists to hold the data
    alpha_values = []
    L_values = []
    svn_values = []

    # Iterate through all the files
    for file in csv_files:
        print(file)

        # Read the CSV file
        data = pd.read_csv(file)

        alpha = float(file.split('_alpha')[-1].split(f'.csv')[0])  # Extract alpha from filename

        # Append data to lists
        # sort by L values
        combined = list(zip(data["L"].values, data["SvN"].values))

        sorted_combined = sorted(combined)
        L, svn = zip(*sorted_combined)

        alpha_values.append(alpha)
        L_values.append(L)
        svn_values.append(svn)

    # sort colorplot vals
    combined = zip(alpha_values, L_values, svn_values)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    alpha_values, L_values, svn_values = zip(*sorted_combined)

    # Convert lists to arrays for plotting
    alpha_values = np.array(alpha_values)
    L_values = np.array(L_values)
    svn_values = np.array(svn_values)


    return alpha_values, L_values, svn_values


def linear_fit(log_x, a, b):
    return a*log_x + b


def perform_log_fit(x, y):

    log_x = np.log(x)


    # TODO: check parameters
    popt, pcov = curve_fit(linear_fit, log_x, y, p0=[0.5, 1])

    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pcov))
    #print("a =", a, "+/-", a_err)
    #print("b =", b, "+/-", b_err)

    return a, a_err, b, b_err
def hex_to_RGB(hex_str):
  """ #FFFFFF -> [255,255,255]"""
  #Pass 16 to the integer function for change of base
  return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


if __name__ == "__main__":
    alpha_values, L_values, svn_values = read_data(file_pattern)
    print(f"alphas: {np.shape(alpha_values)}, Ls: {np.shape(L_values)}, svns: {np.shape(svn_values)}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    #colors = cm.viridis(np.linspace(0, 1, len(alpha_values)))
    #colors = cm.cividis(np.linspace(0, 1, len(alpha_values)))
    colors = get_color_gradient("#457b9d", "#81b29a", len(alpha_values))

    ceffs = []
    ceffs_err = []
    # iterate over all alphas
    for i, alpha in enumerate(alpha_values):
        color = colors[i]

        # plot entanglement entropy as function of log(L)
        Ls = L_values[i]
        svn = svn_values[i]
        ax1.scatter(Ls, svn, alpha=0.75, color=color, s=50, label=f"$\\alpha={alpha:.3f}$")

        # linear fit of data points
        a, a_err, b, berr = perform_log_fit(Ls[5:], svn[5:])
        c_eff = 6*a
        c_eff_err = 6*a_err
        ceffs.append(c_eff)
        ceffs_err.append(c_eff_err)
        print(f"c_eff = {c_eff:.4f} +/- {c_eff_err:.2f}")
        #ax1.text(0.02, 0.95-i*0.07, "$c_{\\rm eff}^{\\alpha=" + f"{alpha}" + "}=$" + f"${c_eff:.4f}\\pm{c_eff_err:.4f}$", transform=ax1.transAxes, fontsize=fs2)

        # plot linear fit
        xrange = np.linspace(Ls[5], Ls[-1], 100)
        ax1.plot(xrange, linear_fit(np.log(xrange), a, b), lw=2.5, alpha=0.75, linestyle='--', color=color)



    #ax1.set_xlim([-0.5, -0.2])
    ax1.set_xlim([40, 1000])
    ax1.set_xscale("log")

    ax1.set_xlabel('$\\log(L)$', fontsize=fs1)
    ax1.set_ylabel('$S_{\\rm VN}$', fontsize=fs1)
    #ax2.set_ylim([-0.1, 3.5])
    #ax2.set_ylim([-0.1, 5.5])
    ax1.legend(loc='best', fontsize=fs2)

    ax2.errorbar(np.reciprocal(alpha_values), ceffs, yerr=ceffs_err, marker="o", ms=8, lw=2.5, alpha=0.6, color="#bc4749")
    ax2.set_xlabel('$1/\\alpha$', fontsize=fs1)
    ax2.set_ylabel('$c_{\\rm eff}$', fontsize=fs1)
    ax2.set_ylim(0.75,1.25)

    # save fig
    plt.tight_layout()
    plt.savefig(output_file)

    # Show plot
    plt.show()
