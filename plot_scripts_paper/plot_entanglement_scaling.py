import numpy as np
import matplotlib

matplotlib.rcParams['text.usetex'] = True
from matplotlib import use, rc, rcParams

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage[greek, english]{babel}")
rcParams['pgf.preamble'] = r"\usepackage[greek, english]{babel}"
from matplotlib import pyplot as plt
from scipy import optimize
import sys
import os

sys.path.insert(0, os.path.abspath('../'))

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.colors as colors

fs = 18
fs2 = 12

# --- Parameters ---
Ds = [-0.5, 0.0, 0.5, 1.25]

alpha_maxs = [0,-10,-7,0]


#directory = f"../output/"  # Change to your directory path if needed
labels = ("$m_z=0$", "$m_z=1$")
out_file = (f"../plots/paper/ee_scaling.pdf")

Lmins = [60, 80, 100, 120, 140, 160, 180, 200]
num_Lmins = len(Lmins)
n_fit_vals = 5

colors = ['#457b9d','#81b29a']

# --- Define fit function ---
def log_fit(L, a, b):
    return a * np.log(L) + b


def perform_fit(L_vals, SvN_vals):
    try:
        # Fit to a*log(L)+b
        popt, pcov = curve_fit(log_fit, L_vals, SvN_vals)
        a, b = popt
        # TODO: Fit exponential S_0 + A*exp(L/2xi)
        # TODO: Or fit power-law near criticality
        # TODO: Check: S ~ n_g/2 log(L)

    except RuntimeError:
        print(f"Fit failed for alpha={alpha}")

    return a, b


def read_fss_data(pattern):
    data = []
    for filepath in glob.glob(pattern):
        L_str = os.path.basename(filepath).split("_L")[-1].split(".csv")[0]
        try:
            L = int(L_str)
        except ValueError:
            continue
        df = pd.read_csv(filepath)
        df["L"] = L
        data.append(df)


    # Combine all into one DataFrame
    all_data = pd.concat(data, ignore_index=True)

    # Group by alpha
    data_by_alphas = all_data.groupby("alpha")

    # color map
    alphas = sorted(data_by_alphas.groups.keys())  # or extract from your data directly

    return alphas, data_by_alphas

def fit_log_scalings(Ls, SvNs, num_Lmins=8):
    # prepare list for finite size scaling
    a_scaling = []
    # iterate over number of minimal Ls, i.e., in what range to fit
    for i in range(num_Lmins):

        a, b = perform_fit(Ls[i:], SvNs[i:])
        a_scaling.append(a)

    return a_scaling


# --- Set up figure ---
fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=False)


for ax, alpha_max, D in zip(axs, alpha_maxs, Ds):

    if D < 0.0 or D > 1.0:
        directories = [f"../data/ee_scaling/D{D}/Sz0/"]
    else:
        directories = [f"../data/ee_scaling/D{D}/Sz0/",
                       f"../data/ee_scaling/D{D}/Sz1/"]  # Change to your directory path if needed

    for idx, (dir, color, label) in enumerate(zip(directories, colors, labels)):

        pattern = os.path.join(dir, f"spinone_heisenberg_obs_chi*_D{D}_L*")
        print(pattern)
        alphas, data_by_alpha = read_fss_data(pattern)
        if len(alphas) == 0:
            continue

        a_coeffs_min = []
        a_coeffs_max = []
        a_infinity = []
        da_infinity = []


        # iterate over every alpha
        for alpha, fss_data in data_by_alpha:

            # read in data for given alpha
            fss_data_sorted = fss_data.sort_values("L")
            L_vals = fss_data_sorted["L"].values
            SvN_vals = fss_data_sorted["SvN"].values

            # prepare list for finite size scaling
            L_scaling = [1./L for L in L_vals[:num_Lmins]]
            a_scaling = fit_log_scalings(L_vals, SvN_vals, num_Lmins=num_Lmins)
            a_coeffs_min.append(a_scaling[0])
            a_coeffs_max.append(a_scaling[-1])

            # extrapolate to the thermodynamic limit using a linear fit
            popt, pcov = curve_fit(lambda x, a, b: a + b * x, L_scaling[-n_fit_vals:], a_scaling[-n_fit_vals:],
                                   p0=[a_scaling[-1], 1.0])
            a, b = popt  # coefficients
            da = np.sqrt(pcov[0, 0])  # standard deviations (errors)
            a_infinity.append(a)
            da_infinity.append(da)

        # --- Bottom plot: a vs alpha ---
        if idx == 0 and alpha_max < 0:
            alphas = alphas[:alpha_max]
            a_coeffs_min = a_coeffs_min[:alpha_max]
            a_coeffs_max = a_coeffs_max[:alpha_max]
            a_infinity = a_infinity[:alpha_max]
            da_infinity = da_infinity[:alpha_max]

        #ax.plot(alphas, a_coeffs_min, 'o-', alpha=0.33, color=f"C{idx}") #, label="$L_{\\rm{min}}=$" + f" {int(1./L_scaling[0])}")
        #ax.plot(alphas, a_coeffs_max, 'o-', alpha=0.66, color=f"C{idx}") #, label="$L_{\\rm{min}}=$" + f" {int(1./L_scaling[num_Lmins-1])}")
        #ax.errorbar(alphas, a_infinity, yerr=da_infinity, marker="o", color=f"C{idx}", label="$L_{\\rm{min}}\\rightarrow\\infty$")
        ax.errorbar(alphas, a_infinity, yerr=da_infinity, marker="o", color=color, label=label)
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.02, 0.1, f"$D={D}$", transform=ax.transAxes, fontsize=12,
                 verticalalignment='center', bbox=props)

        ax.set_xlabel("$\\alpha$", fontsize=fs)
        ax.set_ylabel("$\ell_{E}$", fontsize=fs)
        ax.legend(loc="upper right")

axs[0].set_ylim(-0.2,0.2)
axs[1].set_ylim(-0.2,1.0)
axs[2].set_ylim(-0.2,0.5)
axs[3].set_ylim(-0.2,0.5)

axs[3].axvline(x=2.917929279402158, color='gray', linestyle='--')


plt.tight_layout()
plt.savefig(out_file)
plt.show()
