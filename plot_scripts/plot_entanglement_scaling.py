import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import use, rc, rcParams
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble = r"\usepackage[greek, english]{babel}")
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

# --- Parameters ---
D = 0.0
chi = 600
directory = f"../data/ee_scaling/D{D}/Sz0/"  # Change to your directory path if needed
#directory = f"../output/"  # Change to your directory path if needed
pattern = os.path.join(directory, f"spinone_heisenberg_obs_chi{chi}_D{D}_L*")
out_file = (f"../plots/ee_scaling/ee_scaling_D{D}.pdf")


# --- Load and parse all files ---
data = []
for filepath in glob.glob(pattern):
    L_str = os.path.basename(filepath).split("_L")[-1].split(".csv")[0]
    print(L_str)
    try:
        L = int(L_str)
    except ValueError:
        continue
    df = pd.read_csv(filepath)
    df["L"] = L
    data.append(df)

# Combine all into one DataFrame
all_data = pd.concat(data, ignore_index=True)
print(all_data)

# Group by alpha
grouped = all_data.groupby("alpha")

# color map
all_alphas = sorted(grouped.groups.keys())  # or extract from your data directly
norm = colors.Normalize(vmin=min(all_alphas), vmax=max(all_alphas))
cmap = cm.rainbow  # Or use 'plasma', 'inferno', 'cividis', etc.


# For bottom plot
alphas = []
a_coeffs = []

# --- Set up figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=False, gridspec_kw={"height_ratios": [2, 1]})


# --- Define fit function ---
def log_fit(L, a, b):
    return a * np.log(L) + b


# --- Top plot: SvN vs 1/L with fit ---
for i, (alpha, group) in enumerate(grouped):
    #if alpha < 3.0:
    group_sorted = group.sort_values("L")
    L_vals = group_sorted["L"].values
    SvN_vals = group_sorted["SvN"].values

    print(L_vals)
    print(SvN_vals)

    # Fit to a*log(L)+b
    try:
        popt, pcov = curve_fit(log_fit, L_vals, SvN_vals)
        a, b = popt
        alphas.append(alpha)
        a_coeffs.append(a)

        # TODO: Fit exponential S_0 + A*exp(L/2xi)
        # TODO: Or fit power-law near criticality
        # TODO: Check: S ~ n_g/2 log(L)

        # Use colormap
        color = cmap(norm(alpha))

        # Plot original data
        ax1.plot(L_vals, SvN_vals, 'o', c=color, label=f"$\\alpha=${alpha:.3f}")

        # Plot fit
        L_fit = np.linspace(min(L_vals), max(L_vals), 200)
        SvN_fit = log_fit(L_fit, *popt)
        ax1.plot(L_fit, SvN_fit, '--', c=color) #, label=f"fit $a=${a:.3f}, $b=${b:.3f}")
    except RuntimeError:
        print(f"Fit failed for alpha={alpha}")

ax1.set_xlabel("$L$")
ax1.set_ylabel("$S_{\\rm vN}$")
#ax1.set_xscale('log')
#ax1.legend()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for some matplotlib versions
plt.colorbar(sm, ax=ax1, label=r"$\alpha$")

# --- Bottom plot: a vs alpha ---
ax2.plot(alphas, a_coeffs, 'o-')
ax2.set_xlabel("$\\alpha$")
ax2.set_ylabel("Fit coefficient $a$")

plt.tight_layout()
plt.savefig(out_file)
plt.show()