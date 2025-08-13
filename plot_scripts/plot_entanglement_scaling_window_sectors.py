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

# --- Parameters ---
D = 0.0


directories = [f"../data/ee_scaling/D{D}/Sz0/", f"../data/ee_scaling/D{D}/Sz1/"]   # Change to your directory path if needed
#directory = f"../output/"  # Change to your directory path if needed
chis = (800, 500)
labels = ("$S^z=0$", "$S^z=1$")
out_file = (f"../plots/ee_scaling/ee_scaling_D{D}.pdf")

#Lmins = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
#Lmins = [60, 80, 100, 120, 140, 160, 180, 200]
Lmins = [60, 80, 100, 120, 140]
num_Lmins = len(Lmins)
n_fit_vals = 5


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


# --- Set up figure ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10), sharex=False)

# color map
norm = colors.Normalize(vmin=1.1, vmax=6.0)
cmap = cm.rainbow  # Or use 'plasma', 'inferno', 'cividis', etc.

for idx, (chi, dir, label) in enumerate(zip(chis, directories, labels)):

    pattern = os.path.join(dir, f"spinone_heisenberg_obs_chi{chi}_D{D}_L*")

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
    grouped = all_data.groupby("alpha")

    # color map
    alphas = sorted(grouped.groups.keys())  # or extract from your data directly
    #norm = colors.Normalize(vmin=min(alphas), vmax=max(alphas))
    #cmap = cm.rainbow  # Or use 'plasma', 'inferno', 'cividis', etc.

    # For bottom plot
    a_coeffs1 = []
    a_coeffs2 = []

    # iterate
    alpha_infinity = []
    a_infinity = []
    da_infinity = []

    # iterate over every alpha
    for alpha, group in grouped:

        # read in data for given alpha
        group_sorted = group.sort_values("L")
        L_vals = group_sorted["L"].values
        SvN_vals = group_sorted["SvN"].values

        # prepare list for finite size scaling
        a_scaling = []
        L_scaling = [1./L for L in L_vals[:num_Lmins]]
        # iterate over number of minimal Ls, i.e., in what range to fit
        for i in range(num_Lmins):

            a, b = perform_fit(L_vals[i:], SvN_vals[i:])
            a_scaling.append(a)

            if i == 0:
                a_coeffs1.append(a)

                # Use colormap
                color = cmap(norm(alpha))

                # Plot original data
                ax1.plot(L_vals, SvN_vals, 'o', c=color, label=f"$\\alpha=${alpha:.3f}")

                # Plot fit
                L_fit = np.linspace(min(L_vals), max(L_vals), 200)
                SvN_fit = log_fit(L_fit, a, b)
                ax1.plot(L_fit, SvN_fit, '--', c=color)  #, label=f"fit $a=${a:.3f}, $b=${b:.3f}")

            # append
            if i == num_Lmins-1:
                a_coeffs2.append(a)

        # do the extrapolation to the thermodynamic limit
        #if alpha > 1.0:  #and  alpha < 4.3:
        if idx == 0 and alpha < 3.2 or idx == 1:  # and  alpha < 4.3:
            ax2.set_xlabel("$1/L$")
            ax2.set_ylabel("$a$")
            ax2.scatter(L_scaling, a_scaling, color=color)

            # TODO: Try different fit functions
            # TODO: linear fit works
            # TODO: Quadratic fit also works
            # TODO: Can I make exponential fit work?
            #popt, pcov = curve_fit(lambda x, a, b, c: a + b*x**c, L_scaling, a_scaling, p0=[a_scaling[-1],0.05, -1.0])
            #popt, pcov = curve_fit(lambda x, a, c: a + x**c, L_scaling, a_scaling, p0=[a_scaling[-1],0.8])
            popt, pcov = curve_fit(lambda x, a, b: a + b*x, L_scaling[-n_fit_vals:], a_scaling[-n_fit_vals:], p0=[a_scaling[-1],1.0])
            #a, c = popt  # coefficients
            #print(a,c)
            #popt, pcov = curve_fit(lambda x, a, b, c: a + b*x**(-c), L_scaling, a_scaling, p0=[a,-1.0, c])
            #a, b, c = popt  # coefficients
            #print(a,b,c)
            a, b = popt  # coefficients
            da = np.sqrt(pcov[0,0])  # standard deviations (errors)
            alpha_infinity.append(alpha)
            a_infinity.append(a)
            da_infinity.append(da)
            #db = np.sqrt(pcov[1,1])  # standard deviations (errors)
            #dc = np.sqrt(pcov[2,2])  # standard deviations (errors)
            #a = a_scaling[-1]
            #da = 0.0
            #b = 0.05
            #c = 0.8
            Lmin_fit = 60
            x_fit = np.linspace(0, L_scaling[-n_fit_vals], 200)
            #y_fit = a + b*x_fit**c
            y_fit = a + b*x_fit
            #y_fit = a - x_fit**c
            ax2.plot(x_fit, y_fit, linestyle='--', color=color)
            #ax2.set_ylim(-0.01,0.2)
            #print("alpha,x_c,dx_c")
            #print(f"{alpha},{a},{da}")

    # --- Bottom plot: a vs alpha ---
    ax3.plot(alphas, a_coeffs1, 'o-', alpha=0.33, color=f"C{idx}", label="$L_{\\rm{min}}=$" + f" {int(1./L_scaling[0])}")
    ax3.plot(alphas, a_coeffs2, 'o-', alpha=0.66, color=f"C{idx}", label="$L_{\\rm{min}}=$" + f" {int(1./L_scaling[num_Lmins-1])}")
    ax3.errorbar(alpha_infinity, a_infinity, yerr=da_infinity, marker="o", color=f"C{idx}", label="$L_{\\rm{min}}\\rightarrow\\infty$")

ax1.set_xlabel("$L$")
ax1.set_ylabel("$S_{\\rm vN}$")
#ax1.set_xscale('log')
#ax1.legend()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for some matplotlib versions
plt.colorbar(sm, ax=ax1, label=r"$\alpha$")


ax3.set_xlabel("$\\alpha$")
ax3.set_ylabel("Fit coefficient $a$")
ax3.set_ylim(-0.1, 1.0)
ax3.legend(loc="best")

plt.tight_layout()
plt.savefig(out_file)
plt.show()
