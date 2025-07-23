
import pandas as pd

import numpy as np
from uncertainties import ufloat
from uncertainties.umath import *  # optional for math functions
from uncertainties import nominal_value, std_dev
#from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import use, rc, rcParams
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"

fixed_val = 1.6666666666666667

# Load the CSV file
#file_path = f"../data/fss/largeD_U(1)CSB_transition/alpha{fixed_val}/data_collapse_m_trans_alpha{fixed_val}.csv"  # Change path if needed
#file_path = f"../data/fss/largeD_U(1)CSB_transition/alpha{fixed_val}/data_collapse_m_trans_biased_alpha{fixed_val}.csv"  # Change path if needed
#out_file = f"../plots/fss/u1_csb/data_collapse_scaling_biased_alpha{fixed_val}.pdf"
file_path = f"../data/fss/largeD_U(1)CSB_transition/D{fixed_val}/data_collapse_m_trans_D{fixed_val}.csv"  # Change path if needed
out_file = f"../plots/fss/u1_csb/data_collapse_scaling_D{fixed_val}.pdf"
df = pd.read_csv(file_path)

fs = 16

# Compute 1 / L_min for plotting
df['inv_L_min'] = 1 / df['L_min']

# Set up plot
fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True)
ax_crit, ax_nu, ax_exp = axes

# Get unique red_n_points
red_n_values = sorted(df['red_n_points'].unique())

b_collection = {
    'x_c': [],
    'nu': [],
    'exp': []
}

# Loop over each red_n_points group
for red_n in red_n_values:
    df_sub = df[df['red_n_points'] == red_n].copy()
    df_sub.sort_values('L_min', inplace=True)

    # Plot crit val
    ax_crit.errorbar(df_sub['inv_L_min'], df_sub['x_c'], yerr=df_sub['dx_c'],
                   label=f'red_n={red_n}', marker='o', linestyle='-', zorder=-1)

    # Plot nu
    ax_nu.errorbar(df_sub['inv_L_min'], df_sub['nu'], yerr=df_sub['dnu'],
                   label=f'red_n={red_n}', marker='o', linestyle='-', zorder=-1)

    # Plot exp
    ax_exp.errorbar(df_sub['inv_L_min'], df_sub['exp'], yerr=df_sub['dexp'],
                    label=f'red_n={red_n}', marker='o', linestyle='-', zorder=-1)

    # Linear fit to leading 4 L_min points in range 220-280
    #df_fit = df_sub[df_sub['L_min'].between(200, 280)].head(4)
    #if len(df_fit) >= 2:
    #    for ax, ycol in zip([ax_nu, ax_exp], ['nu', 'exp']):
    #        slope, intercept, *_ = linregress(df_fit['inv_L_min'], df_fit[ycol])
    #        #x_fit = np.linspace(df_fit['inv_L_min'].min(), df_fit['inv_L_min'].max(), 100)
    #        x_fit = np.linspace(0., df_fit['inv_L_min'].min(), 100)
    #        y_fit = slope * x_fit + intercept
    #        ax.plot(x_fit, y_fit, linestyle='--', alpha=0.7)

    # Linear fit for L_min in [200, 280]
    Lmin_fit = 160
    Lmax_fit = 340
    df_fit = df_sub[df_sub['L_min'].between(Lmin_fit, Lmax_fit)]
    if len(df_fit) >= 2:
        x_fit = np.linspace(1 / Lmin_fit, 0, 200)
        b_vals = []
        b_err_vals = []

        for ax, ycol, dycol in zip([ax_crit, ax_nu, ax_exp], ['x_c', 'nu', 'exp'], ['dx_c', 'dnu', 'dexp']):

            print(f"val={ycol}")
            popt, pcov = curve_fit(lambda x, a, b: a * x + b, df_fit['inv_L_min'], df_fit[ycol], sigma=df_fit[dycol],
                                   absolute_sigma=True)
            #popt, pcov = curve_fit(lambda x, a, b: a * x + b, df_fit['inv_L_min'], df_fit[ycol])
            #slope, intercept, *_ = linregress(df_fit['inv_L_min'], df_fit[ycol])
            a, b = popt  # coefficients
            da, db = np.sqrt(np.diag(pcov))  # standard deviations (errors)
            print(b, db)
            b_collection[ycol].append((b, db))
            y_fit = a * x_fit + b
            ax.plot(x_fit, y_fit, linestyle='--', alpha=1.0, lw=2, c="gray", zorder=1) #, label=f'fit red_n={red_n}')
            #ax.errorbar(0.0, b, yerr=db, alpha=1.0, marker="o", ms=10, lw=4, c="gray", zorder=1)

# After red_n loop: compute weighted averages with chi² correction
results_str = ""
for ax, ycol, label in zip([ax_crit, ax_nu, ax_exp], ['x_c', 'nu', 'exp'], ["$\\lambda_c$", "$\\nu$", "$\\beta$"]):
    b_vals, b_err_vals = zip(*b_collection[ycol])
    b_ufloats = [ufloat(val, err) for val, err in zip(b_vals, b_err_vals)]

    # Weighted average
    weights = [1 / err**2 for err in b_err_vals]
    weighted_sum = sum(w * b for w, b in zip(weights, b_ufloats))
    total_weight = sum(weights)
    b_avg = weighted_sum / total_weight

    # Chi-squared calculation
    chi2 = sum(((b.nominal_value - b_avg.nominal_value) ** 2 / std_dev(b) ** 2) for b in b_ufloats)
    dof = len(b_ufloats) - 1
    chi2_red = chi2 / dof if dof > 0 else 1

    # Adjusted uncertainty
    std_dev_adjusted = std_dev(b_avg) * np.sqrt(chi2_red) if chi2_red > 1 else std_dev(b_avg)
    nom_b_val = nominal_value(b_avg)

    # Output
    print(f"\nObservable: {ycol}")
    print(f"Weighted average intercept b: {nominal_value(b_avg):.5f} ± {std_dev(b_avg):.5f}")
    print(f"Adjusted (chi²) error: ± {std_dev_adjusted:.5f}")
    print(f"Reduced chi-squared: {chi2_red:.2f}")


    ax.errorbar(0.0, nom_b_val, yerr=std_dev_adjusted,  marker="o", ms=10, lw=4, c="black", zorder=2)
    #ax.errorbar(0.0, nominal_value(b_avg), yerr=std_dev(b_avg),  marker="o", ms=10, lw=4, c="black", zorder=2)

    plot_res_str = f"{label}$ = {nom_b_val:.5f}\\pm{std_dev_adjusted:.5f}$"
    #resstr = '\n'.join((xlabel + '$\\hspace{-0.5em}\\phantom{x}_c=%.4f$' % (x_c, ), '$\\nu=%.4f$' % (nu, ), '$\\beta=%.4f$' % exp, ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.35, 0.05, plot_res_str, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)
    results_str += f"{nom_b_val},{std_dev_adjusted},"

print("x_c,dx_c,nu,dnu,exp,dexp")
print(results_str)

# Finalize plot formatting
ax_crit.set_xlabel(r'$1/L_{\min}$', fontsize=fs)
ax_crit.set_ylabel(r'$\lambda_c$', fontsize=fs)
ax_crit.legend()
ax_crit.set_xlim(0,0.018)

ax_nu.set_xlabel(r'$1/L_{\min}$', fontsize=fs)
ax_nu.set_ylabel(r'$\nu$', fontsize=fs)
ax_nu.legend()
ax_nu.set_xlim(0,0.018)

ax_exp.set_xlabel(r'$1/L_{\min}$', fontsize=fs)
ax_exp.set_ylabel('$\\beta$', fontsize=fs)
ax_exp.legend()
ax_exp.set_xlim(0,0.018)

plt.tight_layout()
plt.savefig(out_file)
plt.show()