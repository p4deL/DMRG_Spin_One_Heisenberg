import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import curve_fit

from matplotlib import use, rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"

L = 100 # system size
D = 0.0
alpha = 'inf'
G = 1.0
Jz = 1.0
chi = 300
str = "psi3"

# output filename
output_file = f"../plots/analyze_gs_{str}_alpha{alpha}_L{L}_Jz{Jz}.pdf"

# directory and filename
data_dir = f'../output/'


filename_entropy = f"spinone_heisenberg_{str}_entropies_chi{chi}_D{D}_Gamma{G}_Jz{Jz}_alpha{alpha}_L{L}.csv"
filename_mz = f"spinone_heisenberg_{str}_mzs_chi{chi}_D{D}_Gamma{G}_Jz{Jz}_alpha{alpha}_L{L}.csv"
filename_correlations = f"spinone_heisenberg_{str}_correlations_chi{chi}_D{D}_Gamma{G}_Jz{Jz}_alpha{alpha}_L{L}.csv"


def plot_entropy_profile(ax, filename):
    # Set labels and titles for each subplot (optional)
    file = os.path.join(data_dir, filename)
    data = pd.read_csv(file)

    # ax.text(0.98, 0.87, label, transform=ax.transAxes, ha='right', fontsize=fs2)
    # ax.set_ylim(-0.5,0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    pos = data['pos'].values + 0.5 * np.ones(len(data['pos'].values))
    entropies = data['SvN'].values
    # label = r'$\vert\langle S_1^zS_j^z \rangle\vert$',

    ax.plot(pos, entropies, marker=".", markersize=12)

    ax.set_ylabel("$S_{\\rm VN}$", fontsize=fs2)
    ax.set_xlabel("$\\ell$", fontsize=fs2)


def plot_correlations(ax, filename):
    file = os.path.join(data_dir, filename)
    data = pd.read_csv(file)

    pos = data['pos'].values + np.ones(len(data['pos'].values))
    corrxx = 0.5 * data['corr_pm'].values
    corryy = 0.5 * data['corr_mp'].values
    corrzz = data['corr_zz'].values
    corrstr = data['corr_str_order'].values

    # ax.plot(pos, corrstr, label=r'$\mathcal{O}^z_{1,j}$', marker="s", color=colors[0])
    ax.plot(pos, corrxx, label=r'$\langle S_1^xS_j^x \rangle$', marker="x", ms=6)
    ax.plot(pos, corryy, label=r'$\langle S_1^yS_j^y \rangle$', marker="+", ms=8)
    ax.plot(pos, corrzz, label=r'$\langle S_1^zS_j^z \rangle$', marker=".", ms=6)
    ax.plot(pos, corrstr, label=r'$\mathcal{O}_{\rm str}$', marker=".", ms=6)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    #ax.set_ylim(-0.85, 0.85)
    ax.set_xlabel("$j$", fontsize=fs2)


def plot_mz_profile(ax, filename):
    # Set labels and titles for each subplot (optional)
    file = os.path.join(data_dir, filename)
    data = pd.read_csv(file)

    # ax.text(0.98, 0.87, label, transform=ax.transAxes, ha='right', fontsize=fs2)
    # ax.set_ylim(-0.5,0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    pos = data['pos'].values + 0.5 * np.ones(len(data['pos'].values))
    entropies = data['m_long'].values
    # label = r'$\vert\langle S_1^zS_j^z \rangle\vert$',

    ax.plot(pos, entropies, marker=".", markersize=10)

    ax.set_ylabel("$M^{z}$", fontsize=fs2)
    ax.set_xlabel("$j$", fontsize=fs2)


fs1 = 18
fs2 = 13

# Create the figure and the subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

(ax1, ax2, ax3) = axs

plot_entropy_profile(ax1, filename_entropy)
plot_correlations(ax2, filename_correlations)
plot_mz_profile(ax3, filename_mz)

#plt.title(f"$D={D}\\sim D_c$", fontsize=fs1)
fig.legend()
# Ensure that subplots are tightly packed vertically
plt.subplots_adjust(hspace=0.1, top=0.9)

## save figure
plt.savefig(output_file)

# Show the plot
plt.show()


