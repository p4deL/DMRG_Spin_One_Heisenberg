import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams['text.usetex'] = True

fs = 18
fs2 = 12

output_file = "../plots/paper/phase_diagram_real.pdf"


def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False, gridspec_kw={'wspace': 0})

    qpt = ("#457b9d", 's-', 6, 1.5, 0.75, "DMRG - data collapse in $D$")
    qpt_svn = ("#bc4749", 'd-', 8, 1.5, 0.75, "DMRG - SvN sym. sectors in $D$")
    qpt_alpha = ("#BD93D8", 'h-', 7, 1.5, 2, 0.75, "DMRG - data collapse in $\\alpha$")
    qpt_pcut = ("#81b29a", '.-', 10, 1.5, 0.75, "pCUT")


    # Ising transition data
    file = "../data/fss/ising_transition/data_collapse_D_mag.csv"
    data = pd.read_csv(file)
    alphas_ising = data["alpha"].values
    alphas_ising[np.isinf(alphas_ising)] = 999999
    D_ising = data["D"].values
    dD_ising = data["dD"].values

    # SU2 transition data
    file = "../data/fss/haldane_SU(2)CSB_transition/data_collapse_alpha_mag.csv"
    data = pd.read_csv(file)
    alphas_su2_alpha = data["alpha"].values
    dalphas_su2_alpha = data["dalpha"].values
    D_su2_alpha = data["D"].values

    # guassian transition data
    file = "../data/fss/gaussian_transition/SvN_intersection_scaling.csv"
    data = pd.read_csv(file)
    alphas_gaussian = data["alpha"].values
    alphas_gaussian[np.isinf(alphas_gaussian)] = 999999
    D_gaussian = data["D"].values
    dD_gaussian = data["dD"].values

    # Haldane - U1 transition data
    file = "../data/fss/haldane_U(1)CSB_transition/data_collapse_alpha_mag.csv"
    data = pd.read_csv(file)
    D_hu1_alpha = data["D"].values
    alphas_hu1_alpha = data["alpha"].values
    dalphas_hu1_alpha = data["dalpha"].values

    # large D - U1 transition data
    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_lambda_mag.csv"
    data = pd.read_csv(file)
    alphas_u1 = data["alpha"].values
    lambdas_u1 = data["lambda"].values
    dlambdas_u1 = data["dlambda"].values

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_alpha_mag.csv"
    data = pd.read_csv(file)
    alphas_u1_alpha = data["alpha"].values
    dalphas_u1_alpha = data["dalpha"].values
    lambdas_u1_alpha = data["lambda"].values

    file = "../data/fss/largeD_U(1)CSB_transition/pcut_1qp_gap.csv"
    data = pd.read_csv(file)
    alphas_pcut = data["alpha"].values
    lambdas_pcut = data["lambda"].values
    dlambdas_pcut = data["dlambda"].values


    # Ising transition
    ax1.errorbar(D_ising, np.reciprocal(alphas_ising), xerr=dD_ising, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3], alpha=qpt[4], label=qpt[5])

    # SU2 transition
    ax1.errorbar(D_su2_alpha, np.reciprocal(alphas_su2_alpha), yerr=1./alphas_su2_alpha**2 * dalphas_su2_alpha, color=qpt_alpha[0], fmt=qpt_alpha[1], ms=qpt_alpha[2], lw=qpt_alpha[3], mew=qpt_alpha[4], alpha=qpt_alpha[5], label=qpt_alpha[6])

    # gaussian transition
    ax1.errorbar(D_gaussian, np.reciprocal(alphas_gaussian), xerr=dD_gaussian, color=qpt_svn[0], fmt=qpt_svn[1], ms=qpt_svn[2], lw=qpt_svn[3], alpha=qpt_svn[4], label=qpt_svn[5])
    ax2.errorbar(np.reciprocal(D_gaussian), np.reciprocal(alphas_gaussian), xerr=dD_gaussian, color=qpt_svn[0], fmt=qpt_svn[1], ms=qpt_svn[2], lw=qpt_svn[3], alpha=qpt_svn[4], label=qpt_svn[5])

    # U1 transition
    # D plane
    ax1.errorbar(D_hu1_alpha, np.reciprocal(alphas_hu1_alpha), yerr=1./alphas_hu1_alpha**2 * dalphas_hu1_alpha, color=qpt_alpha[0], fmt=qpt_alpha[1], ms=qpt_alpha[2], lw=qpt_alpha[3], mew=qpt_alpha[4], alpha=qpt_alpha[5], label=qpt_alpha[6])
    # lambda plane
    ax2.errorbar(lambdas_u1, np.reciprocal(alphas_u1), xerr=dlambdas_u1, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3], alpha=qpt[4], label=qpt[5])
    ax2.errorbar(lambdas_u1_alpha, np.reciprocal(alphas_u1_alpha), yerr=1./alphas_u1_alpha**2 * dalphas_u1_alpha, color=qpt_alpha[0], fmt=qpt_alpha[1], ms=qpt_alpha[2], lw=qpt_alpha[3], mew=qpt_alpha[4], alpha=qpt_alpha[5], label=qpt_alpha[6])
    ax2.errorbar(lambdas_pcut, np.reciprocal(alphas_pcut), xerr=dlambdas_pcut, color=qpt_pcut[0], fmt=qpt_pcut[1], ms=qpt_pcut[2], lw=qpt_pcut[3], alpha=qpt_pcut[4], label=qpt_pcut[5])


    # Remove extra space between subplots
    ax2.set_yticklabels([])
    # Label axes
    # ax1.set_xlabel(r'$D$', fontsize=fs2)
    ax2.set_xlim([1.0, 0.02])
    ax2.set_ylim([0.0, 1.0])
    ax1.set_xlim([-0.5, 1.0])
    ax1.set_ylim([0.0, 1.0])
    # Set custom tick positions and labels
    tick_positions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    tick_labels = ["$1.0$", "$5/4$", "$5/3$", "$5/2$", "$5$", "$\\infty$"]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    # ax2.set_xlabel('$D$', fontsize=fs2)
    ax1.set_ylabel('$1/\\alpha$', fontsize=fs)
    ax1.set_xlabel('$D$', fontsize=fs, color='white')

    ax1.vlines(x=0.0, ymin=0.37, ymax=1.0, color='gray', lw=2)

    # Remove left spine from the first plot
    ax1.spines['right'].set_visible(False)
    #ax1.yaxis.set_ticks([])

    # Remove right spine from the second plot
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_ticks([])

    # labels
    fig.text(0.5, 0.02, "$D$", ha='center', fontsize=fs)
    fig.text(0.35, 0.25, "Haldane", ha='center', fontsize=fs)
    fig.text(0.14, 0.6, "z-AF", ha='center', fontsize=fs)
    fig.text(0.9, 0.25, "large D", ha='center', fontsize=fs)
    fig.text(0.7, 0.6, "U(1) CSB", ha='center', fontsize=fs)
    fig.text(0.24, 0.72, "SU(2) CSB", ha='center', fontsize=fs, rotation=90)

    # large D, zAF, SU(2) CSB, Haldane, U(1) CSB
    probes = [(1.4, 10.0), (-0.3, 1.5), (0.0, 10.0), (0.0, 1.5), (1.0, 1.5)] #, (0.1, 2.8)]

    colors = cm.viridis(np.linspace(0, 1, len(probes)))


    #colors = ['#E9EB9E', '#ACC196', '#93AB96', '#799496', '#49475B'] #, '#14080E']

    for idx, (probe, color) in enumerate(zip(probes, colors)):
        if idx == 0 or idx > 3:
            ax2.scatter(1./probe[0], 1./probe[1], color=color, marker='+', s=100, lw=3, zorder=10)

        ax1.scatter(probe[0], 1./probe[1], color=color, marker='+', s=100, lw=3, zorder=10)


    ax2.legend(bbox_to_anchor=(0.65, 1.0), fontsize=fs2)

    # save fig
    plt.tight_layout()
    plt.savefig(output_file)

    # Show plot
    plt.show()

if __name__ == '__main__':
    plot()


