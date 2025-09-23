import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams['text.usetex'] = True

fs = 18
fs2 = 12

output_file = "../plots/paper/all_exponents.pdf"


def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False, gridspec_kw={'wspace': 0})

    qpt = ("#457b9d", 's', 8, 1.5, 0.75, "DMRG - data collapse in $D$")
    qpt_alpha = ("#BD93D8", 'h', 9, 1.5, 2, 0.75, "DMRG - data collapse in $\\alpha$")
    qpt_bias = ("#25a18e", '*', 12, 1.5, 0.75, "DMRG biased -- data collapse in $\\lambda$")


    # Ising transition data
    file = "../data/fss/ising_transition/data_collapse_D_mag.csv"
    data = pd.read_csv(file)
    #alphas_ising = data["alpha"].values
    #alphas_ising[np.isinf(alphas_ising)] = 999999
    D_ising = data["D"].values
    dD_ising = data["dD"].values
    beta_ising = data["beta"].values
    dbeta_ising = data["dbeta"].values

    # SU2 transition data
    file = "../data/fss/haldane_SU(2)CSB_transition/data_collapse_alpha_mag.csv"
    data = pd.read_csv(file)
    #alphas_su2_alpha = data["alpha"].values
    #dalphas_su2_alpha = data["dalpha"].values
    D_su2_alpha = data["D"].values
    beta_su2_alpha = data["beta"].values
    dbeta_su2_alpha = data["dbeta"].values

    # Haldane - U1 transition data
    file = "../data/fss/haldane_U(1)CSB_transition/data_collapse_alpha_mag.csv"
    data = pd.read_csv(file)
    D_hu1_alpha = data["D"].values
    #alphas_hu1_alpha = data["alpha"].values
    #dalphas_hu1_alpha = data["dalpha"].values
    beta_hu1_alpha = data["beta"].values
    dbeta_hu1_alpha = data["dbeta"].values

    # large D - U1 transition data
    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_lambda_mag.csv"
    data = pd.read_csv(file)
    #alphas_u1 = data["alpha"].values
    lambdas_u1 = data["lambda"].values
    dlambdas_u1 = data["dlambda"].values
    beta_u1 = data["beta"].values
    dbeta_u1 = data["dbeta"].values

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_lambda_mag_biased.csv"
    data = pd.read_csv(file)
    lambdas_bias = data["lambda"].values
    dlambdas_bias = data["dlambda"].values
    beta_bias = data["beta"].values
    dbeta_bias = data["dbeta"].values

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_alpha_mag.csv"
    data = pd.read_csv(file)
    #alphas_u1_alpha = data["alpha"].values
    #dalphas_u1_alpha = data["dalpha"].values
    lambdas_u1_alpha = data["lambda"].values
    beta_u1_alpha = data["beta"].values
    dbeta_u1_alpha = data["dbeta"].values

    # Ising transition
    ax1.errorbar(D_ising, beta_ising, yerr=dbeta_ising, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3], alpha=qpt[4], label=qpt[5])

    # SU2 transition
    ax1.errorbar(D_su2_alpha, beta_su2_alpha, yerr=dbeta_su2_alpha, color=qpt_alpha[0], fmt=qpt_alpha[1], ms=qpt_alpha[2], lw=qpt_alpha[3], mew=qpt_alpha[4], alpha=qpt_alpha[5], label=qpt_alpha[6])

    # U1 transition
    # D plane
    ax1.errorbar(D_hu1_alpha, beta_hu1_alpha, yerr=dbeta_hu1_alpha, color=qpt_alpha[0], fmt=qpt_alpha[1], ms=qpt_alpha[2], lw=qpt_alpha[3], mew=qpt_alpha[4], alpha=qpt_alpha[5], label=qpt_alpha[6])
    # lambda plane
    ax2.errorbar(lambdas_u1, beta_u1, xerr=dlambdas_u1, yerr=dbeta_u1, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3], alpha=qpt[4], label=qpt[5])
    ax2.errorbar(lambdas_bias, beta_bias, yerr=dbeta_bias, color=qpt_bias[0], fmt=qpt_bias[1], ms=qpt_bias[2], lw=qpt_bias[3], alpha=qpt_bias[4], label=qpt_bias[5])
    ax1.errorbar(np.reciprocal(lambdas_u1_alpha), beta_u1_alpha, yerr=dbeta_u1_alpha, color=qpt_alpha[0], fmt=qpt_alpha[1], ms=qpt_alpha[2], lw=qpt_alpha[3], mew=qpt_alpha[4], alpha=qpt_alpha[5], label=qpt_alpha[6])
    ax2.errorbar(lambdas_u1_alpha, beta_u1_alpha, yerr=dbeta_u1_alpha, color=qpt_alpha[0], fmt=qpt_alpha[1], ms=qpt_alpha[2], lw=qpt_alpha[3], mew=qpt_alpha[4], alpha=qpt_alpha[5], label=qpt_alpha[6])


    # Remove extra space between subplots
    ax2.set_yticklabels([])
    # Label axes
    # ax1.set_xlabel(r'$D$', fontsize=fs2)
    ax2.set_xlim([1.0, 0.02])
    ax2.set_ylim([0.0, 1.5])
    ax1.set_xlim([-0.5, 1.0])
    ax1.set_ylim([0.0, 1.5])
    # Set custom tick positions and labels
    tick_positions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    tick_labels = ["$1.0$", "$5/4$", "$5/3$", "$5/2$", "$5$", "$\\infty$"]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    # ax2.set_xlabel('$D$', fontsize=fs2)
    ax1.set_ylabel('$\\beta$', fontsize=fs)
    ax1.set_xlabel('$D$', fontsize=fs, color='white')
    fig.text(0.5, 0.02, "$D$", ha='center', fontsize=fs)

    # Remove left spine from the first plot
    ax1.spines['right'].set_visible(False)
    #ax1.yaxis.set_ticks([])

    # Remove right spine from the second plot
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_ticks([])

    # labels
    fig.text(0.37, 0.6, "Haldane $\\leftrightarrow$ U(1) CSB", ha='center', fontsize=fs)
    fig.text(0.17, 0.6, "Ising", ha='center', fontsize=fs)
    fig.text(0.75, 0.6, "large D $\\leftrightarrow$ U(1) CSB", ha='center', fontsize=fs)
    #fig.text(0.7, 0.6, "U(1) CSB", ha='center', fontsize=fs)
    fig.text(0.222, 0.5, "Haldane $\\leftrightarrow$ SU(2) CSB", ha='center', fontsize=fs, rotation=90)

    # large D, zAF, SU(2) CSB, Haldane, U(1) CSB

    ax1.axvspan(-0.325, -0.011, color='#bbd1ea', alpha=0.3)
    ax1.axvspan(-0.01, 0.01, color='#d9d9d9', alpha=0.3)
    ax1.axvspan(0.011, 0.98, color='#faf0ca', alpha=0.3)
    ax1.axvspan(0.99, 1.01, color='#e56b6f', alpha=0.2)
    ax2.axvspan(0.99, 1.01, color='#e56b6f', alpha=0.2)
    ax2.axvspan(0.99, 0.0, color='#99e2b4', alpha=0.1)



    ax2.legend(bbox_to_anchor=(0.55, 0.75), fontsize=fs2)

    # save fig
    plt.tight_layout()
    plt.savefig(output_file)

    # Show plot
    plt.show()

if __name__ == '__main__':
    plot()


