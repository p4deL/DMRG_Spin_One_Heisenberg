import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams['text.usetex'] = True

fs = 18
fs2 = 12


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

colors = get_color_gradient("#ff006e", "#3a86ff", 5)

def plot():
    fig = plt.figure(figsize=(6.51, 8.51))
    # matplotlib.figure.SubplotParams(left=0.0,right=1.0,bottom=0.5,top=1.0)

    gs = GridSpec(5, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    ax1.set_position([0.11, 0.76, 0.85, 0.23], which='both')
    ax2.set_position([0.11, 0.45, 0.35, 0.23], which='both')
    ax3.set_position([0.61, 0.45, 0.35, 0.23], which='both')
    ax4.set_position([0.11, 0.14, 0.35, 0.23], which='both')
    ax5.set_position([0.61, 0.14, 0.35, 0.23], which='both')

    qpt = ("#3a86ff", 's-', 5, 1, "DMRG - data collapse in $\\lambda$")
    qpt_pcut = ("#ff006e", '.-', 9, 1, "pCUT - DlogPadé")
    qpt2 = ("#BD93D8", 'X-', 6, 1, "DMRG - data collapse in $\\alpha$")
    exp = ("#3a86ff", 's', 6, "DMRG - data collapse in $\\lambda$")
    exp_bias = ("#83BCA9", '*', 10, "DMRG biased - data collapse in $\\lambda$")
    exp_pcut = ("#ff006e", '.', 11, "pCUT - DlogPadé")
    exp2 = ("#BD93D8", 'X', 7, "DMRG - data collapse in $\\alpha$")

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_lambda_mag.csv"
    data = pd.read_csv(file)
    alphas = data["alpha"].values
    alphas[np.isinf(alphas)] = 999999
    lambdas = data["lambda"].values
    dlambdas = data["dlambda"].values
    nus = data["nu"].values
    dnus = data["dnu"].values
    betas = data["beta"].values
    dbetas = data["dbeta"].values


    file = "../data/fss/largeD_U(1)CSB_transition/pcut_1qp_gap.csv"
    data = pd.read_csv(file)
    alphas_pcut = data["alpha"].values
    alphas_pcut[np.isinf(alphas_pcut)] = 999999
    lambdas_pcut = data["lambda"].values
    dlambdas_pcut = data["dlambda"].values
    znus = data["exp"].values
    dznus = data["dexp"].values

    file = "../data/fss/largeD_U(1)CSB_transition/pcut_1qp_sw.csv"
    data = pd.read_csv(file)
    alphas_pcut = data["alpha"].values
    alphas_pcut[np.isinf(alphas_pcut)] = 999999
    exps = data["exp"].values
    dexps = data["dexp"].values

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_lambda_mag_biased.csv"
    data = pd.read_csv(file)
    alphas_bias = data["alpha"].values
    #alphas_bias[np.isinf(alphas)] = 999999
    nus_bias = data["nu"].values
    dnus_bias = data["dnu"].values
    betas_bias = data["beta"].values
    dbetas_bias = data["dbeta"].values

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_alpha_mag.csv"
    data = pd.read_csv(file)
    alphas2 = data["alpha"].values
    dalphas2 = data["dalpha"].values
    lambdas2 = data["lambda"].values
    nus2 = data["nu"].values
    dnus2 = data["dnu"].values
    betas2 = data["beta"].values
    dbetas2 = data["dbeta"].values

    ax1.errorbar(alphas, lambdas, yerr=dlambdas, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3])
    ax1.errorbar(alphas_pcut, lambdas_pcut, yerr=dlambdas_pcut, color=qpt_pcut[0], fmt=qpt_pcut[1], ms=qpt_pcut[2], lw=qpt_pcut[3])
    ax1.errorbar(alphas2, lambdas2, xerr=dalphas2, color=qpt2[0], fmt=qpt2[1], ms=qpt2[2], lw=qpt2[3])
    print(alphas, lambdas)
    ax1.set_xlabel('$\\alpha$', fontsize=fs)
    ax1.set_ylabel('$\\lambda_c$', fontsize=fs)
    ax1.set_xlim([1., 3.])
    ax1.set_ylim([0., 0.6])
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.text(0.05, 0.8, "$U(1)$ CSB", fontsize=fs2, color='black', transform=ax1.transAxes)
    ax1.text(0.75, 0.2, "large D phase", fontsize=fs2, color='black', transform=ax1.transAxes)

    ax2.errorbar(alphas, nus, yerr=dnus, color=exp[0], fmt=exp[1], ms=exp[2])
    ax2.errorbar(alphas_bias, nus_bias, yerr=dnus_bias, color=exp_bias[0], fmt=exp_bias[1], ms=exp_bias[2])
    ax2.errorbar(alphas2, nus2, xerr=dalphas2, yerr=dnus2, color=exp2[0], fmt=exp2[1], ms=exp2[2])
    #ax2.errorbar(alphas2, lambdas2*np.exp(nus2), xerr=dalphas2, yerr=dnus2, color=exp2[0], fmt=exp2[1], ms=exp2[2])
    ax2.set_xlabel('$\\alpha$', fontsize=fs)
    ax2.set_ylabel('$\\nu$', fontsize=fs)
    ax2.set_xlim([1., 3.])
    ax2.set_ylim([0.5, 3.0])
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.25))

    ax3.errorbar(alphas, betas, yerr=dbetas, color=exp[0], fmt=exp[1], ms=exp[2], label=exp[3])
    ax3.errorbar(alphas_bias, betas_bias, yerr=dbetas_bias, color=exp_bias[0], fmt=exp_bias[1], ms=exp_bias[2], label=exp_bias[3])
    ax3.errorbar(alphas2, betas2, xerr=dalphas2, yerr=dbetas2, color=exp2[0], fmt=exp2[1], ms=exp2[2], label=exp2[3])
    ax3.set_xlabel('$\\alpha$', fontsize=fs)
    ax3.set_ylabel('$\\beta$', fontsize=fs)
    ax3.set_xlim([1., 3.])
    ax3.set_ylim([0.0, 1.5])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax3.xaxis.set_major_locator(MultipleLocator(0.5))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax3.yaxis.set_major_locator(MultipleLocator(0.25))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.125))

    ax4.errorbar(alphas_pcut, znus, yerr=dznus, color=exp_pcut[0], fmt=exp_pcut[1], ms=exp_pcut[2], label=exp_pcut[3])
    ax4.set_xlabel('$\\alpha$', fontsize=fs)
    ax4.set_ylabel('$z\\nu$', fontsize=fs)
    ax4.set_xlim([1., 3.])
    ax4.set_ylim([0.0, 2.0])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax4.xaxis.set_major_locator(MultipleLocator(0.5))
    ax4.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.25))

    ax5.errorbar(alphas_pcut, exps+znus, yerr=np.sqrt(dexps**2+dznus**2), color=exp_pcut[0], fmt=exp_pcut[1], ms=exp_pcut[2])
    ax5.set_xlabel('$\\alpha$', fontsize=fs)
    ax5.set_ylabel('$\\gamma$', fontsize=fs)
    ax5.set_xlim([1., 3.])
    ax5.set_ylim([0.5, 3.5])
    # 5x3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax5.xaxis.set_major_locator(MultipleLocator(0.5))
    ax5.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax5.yaxis.set_major_locator(MultipleLocator(0.5))
    ax5.yaxis.set_minor_locator(MultipleLocator(0.25))

    #xnu, ynu = [3.0, 1000000], [1.0, 1.0]
    #ax2.plot(xnu, ynu, c='gray', linewidth=2, zorder=-1)
    #ax2b.plot(xnu, ynu, c='gray', linewidth=2, zorder=-1)

    #xbeta, ybeta = [3.0, 1000000], [0.125, 0.125]
    #ax3.plot(xbeta, ybeta, c='gray', linewidth=2, zorder=-1)
    #ax3b.plot(xbeta, ybeta, c='gray', linewidth=2, zorder=-1)

    #xmu, ymu = [3.0, 1000000], [2.0, 2.0]
    #ax4.plot(xmu, ymu, c='gray', linewidth=2, zorder=-1)
    #ax4b.plot(xmu, ymu, c='gray', linewidth=2, zorder=-1)

    # plt.tight_layout()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5,0.085), ncols=2, handletextpad=0.2, fontsize=12)
    #fig.legend(loc='lower right',ncol=1, handletextpad=-0.2, columnspacing=0.5, fontsize=fs2)
    fig.savefig("../plots/fss/u1_csb/crit_u1csb_alpha.pdf")
    plt.show()

if __name__ == '__main__':
    plot()



