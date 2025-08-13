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
    fig = plt.figure(figsize=(6.51, 7))
    # matplotlib.figure.SubplotParams(left=0.0,right=1.0,bottom=0.5,top=1.0)

    gs = GridSpec(4, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    #ax4 = fig.add_subplot(gs[2, 0])
    #ax5 = fig.add_subplot(gs[2, 1])

    ax1.set_position([0.11, 0.65, 0.85, 0.3], which='both')
    ax2.set_position([0.11, 0.2, 0.35, 0.3], which='both')
    ax3.set_position([0.61, 0.2, 0.35, 0.3], which='both')
    #ax3.set_position([0.11, 0.14, 0.35, 0.23], which='both')
    #ax5.set_position([0.61, 0.14, 0.35, 0.23], which='both')

    qpt = ("#3a86ff", 's-', 5, 1, "DMRG - data collapse in D")
    exp = ("#3a86ff", 's', 6, "DMRG - data collapse in D")

    file = "../data/fss/ising_transition/data_collapse_D_mag.csv"
    data = pd.read_csv(file)
    alphas = data["alpha"].values
    alphas[np.isinf(alphas)] = 999999
    Ds = data["D"].values
    dDs = data["dD"].values
    nus = data["nu"].values
    dnus = data["dnu"].values
    betas = data["beta"].values
    dbetas = data["dbeta"].values

    ax1.errorbar(Ds, np.reciprocal(alphas), xerr=dDs, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3])
    ax1.set_ylabel('$1/\\alpha$', fontsize=fs)
    ax1.set_xlabel('$D_c$', fontsize=fs)
    ax1.set_ylim([0.0, 0.36])
    ax1.set_xlim([-0.4, 0.0])
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax1.text(0.05, 0.8, "z-AF phase", fontsize=fs2, color='black', transform=ax1.transAxes)
    ax1.text(0.75, 0.2, "Haldane phase", fontsize=fs2, color='black', transform=ax1.transAxes)

    ax2.errorbar(np.reciprocal(alphas), nus, yerr=dnus, color=exp[0], fmt=exp[1], ms=exp[2])
    #ax2.errorbar(alphas2, lambdas2*np.exp(nus2), xerr=dalphas2, yerr=dnus2, color=exp2[0], fmt=exp2[1], ms=exp2[2])
    ax2.set_xlabel('$1/\\alpha$', fontsize=fs)
    ax2.set_ylabel('$\\nu$', fontsize=fs)
    ax2.set_xlim([0.0, 0.36])
    ax2.set_ylim([0.5, 1.5])
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax3.errorbar(np.reciprocal(alphas), betas, yerr=dbetas, color=exp[0], fmt=exp[1], ms=exp[2], label=exp[3])
    ax3.set_xlabel('$1/\\alpha$', fontsize=fs)
    ax3.set_ylabel('$\\beta$', fontsize=fs)
    ax3.set_xlim([0.0, 0.36])
    ax3.set_ylim([0.0, 0.3])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax3.xaxis.set_major_locator(MultipleLocator(0.1))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax3.yaxis.set_major_locator(MultipleLocator(0.1))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.05))


    xnu, ynu = [0.0, 0.36], [1.0, 1.0]
    ax2.plot(xnu, ynu, c='gray', linewidth=2, zorder=-1)

    xbeta, ybeta = [0.0, 0.36], [0.125, 0.125]
    ax3.plot(xbeta, ybeta, c='gray', linewidth=2, zorder=-1)
    #ax2b.plot(xnu, ynu, c='gray', linewidth=2, zorder=-1)

    # plt.tight_layout()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5,0.085), ncols=2, handletextpad=0.2, fontsize=12)
    #fig.legend(loc='lower right',ncol=1, handletextpad=-0.2, columnspacing=0.5, fontsize=fs2)
    fig.savefig("../plots/fss/ising/crit_ising_alpha.pdf")
    plt.show()

if __name__ == '__main__':
    plot()



