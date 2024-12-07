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
    fig = plt.figure(figsize=(5.51, 8.51))
    # matplotlib.figure.SubplotParams(left=0.0,right=1.0,bottom=0.5,top=1.0)

    gs = GridSpec(5, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax1b = fig.add_subplot(gs[2, 1])
    ax2b = fig.add_subplot(gs[3, 0])
    ax3b = fig.add_subplot(gs[3, 1])
    ax4b = fig.add_subplot(gs[4, 0])

    # ax1.set_position([0.11,0.65,0.85,0.3],which='both')
    #ax1.set_position([0.11, 0.65, 0.82, 0.3], which='both')
    # ax1.spines['right'].set_visible(False)

    # ax2.set_position([0.11,0.20,0.32,0.35],which='both')
    #ax2.set_position([0.11, 0.20, 0.32, 0.35], which='both')

    # ax3.set_position([0.64,0.20,0.32,0.35],which='both')
    #ax3.set_position([0.61, 0.20, 0.32, 0.35], which='both')

    # ax1.set_position([0.11,0.65,0.85,0.3],which='both')
    ax1.set_position([0.11, 0.73, 0.82, 0.25], which='both')
    ax1b.set_position([0.94, 0.73, 0.02, 0.25], which='both')
    ax1b.set_yticks([])
    ax1b.set_xticks([])
    # ax1b.set_xticklabels([1000000],labels=['$\\infty$'])
    # ax1b.set_xticklabels([1000000])
    # ax1b.set_xticklabels([999999,'$\\infty$'])
    ax1b.text(0., -0.09, '$\\infty$', transform=ax1b.transAxes)
    ax1.spines['right'].set_visible(False)
    ax1b.spines['left'].set_visible(False)

    kwa = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1)
    d = .005  # how big to make the diagonal lines in axes coordinatess
    s = 5
    ax1.plot((1 + d, 1 - d), (+d * s, -d * s), **kwa)
    ax1.plot((1 + d, 1 - d), (1 + d * s, 1 - d * s), **kwa)

    kwa = dict(transform=ax1b.transAxes, color='k', clip_on=False, lw=1)
    d = 0.215
    s = 0.125
    ax1b.plot((+d, -d), (+d * s, -d * s), **kwa)
    ax1b.plot((+d, -d), (1 + d * s, 1 - d * s), **kwa)

    # ax2.set_position([0.11,0.20,0.32,0.35],which='both')
    ax2.set_position([0.11, 0.40, 0.32, 0.25], which='both')
    ax2b.set_position([0.44, 0.40, 0.02, 0.25], which='both')
    ax2b.set_yticks([])
    ax2b.set_xticks([])
    ax2b.text(0., -0.09, '$\\infty$', transform=ax2b.transAxes)
    ax2.spines['right'].set_visible(False)
    ax2b.spines['left'].set_visible(False)

    kwa = dict(transform=ax2.transAxes, color='k', clip_on=False, lw=1)
    d = 0.015
    s = 1.8
    ax2.plot((1 + d, 1 - d), (+d * s, -d * s), **kwa)
    ax2.plot((1 + d, 1 - d), (1 + d * s, 1 - d * s), **kwa)

    kwa = dict(transform=ax2b.transAxes, color='k', clip_on=False, lw=1)
    d = 0.215
    s = 0.125
    ax2b.plot((+d, -d), (+d * s, -d * s), **kwa)
    ax2b.plot((+d, -d), (1 + d * s, 1 - d * s), **kwa)

    # ax3.set_position([0.64,0.20,0.32,0.35],which='both')
    ax3.set_position([0.61, 0.40, 0.32, 0.25], which='both')
    ax3b.set_position([0.94, 0.40, 0.02, 0.25], which='both')
    ax3b.set_yticks([])
    ax3b.set_xticks([])
    ax3b.text(0., -0.09, '$\\infty$', transform=ax3b.transAxes)
    ax3.spines['right'].set_visible(False)
    ax3b.spines['left'].set_visible(False)

    kwa = dict(transform=ax3.transAxes, color='k', clip_on=False, lw=1)
    d = 0.015
    s = 1.8
    ax3.plot((1 + d, 1 - d), (+d * s, -d * s), **kwa)
    ax3.plot((1 + d, 1 - d), (1 + d * s, 1 - d * s), **kwa)

    kwa = dict(transform=ax3b.transAxes, color='k', clip_on=False, lw=1)
    d = 0.215
    s = 0.125
    ax3b.plot((+d, -d), (+d * s, -d * s), **kwa)
    ax3b.plot((+d, -d), (1 + d * s, 1 - d * s), **kwa)

    # ax3.set_position([0.64,0.20,0.32,0.35],which='both')
    ax4.set_position([0.11, 0.07, 0.32, 0.25], which='both')
    ax4b.set_position([0.44, 0.07, 0.02, 0.25], which='both')
    ax4b.set_yticks([])
    ax4b.set_xticks([])
    ax4b.text(0., -0.09, '$\\infty$', transform=ax4b.transAxes)
    ax4.spines['right'].set_visible(False)
    ax4b.spines['left'].set_visible(False)

    kwa = dict(transform=ax4.transAxes, color='k', clip_on=False, lw=1)
    d = 0.015
    s = 1.8
    ax4.plot((1 + d, 1 - d), (+d * s, -d * s), **kwa)
    ax4.plot((1 + d, 1 - d), (1 + d * s, 1 - d * s), **kwa)

    kwa = dict(transform=ax4b.transAxes, color='k', clip_on=False, lw=1)
    d = 0.215
    s = 0.125
    ax4b.plot((+d, -d), (+d * s, -d * s), **kwa)
    ax4b.plot((+d, -d), (1 + d * s, 1 - d * s), **kwa)

    qpt = ("#3a86ff", '.-', 9, 1, "data collapse - $M_z$")
    qpt2 = ("#ff006e", '.-', 9, 1, "data collapse - $\\chi_{\\rm fidelity}$")
    exp = ("#3a86ff", '.', 11, "data collapse - $M_z$")
    exp2 = ("#ff006e", '.', 11, "data collapse - $\\chi_{\\rm fidelity}$")

    file = "../data/fss/ising_transition/data_collapse_mag.csv"
    data = pd.read_csv(file)
    alphas = data["alpha"].values
    alphas[np.isinf(alphas)] = 999999
    lambdas = data["lambda"].values
    dlambdas = data["dlambda"].values
    nus = data["nu"].values
    dnus = data["dnu"].values
    betas = data["beta"].values
    dbetas = data["dbeta"].values


    file = "../data/fss/ising_transition/data_collapse_fidelity.csv"
    data = pd.read_csv(file)
    alphas2 = data["alpha"].values
    alphas2[np.isinf(alphas2)] = 999999
    lambdas2 = data["lambda"].values
    dlambdas2 = data["dlambda"].values
    nus2 = data["nu"].values
    dnus2 = data["dnu"].values
    mus2 = data["mu"].values
    dmus2 = data["dmu"].values

    ax1.errorbar(alphas, lambdas, yerr=dlambdas, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3])
    ax1.errorbar(alphas2, lambdas2, yerr=dlambdas2, color=qpt2[0], fmt=qpt2[1], ms=qpt2[2], lw=qpt2[3])
    print(alphas, lambdas)
    ax1.set_xlabel('$\\alpha$', fontsize=fs)
    ax1.set_ylabel('$\\lambda_c$', fontsize=fs)
    ax1.set_xlim([3., 11.])
    ax1.set_ylim([0., -0.4])
    ax1.xaxis.set_major_locator(MultipleLocator(2.0))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax1b.errorbar(alphas[:2], lambdas[:2], yerr=dlambdas[:2], color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3])
    ax1b.errorbar(alphas2[:2], lambdas2[:2], yerr=dlambdas2[:2], color=qpt2[0], fmt=qpt2[1], ms=qpt2[2], lw=qpt2[3])
    ax1b.set_xlim([999998, 1000000])
    ax1b.set_ylim([0., -0.4])

    ax1.text(0.05, 0.8, "$z$-AF phase", fontsize=fs2, color='black', transform=ax1.transAxes)
    ax1.text(0.75, 0.2, "Haldane phase", fontsize=fs2, color='black', transform=ax1.transAxes)

    ax2.errorbar(alphas, nus, yerr=dnus, color=exp[0], fmt=exp[1], ms=exp[2], label=exp[3])
    ax2.errorbar(alphas2, nus2, yerr=dnus2, color=exp2[0], fmt=exp2[1], ms=exp2[2], label=exp2[3])
    ax2.set_xlabel('$\\alpha$', fontsize=fs)
    ax2.set_ylabel('$\\nu$', fontsize=fs)
    ax2.set_xlim([3., 11.])
    ax2.set_ylim([0.5, 2.0])
    ax2.xaxis.set_major_locator(MultipleLocator(2.0))
    ax2.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.25))

    ax2b.errorbar(alphas[:1], nus[:1], yerr=dnus[:1], color=exp[0], fmt=exp[1], ms=exp[2])
    ax2b.errorbar(alphas2[:1], nus2[:1], yerr=dnus2[:1], color=exp2[0], fmt=exp2[1], ms=exp2[2])
    ax2b.set_xlim([999998, 1000000])
    ax2b.set_ylim([0.5, 2.0])

    ax3.errorbar(alphas, betas, yerr=dbetas, color=exp[0], fmt=exp[1], ms=exp[2])
    ax3.set_xlabel('$\\alpha$', fontsize=fs)
    ax3.set_ylabel('$\\beta$', fontsize=fs)
    ax3.set_xlim([3., 11.])
    ax3.set_ylim([0.0, 0.5])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax3.xaxis.set_major_locator(MultipleLocator(2.0))
    ax3.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax3.yaxis.set_major_locator(MultipleLocator(0.25))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.125))

    ax3b.errorbar(alphas[:1], betas[:1], yerr=dbetas[:1], color=exp[0], fmt=exp[1], ms=exp[2])
    ax3b.set_xlim([999998, 1000000])
    ax3b.set_ylim([0.0, 0.5])

    ax4.errorbar(alphas2, mus2, yerr=dmus2, color=exp2[0], fmt=exp2[1], ms=exp2[2])
    ax4.set_xlabel('$\\alpha$', fontsize=fs)
    ax4.set_ylabel('$\\mu$', fontsize=fs)
    ax4.set_xlim([3., 11.])
    ax4.set_ylim([1.0, 3.0])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax4.xaxis.set_major_locator(MultipleLocator(2.0))
    ax4.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.25))

    print(mus2[:1])
    print(alphas2)
    ax4b.errorbar(alphas2[:1], mus2[:1], yerr=dmus2[:1], color=exp2[0], fmt=exp2[1], ms=exp2[2])
    ax4b.set_xlim([999998, 1000000])
    ax4b.set_ylim([1.0, 3.0])

    xnu, ynu = [3.0, 1000000], [1.0, 1.0]
    ax2.plot(xnu, ynu, c='gray', linewidth=2, zorder=-1)
    ax2b.plot(xnu, ynu, c='gray', linewidth=2, zorder=-1)

    xbeta, ybeta = [3.0, 1000000], [0.125, 0.125]
    ax3.plot(xbeta, ybeta, c='gray', linewidth=2, zorder=-1)
    ax3b.plot(xbeta, ybeta, c='gray', linewidth=2, zorder=-1)

    xmu, ymu = [3.0, 1000000], [2.0, 2.0]
    ax4.plot(xmu, ymu, c='gray', linewidth=2, zorder=-1)
    ax4b.plot(xmu, ymu, c='gray', linewidth=2, zorder=-1)

    # plt.tight_layout()
    fig.legend(loc='lower right', bbox_to_anchor=(0.98,0.15), handletextpad=0.2, fontsize=14)
    #fig.legend(loc='lower right',ncol=1, handletextpad=-0.2, columnspacing=0.5, fontsize=fs2)
    fig.savefig("../plots/crit_ising_alpha.pdf")
    plt.show()

if __name__ == '__main__':
    plot()


