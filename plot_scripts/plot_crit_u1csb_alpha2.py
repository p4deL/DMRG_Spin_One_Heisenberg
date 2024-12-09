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

    fig = plt.figure(figsize = (5.51,5.51))
    #matplotlib.figure.SubplotParams(left=0.0,right=1.0,bottom=0.5,top=1.0)

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])


    #ax1.set_position([0.11,0.65,0.85,0.3],which='both')
    ax1.set_position([0.11,0.65,0.82,0.3],which='both')
    #ax1.spines['right'].set_visible(False)

    #ax2.set_position([0.11,0.20,0.32,0.35],which='both')
    ax2.set_position([0.11,0.20,0.32,0.35],which='both')

    #ax3.set_position([0.64,0.20,0.32,0.35],which='both')
    ax3.set_position([0.61,0.20,0.32,0.35],which='both')

    qpt = ("#3a86ff", '.-', 7, 1)
    exp = ("#3a86ff", '.', 7)

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_mag.csv"
    data = pd.read_csv(file)
    alphas = data["alpha"].values
    lambdas = data["lambda"].values
    dlambdas = data["dlambda"].values
    nus = data["nu"].values
    dnus = data["dnu"].values
    betas = data["beta"].values
    dbetas = data["dbeta"].values

    ax1.errorbar(alphas, lambdas, yerr=dlambdas, color=qpt[0], fmt=qpt[1], ms=qpt[2], lw=qpt[3])

    ax1.set_xlabel('$\\alpha$', fontsize=fs)
    ax1.set_ylabel('$\\lambda_c$', fontsize=fs)
    ax1.set_xlim([1., 3.])
    ax1.set_ylim([0.,0.6])
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    ax1.text(0.05, 0.8, "$U(1)$ CSB phase", fontsize=10, color='black', transform=ax1.transAxes)
    ax1.text(0.75, 0.2, "large-D phase", fontsize=10, color='black', transform=ax1.transAxes)

    ax2.errorbar(alphas, nus, yerr=dnus, color=exp[0], fmt=exp[1], ms=exp[2])
    ax2.set_xlabel('$\\alpha$', fontsize=fs)
    ax2.set_ylabel('$\\nu$', fontsize=fs)
    ax2.set_xlim([1.,3.])
    ax2.set_ylim([0.5,2.0])
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
    
    ax3.errorbar(alphas, betas, yerr=dbetas, color=exp[0], fmt=exp[1], ms=exp[2])
    ax3.set_xlabel('$\\alpha$', fontsize=fs)
    ax3.set_ylabel('$\\beta$', fontsize=fs)
    ax3.set_xlim([1., 3.])
    ax3.set_ylim([0.0,0.5])
    #ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax3.xaxis.set_major_locator(MultipleLocator(0.5))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax3.yaxis.set_major_locator(MultipleLocator(0.25))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.125))

    # plot lines
    #xmf, ymf = [0.0, 1.3], [0.5, 0.5]
    #ax2.plot(xmf, ymf, c='gray', linewidth=2)

    #ax2.axvspan(0.0, 1.333, facecolor='#00b4d8', alpha=0.2)
    #ax2.text(0.015, 0.5, "LRMF", fontsize=8, color='gray', transform=ax2.transAxes)
    #ax2.axvspan(1.96, 6.2, facecolor='#00b4d8', alpha=0.2)
    #ax2.text(0.6, 0.25, "NN", fontsize=8, color='gray', transform=ax2.transAxes)

    #xmf, ymf = [0.0, 1.3], [0.5, 0.5]
    #ax3.plot(xmf, ymf, c='gray', linewidth=2)

    #ax3.axvspan(0.0, 1.333, facecolor='#00b4d8', alpha=0.2)
    #ax3.text(0.015, 0.5, "LRMF", fontsize=8, color='gray', transform=ax3.transAxes)
    #ax3.axvspan(1.96, 6.2, facecolor='#00b4d8', alpha=0.2)
    #ax3.text(0.6, 0.25, "NN", fontsize=8, color='gray', transform=ax3.transAxes)


    #plt.tight_layout()
    #fig.legend(loc='lower center',ncol=5, handletextpad=-0.2, columnspacing=0.5, fontsize=fs2)
    fig.savefig("../plots/crit_larged_u1csb_alpha.pdf")
    plt.show()


if __name__ == '__main__':
    plot()


