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
    fig = plt.figure(figsize=(6.51, 6.81))
    # matplotlib.figure.SubplotParams(left=0.0,right=1.0,bottom=0.5,top=1.0)

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.set_position([0.11, 0.62, 0.35, 0.34], which='both')
    ax2.set_position([0.61, 0.62, 0.35, 0.34], which='both')
    ax3.set_position([0.11, 0.18, 0.35, 0.34], which='both')
    ax4.set_position([0.61, 0.18, 0.35, 0.34], which='both')

    exp = ("#457b9d", 's', 6, 0.75, "DMRG -- data collapse in $\\lambda$")
    exp_bias = ("#25a18e", '*', 10, 0.75, "DMRG biased -- data collapse in $\\lambda$")
    exp_pcut = ("#81b29a", '.', 11, 0.75, "pCUT -- DlogPadé")
    exp2 = ("#BD93D8", 'X', 7, 0.75, "DMRG -- data collapse in $\\alpha$")

    file = "../data/fss/largeD_U(1)CSB_transition/data_collapse_lambda_mag.csv"
    data = pd.read_csv(file)
    alphas = data["alpha"].values
    alphas[np.isinf(alphas)] = 999999
    nus = data["nu"].values
    dnus = data["dnu"].values
    betas = data["beta"].values
    dbetas = data["dbeta"].values


    file = "../data/fss/largeD_U(1)CSB_transition/pcut_1qp_gap.csv"
    data = pd.read_csv(file)
    alphas_pcut = data["alpha"].values
    alphas_pcut[np.isinf(alphas_pcut)] = 999999
    znus = data["exp"].values
    dznus = data["dexp"].values

    file = "../data/fss/largeD_U(1)CSB_transition/pcut_1qp_sw.csv"
    data = pd.read_csv(file)
    #alphas_pcut = data["alpha"].values
    #alphas_pcut[np.isinf(alphas_pcut)] = 999999
    gammas = data["exp"].values + znus
    dgammas = data["dexp"].values + dznus

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
    nus2 = data["nu"].values
    dnus2 = data["dnu"].values
    betas2 = data["beta"].values
    dbetas2 = data["dbeta"].values

    ax1.errorbar(alphas, nus, yerr=dnus, color=exp[0], fmt=exp[1], ms=exp[2], alpha=exp[3])
    ax1.errorbar(alphas_bias, nus_bias, yerr=dnus_bias, color=exp_bias[0], fmt=exp_bias[1], ms=exp_bias[2], alpha=exp_bias[3])
    #ax1.errorbar(alphas2, nus2, xerr=dalphas2, yerr=dnus2, color=exp2[0], fmt=exp2[1], ms=exp2[2], alpha=exp2[3])
    ax1.set_xlabel('$\\alpha$', fontsize=fs)
    ax1.set_ylabel('$\\nu$', fontsize=fs)
    ax1.set_xlim([1., 3.])
    ax1.set_ylim([0.5, 3.0])
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.25))


    # ADD COLORGRASIENT FOR LRMF regime
    # Define the x-range for the gradient
    x_start = 1
    x_end = 1.67


    # Create a gradient image (1D for x-direction)
    colors = np.zeros((1, 100, 4))  # RGBA array
    cmap = plt.get_cmap('Blues_r')  # Use reversed Blues colormap
    for i in range(100):
        colors[0, i] = cmap(i / 100)  # Color from light to dark blue (reversed)
        colors[0, i, 3] = 0.25  # Constant alpha of 0.75

    # Add the gradient as an image
    ax1.imshow(colors, extent=[x_start, x_end, 0.0, 3.5], aspect='auto', zorder=0)
    ax2.imshow(colors, extent=[x_start, x_end, 0.0, 3.5], aspect='auto', zorder=0)
    ax3.imshow(colors, extent=[x_start, x_end, 0.0, 3.5], aspect='auto', zorder=0)
    ax4.imshow(colors, extent=[x_start, x_end, 0.0, 3.5], aspect='auto', zorder=0)

    x = np.linspace(x_start, x_end, 100)  # 100 points for smooth gradient
    y = 1 / (x-1)  # Compute 1/x values

    # Create a gray colormap for the gradient
    cmap = plt.get_cmap('Greys_r')  # Gray colormap (light to dark)

    # Plot the line as segments with a color gradient
    for i in range(len(x) - 1):
        ax1.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=cmap(i / (len(x) - 1)), alpha=0.5, zorder=1)
        ax2.plot([x[i], x[i + 1]], [0.5,0.5], color=cmap(i / (len(x) - 1)), alpha=0.2, zorder=1)
        ax3.plot([x[i], x[i + 1]], [0.5,0.5], color=cmap(i / (len(x) - 1)), alpha=0.2, zorder=1)
        ax4.plot([x[i], x[i + 1]], [1.0,1.0], color=cmap(i / (len(x) - 1)), alpha=0.2, zorder=1)


    # Ensure the gradient doesn't interfere with other plot elements
    #ax2.set_xlim(ax2.get_xlim())  # Preserve original x-limits
    #ax2.set_ylim(y)  # Restore original y-limits

    ax2.errorbar(alphas, betas, yerr=dbetas, color=exp[0], fmt=exp[1], ms=exp[2], alpha=exp[3], label=exp[4])
    ax2.errorbar(alphas_bias, betas_bias, yerr=dbetas_bias, color=exp_bias[0], fmt=exp_bias[1], ms=exp_bias[2], alpha=exp_bias[3], label=exp_bias[4])
    ax2.errorbar(alphas2, betas2, xerr=dalphas2, yerr=dbetas2, color=exp2[0], fmt=exp2[1], ms=exp2[2], alpha=exp2[3], label=exp2[4])
    ax2.set_xlabel('$\\alpha$', fontsize=fs)
    ax2.set_ylabel('$\\beta$', fontsize=fs)
    ax2.set_xlim([1., 3.])
    ax2.set_ylim([0.0, 1.5])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax2.yaxis.set_major_locator(MultipleLocator(0.25))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.125))

    ax3.errorbar(alphas_pcut, znus, yerr=dznus, color=exp_pcut[0], fmt=exp_pcut[1], ms=exp_pcut[2], alpha=exp_pcut[3], label=exp_pcut[4])
    ax3.set_xlabel('$\\alpha$', fontsize=fs)
    ax3.set_ylabel('$z\\nu$', fontsize=fs)
    ax3.set_xlim([1., 3.])
    ax3.set_ylim([0.0, 2.0])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax3.xaxis.set_major_locator(MultipleLocator(0.5))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax3.yaxis.set_major_locator(MultipleLocator(0.5))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.25))

    ax4.errorbar(alphas_pcut, gammas, yerr=dgammas, color=exp_pcut[0], fmt=exp_pcut[1], ms=exp_pcut[2], alpha=exp_pcut[3])
    ax4.set_xlabel('$\\alpha$', fontsize=fs)
    ax4.set_ylabel('$\\gamma$', fontsize=fs)
    ax4.set_xlim([1., 3.])
    ax4.set_ylim([0.5, 3.5])
    # ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax4.xaxis.set_major_locator(MultipleLocator(0.5))
    ax4.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.25))

    # plt.tight_layout()
    fig.legend(loc='lower right', bbox_to_anchor=(0.96,0.00), ncols=2, handletextpad=0.2, fontsize=12)
    #fig.legend(loc='lower right',ncol=1, handletextpad=-0.2, columnspacing=0.5, fontsize=fs2)
    fig.savefig("../plots/paper/exponents_u1csb.pdf")
    plt.show()

if __name__ == '__main__':
    plot()


