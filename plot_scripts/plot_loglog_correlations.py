import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from matplotlib import use, rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 400  # system size

# output filename
output_file = f"../plots/correlations_L{L}.pdf"

# directory and filename
data_dir = f'../data/correlations/L{L}/'


filename_largeD = f'spinone_heisenberg_correlations_chi300_D1.4_Gamma1.0_Jz1.0_alpha10.0_L{L}.csv'
filename_af = f'spinone_heisenberg_correlations_chi300_D-0.6_Gamma1.0_Jz1.0_alpha4.0_L{L}.csv'
filename_haldane = f'spinone_heisenberg_correlations_chi300_D0.0_Gamma1.0_Jz1.0_alpha10.0_L{L}.csv'
#filename_su2csb = f'spinone_heisenberg_correlations_chi300_D0.0_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'
#filename_u1csb = f'spinone_heisenberg_correlations_chi300_D1.0_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'
#filename_unkown = f"spinone_heisenberg_correlations_chi300_D0.1_Gamma1.0_Jz1.0_alpha2.8_L{L}.csv"
#filename_unkown = f"spinone_heisenberg_correlations_chi300_D0.3_Gamma1.0_Jz1.0_alpha3.2_L{L}.csv"
filename_unkown = f"spinone_heisenberg_correlations_chi300_D1.0_Gamma1.0_Jz1.0_alpha2.5_L{L}.csv"
#filename_other = f'spinone_heisenberg_correlations_D-0.5_alpha1.5_L{L}.csv'


filenames = [filename_largeD, filename_haldane, filename_unkown]
#labels = ["large D $(D=1.4,~\\alpha=10.0)$", "Haldane $(D=0.0,~\\alpha=10.0)$", "unkown $(D=1.0,~\\alpha=3.0)$"]
#markers = ["x", "s", "."]

#filenames = [filename_haldane]
#labels = ["Haldane $(D=0.0,~\\alpha=10.0)$"]
#markers = ["s"]

filenames = [filename_largeD, filename_af, filename_haldane, filename_unkown]
labels = ["large D $(D=1.4,~\\alpha=10.0)$", "z-AF $(D=-0.3,~\\alpha=1.5)$", "Haldane $(D=0.0,~\\alpha=10.0)$", "unkown $(D=1.0,~\\alpha=2.5)$"]
colors = ['C3', 'C0', 'C1', 'C2']

fs1 = 18
fs2 = 13

# Create the figure and the subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

# Set labels and titles for each subplot (optional)
for idx, ax in enumerate(axs):
    for filename, label, color in zip(filenames, labels, colors):
        #ax.grid(True)
        file = os.path.join(data_dir, filename)
        data = pd.read_csv(file)

        #ax.text(0.98, 0.87, label, transform=ax.transAxes, ha='right', fontsize=fs2)
        #ax.set_ylim(-0.5,0.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))

        pos = data['pos'].values + np.ones(len(data['pos'].values))
        corrxx = 0.5*data['corr_pm'].values
        corryy = 0.5*data['corr_mp'].values
        corrzz = data['corr_zz'].values
        corrstr = data['corr_str_order'].values

        #label = r'$\vert\langle S_1^zS_j^z \rangle\vert$',



        #ax.plot(pos, np.abs(corrstr), label=r'$\vert O^z_{1,j}\vert$', marker=marker, color=colors[0])
        #ax.plot(pos, np.abs(corrxx), label=r'$\vert\langle S_1^xS_j^x \rangle\vert$', marker=marker, color=colors[1])
        #ax.plot(pos, np.abs(corryy), label=r'$\vert\langle S_1^yS_j^y \rangle\vert$', marker=markers[2], color=color)
        if idx == 0:
            ax.plot(pos, np.abs(corrxx), label=label, marker=".", color=color)
            ax.set_ylabel("$\\vert\\langle S_1^xS_j^x \\rangle\\vert$", fontsize=fs2)
        else:
            ax.plot(pos, np.abs(corrzz), marker=".", color=color)
            ax.set_ylabel("$\\vert\\langle S_1^zS_j^z \\rangle\\vert$", fontsize=fs2)



        #if idx == 0:
        #    #ax.plot(pos, np.abs(corrstr), label=r'$\vert O^z_{1,j}\vert$', marker=marker, color=colors[0])
        #    ax.plot(pos, np.abs(corrxx), label=r'$\vert\langle S_1^xS_j^x \rangle\vert$', marker=marker, color=colors[1])
        #    #ax.plot(pos, np.abs(corryy), label=r'$\vert\langle S_1^yS_j^y \rangle\vert$', marker=marker, color=colors[2])
        #    ax.plot(pos, np.abs(corrzz), label=r'$\vert\langle S_1^zS_j^z \rangle\vert$', marker=marker, color=colors[3])
        #else:
        #    #ax.plot(pos, np.abs(corrstr), marker=marker, color=colors[0])
        #    ax.plot(pos, np.abs(corrxx), marker=marker, color=colors[1])
        #    #ax.plot(pos, np.abs(corryy), marker=marker, color=colors[2])
        #    ax.plot(pos, np.abs(corrzz), marker=marker, color=colors[3])

        ax.set_xscale("log")
        ax.set_yscale("log")


    # Add a common legend on top
    #fig.legend(loc=(0.15,0.55), ncol=1, fontsize=fs2)
    fig.legend(loc=(0.15,0.15), ncol=1, fontsize=fs2)


    axs[0].set_title(f"$L={L}$", fontsize=fs1)

    # Ensure that subplots are tightly packed vertically
    plt.subplots_adjust(hspace=0.1, top=0.9)

    # Align the x-axis of all subplots
    axs[-1].set_xlabel("site $j$", fontsize=fs2)

    #axs[0].set_ylabel("site $j$")

    # Remove x-ticks for all but the bottom subplot
    #for ax in axs[:-1]:
    #    ax.label_outer()

#plt.tight_layout()

## save figure
plt.savefig(output_file)

# Show the plot
plt.show()


