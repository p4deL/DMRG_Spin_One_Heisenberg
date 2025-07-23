import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from matplotlib import use, rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}\usepackage{xcolor}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}\usepackage{xcolor}"



L = 100  # system size

# output filename
output_file = f"../plots/correlations_L{L}.pdf"

# directory and filename
data_dir = f'../data/correlations/L{L}/'


filename_largeD = f'spinone_heisenberg_correlations_chi300_D1.4_Gamma1.0_Jz1.0_alpha10.0_L{L}.csv'
filename_af = f'spinone_heisenberg_correlations_chi300_D-0.3_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'
filename_haldane = f'spinone_heisenberg_correlations_chi300_D0.0_Gamma1.0_Jz1.0_alpha10.0_L{L}.csv'
filename_su2csb = f'spinone_heisenberg_correlations_chi300_D0.0_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'
filename_u1csb = f'spinone_heisenberg_correlations_chi300_D1.0_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'
filename_unkown = f"spinone_heisenberg_correlations_chi300_D0.1_Gamma1.0_Jz1.0_alpha2.8_L{L}.csv"
#filename_unkown = f"spinone_heisenberg_correlations_chi300_D0.3_Gamma1.0_Jz1.0_alpha3.2_L{L}.csv"
#filename_unkown = f"spinone_heisenberg_correlations_chi300_D1.0_Gamma1.0_Jz1.0_alpha3.0_L{L}.csv"
#filename_other = f'spinone_heisenberg_correlations_D-0.5_alpha1.5_L{L}.csv'


filenames = [filename_largeD, filename_af, filename_haldane, filename_su2csb, filename_u1csb, filename_unkown] #, filename_other]
labels = ["large D $(D=1.4,~\\alpha=10.0)$", "  z-AF $(D=-0.3,~\\alpha=1.5)$", "  Haldane $(D=0.0,~\\alpha=10.0)$", "SU(2) CSB $(D=0.0,~\\alpha=1.5)$", "  U(1) CSB $(D=1.0,~\\alpha=1.5)$", "  critical $(D=0.1,~\\alpha=2.8)$"]
colors = ['#E9EB9E', '#ACC196', '#93AB96', '#799496', '#49475B', '#14080E']


corrstr_info = [r'$O^z_{1,j}$', "s", 4, '#e07a5f']
corrxx_info = [r'$\langle S_1^xS_j^x \rangle$', "X", 6, '#457b9d']
corryy_info = [r'$\langle S_1^yS_j^y \rangle$', "+", 7, '#f2cc8f']
corrzz_info = [r'$\langle S_1^zS_j^z \rangle$', ".", 8, '#74c69d']


fs1 = 15
fs2 = 13

# Create the figure and the subplots
fig, axs = plt.subplots(6, 1, figsize=(6, 9), sharex=True)

# Set labels and titles for each subplot (optional)
for idx, (ax, filename, label, color) in enumerate(zip(axs, filenames, labels, colors)):
    file = os.path.join(data_dir, filename)
    data = pd.read_csv(file)

    ax.text(0.96, 0.8, label + "\\phantom{a}s", transform=ax.transAxes, ha='right', fontsize=fs2, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round'))
    ax.scatter(99.5, 0.77, s=70, marker='o', color=color, zorder=10)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_ylim(-1.1,1.1)
    ax.set_yticks(np.arange(-1.0, 1.5, 0.5))

    pos = data['pos'].values + np.ones(len(data['pos'].values))
    corrxx = 0.5*data['corr_pm'].values
    corryy = 0.5*data['corr_mp'].values
    corrzz = data['corr_zz'].values
    corrstr = data['corr_str_order'].values


    if idx == 0:
        ax.plot(pos, corrstr, label=corrstr_info[0], marker=corrstr_info[1], ms=corrstr_info[2], color=corrstr_info[3])
        ax.plot(pos, corrxx, label=corrxx_info[0], marker=corrxx_info[1], ms=corrxx_info[2], color=corrxx_info[3])
        ax.plot(pos, corryy, label=corryy_info[0], marker=corryy_info[1], ms=corryy_info[2], color=corryy_info[3])
        ax.plot(pos, corrzz, label=corrzz_info[0], marker=corrzz_info[1], ms=corrzz_info[2], color=corrzz_info[3])
    else:
        ax.plot(pos, corrstr, marker=corrstr_info[1], ms=corrstr_info[2], color=corrstr_info[3])
        ax.plot(pos, corrxx,  marker=corrxx_info[1], ms=corrxx_info[2], color=corrxx_info[3])
        ax.plot(pos, corryy, marker=corryy_info[1], ms=corryy_info[2], color=corryy_info[3])
        ax.plot(pos, corrzz, marker=corrzz_info[1], ms=corrzz_info[2], color=corrzz_info[3])


#axs[0].set_yticks(np.arange(-1.0, 1.5, 0.5 ))
#axs[1].set_ylim(-1.35, 1.35)
#axs[1].set_yticks(np.arange(-1.0, 1.5, 0.5 ))

# Add a common legend on top
lines_labels = [ax.get_legend_handles_labels() for ax in axs]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper center', ncol=4, fontsize='large')

# Ensure that subplots are tightly packed vertically
plt.subplots_adjust(hspace=0.2, top=0.93)

# Align the x-axis of all subplots
axs[-1].set_xlabel("site $j$", fontsize=fs1)

#axs[0].set_ylabel("site $j$")

# Remove x-ticks for all but the bottom subplot
for ax in axs[:-1]:
    ax.label_outer()

## save figure
plt.savefig(output_file)

# Show the plot
plt.show()


