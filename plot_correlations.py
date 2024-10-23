import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from matplotlib import use, rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 100  # system size

# output filename
output_file = f"correlations_L{L}.pdf"

# directory and filename
data_dir = 'output/correlations/'
filename_largeD = f'spinone_heisenberg_correlations_D1.5_alphaInf_L{L}.csv'
filename_af = f'spinone_heisenberg_correlations_D-1.5_alphaInf_L{L}.csv'
filename_haldane = f'spinone_heisenberg_correlations_D0.0_alphaInf_L{L}.csv'
filename_su2csb = f'spinone_heisenberg_correlations_D0.0_alpha2.0_L{L}.csv'
filename_u1csb = f'spinone_heisenberg_correlations_D0.5_alpha2.0_L{L}.csv'
#filename_other = f'spinone_heisenberg_correlations_D-0.5_alpha1.5_L{L}.csv'


filenames = [filename_largeD, filename_af, filename_haldane, filename_su2csb, filename_u1csb] #, filename_other]
labels = ["large D $(D=1.5,~\\alpha=\\infty)$", "z-AF $(D=-1.5,~\\alpha=\\infty)$", "Haldane $(D=0.0,~\\alpha=\\infty)$", "SU(2) CSB $(D=0.0,~\\alpha=2.0)$", "U(1) CSB $(D=0.5,~\\alpha=2.0)$"] #, "other? $(D=-0.5,~\\alpha=1.5)$"]
colors = ['C3', 'C0', 'C1', 'C2']

fs1 = 18
fs2 = 13


# Create the figure and the subplots
fig, axs = plt.subplots(5, 1, figsize=(6, 10), sharex=True)

# Set labels and titles for each subplot (optional)
for idx, (ax, filename, label) in enumerate(zip(axs, filenames, labels)):
    ax.grid(True)
    file = os.path.join(data_dir, filename)
    data = pd.read_csv(file)

    ax.text(0.98, 0.9, label, transform=ax.transAxes, ha='right', fontsize=fs2)
    ax.set_ylim(-1.2,1.2)

    pos = data['pos'].values
    corrxx = data['corrxx'].values
    corryy = data['corryy'].values
    corrzz = data['corrzz'].values
    corrstr = data['corrstr'].values

    if idx == 0:
        ax.plot(pos, corrstr, label=r'$O^z_{1,j}$', marker="s", color=colors[0])
        ax.plot(pos, corrxx, label=r'$\langle S_1^xS_j^x \rangle$', marker="x", color=colors[1])
        ax.plot(pos, corryy, label=r'$\langle S_1^yS_j^y \rangle$', marker="+", color=colors[2])
        ax.plot(pos, corrzz, label=r'$\langle S_1^zS_j^z \rangle$', marker=".", color=colors[3])
    else:
        ax.plot(pos, corrstr, marker="s", color=colors[0])
        ax.plot(pos, corrxx, marker="x", color=colors[1])
        ax.plot(pos, corryy, marker="+", color=colors[2])
        ax.plot(pos, corrzz, marker=".", color=colors[3])

# Add a common legend on top
lines_labels = [ax.get_legend_handles_labels() for ax in axs]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper center', ncol=4, fontsize='large')

# Ensure that subplots are tightly packed vertically
plt.subplots_adjust(hspace=0.1, top=0.9)

# Align the x-axis of all subplots
axs[4].set_xlabel("site $j$")

#axs[0].set_ylabel("site $j$")

# Remove x-ticks for all but the bottom subplot
for ax in axs[:-1]:
    ax.label_outer()

## save figure
plt.savefig(output_file)

# Show the plot
plt.show()


#
#
## Label axes
#plt.xlabel(r'$D$', fontsize=fs2)
#plt.ylabel(r'$\chi_{\rm fidelity}$', fontsize=fs2)
#
#plt.ylim(0,2500)
##plt.legend(fontsize=fs2)
#
## title
#plt.title(f"$L={L}$", fontsize=fs1)
#
## save figure
#plt.savefig(output_file)
#
## show plot
#plt.show()
