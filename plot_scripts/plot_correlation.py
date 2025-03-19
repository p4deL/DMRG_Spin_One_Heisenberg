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

L = 100  # system size

# output filename
output_file = f"../plots/correlations_L{L}.pdf"

# directory and filename
data_dir = f'../data/correlations/L{L}/'

#filename = f'spinone_heisenberg_correlations_chi300_D1.4_Gamma1.0_Jz1.0_alpha10.0_L{L}.csv'
#filename = f'spinone_heisenberg_correlations_chi300_D-0.3_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'
#filename = f'spinone_heisenberg_correlations_chi300_D0.0_Gamma1.0_Jz1.0_alpha10.0_L{L}.csv'
filename = f'spinone_heisenberg_correlations_chi300_D0.0_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'
#filename = f'spinone_heisenberg_correlations_chi300_D1.0_Gamma1.0_Jz1.0_alpha1.5_L{L}.csv'



#labels = ["large D $(D=1.4,~\\alpha=10.0)$", "z-AF $(D=-0.3,~\\alpha=1.5)$", "Haldane $(D=0.0,~\\alpha=10.0)$", "SU(2) CSB $(D=0.0,~\\alpha=1.5)$", "U(1) CSB $(D=1.0,~\\alpha=1.5)$", "unkown $(D=1.0,~\\alpha=3.0)$"]
colors = ['C3', 'C0', 'C1', 'C2']

fs1 = 14
fs2 = 12

# Create the figure and the subplots
fig, ax = plt.subplots(1, 1, figsize=(4, 5), sharex=True)

# Set labels and titles for each subplot (optional)
file = os.path.join(data_dir, filename)
data = pd.read_csv(file)


pos = data['pos'].values + np.ones(len(data['pos'].values))
corrxx = 0.5*data['corr_pm'].values
corryy = 0.5*data['corr_mp'].values
corrzz = data['corr_zz'].values
corrstr = data['corr_str_order'].values

#ax.plot(pos, corrstr, label=r'$\mathcal{O}^z_{1,j}$', marker="s", color=colors[0])
ax.plot(pos, corrxx, label=r'$\langle S_1^xS_j^x \rangle$', marker="x", color=colors[1], ms=6)
ax.plot(pos, corryy, label=r'$\langle S_1^yS_j^y \rangle$', marker="+", color=colors[2], ms=8)
ax.plot(pos, corrzz, label=r'$\langle S_1^zS_j^z \rangle$', marker=".", color=colors[3], ms=6)


ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.set_ylim(-0.85, 0.85)
ax.set_xlabel("site $j$", fontsize=fs1)

plt.legend(loc='upper right', fontsize=fs2)
plt.tight_layout()

# Add a common legend on top
#lines_labels = [ax.get_legend_handles_labels() for ax in axs]
#lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#fig.legend(lines, labels, loc='upper center', ncol=4, fontsize='large')

## save figure
plt.savefig(output_file)

# Show the plot
plt.show()


