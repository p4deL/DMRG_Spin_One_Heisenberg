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


alpha = "inf"
L = 1000  # system size

phase_diag = "D_alpha"

chi = 300

# output filename
output_file = f"../plots/order_parameters_alpha{alpha}_L{L}.pdf"

# directory and filename
data_dir = f'../data/phase_diagram/{phase_diag}_observables/order_parameters/L{L}/'

filename = f'spinone_heisenberg_obs_chi{chi}_alpha{alpha}_L{L}.csv'

fs1 = 18
fs2 = 13

# Create the figure and the subplots
#fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), sharex=True)
fig, ax = plt.subplots(1, 1, figsize=(15, 3), sharex=True)

#ax.grid(True)
file = os.path.join(data_dir, filename)
data = pd.read_csv(file)

#ax.text(0.98, 0.87, label, transform=ax.transAxes, ha='right', fontsize=fs2)
#ax.set_ylim(-0.5,0.5)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))


combined = list(zip(data["D"].values, data["m_long"].values))
sorted_combined = sorted(combined)
Ds, mz = zip(*sorted_combined)

combined = list(zip(data["D"].values, data["eff_str_order"].values))
sorted_combined = sorted(combined)
Ds, eff_str_order = zip(*sorted_combined)

#label = r'$\vert\langle S_1^zS_j^z \rangle\vert$',


ax.plot(Ds, mz, color="#00afb9", label="$M^z$", lw=2, marker='o', linestyle='-')
ax.plot(Ds, eff_str_order, color="#f07167", label="$\\mathcal{O}^{\\rm str}- C^{zz}$", lw=2, marker='s', linestyle='-')

ax.set_xlabel(r"$D$", fontsize=fs1)
ax.set_ylabel(r"order parameters", fontsize=fs1)

# Add a common legend on top
plt.legend(loc="upper right", ncol=1, fontsize=fs2)
#axs[0].set_title(f"$L={L}$", fontsize=fs1)

fig.tight_layout()

## save figure
fig.savefig(output_file)

# Show the plot
plt.show()


