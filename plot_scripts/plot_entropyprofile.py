import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import curve_fit

from matplotlib import use, rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"

L = 1000 # system size
D = -0.3
alpha = 'inf'
G = 1.0
Jz = 1.0
chi = 300


# output filename
output_file = f"../plots/entropies_chi{chi}_D{D}_Gamma{G}_Jz{Jz}_alpha{alpha}_L{L}.pdf"

# directory and filename
data_dir = f'../data/fss/ising_transition/central_charge/'
filename = f"spinone_heisenberg_entropies_chi{chi}_D{D}_Gamma{G}_Jz{Jz}_alpha{alpha}_L{L}.csv"


def fit_func(x, ceff, const):
    return ceff/6*np.log(2*L/np.pi*np.sin(np.pi*x/L)) + const

def perform_entropy_fit(x, y):

    # TODO: check parameters
    popt, pcov = curve_fit(fit_func, x, y, p0=[0.5, 0.7])

    ceff, const = popt
    ceff_err, const_err = np.sqrt(np.diag(pcov))
    print(ceff, "+/-", ceff_err)
    print(const, "+/-", const_err)

    return ceff, ceff_err, const, const_err

#labels = ["large D $(D=1.4,~\\alpha=10.0)$", "Haldane $(D=0.0,~\\alpha=10.0)$", "unkown $(D=1.0,~\\alpha=3.0)$"]
#markers = ["x", "s", "."]

#filenames = [filename_haldane]
#labels = ["Haldane $(D=0.0,~\\alpha=10.0)$"]
#markers = ["s"]

#filenames = [filename_largeD, filename_af, filename_haldane, filename_unkown]
#labels = ["large D $(D=1.4,~\\alpha=10.0)$", "z-AF $(D=-0.3,~\\alpha=1.5)$", "Haldane $(D=0.0,~\\alpha=10.0)$", "unkown $(D=1.0,~\\alpha=3.2)$"]
#colors = ['C3', 'C0', 'C1', 'C2']

fs1 = 18
fs2 = 13

# Create the figure and the subplots
fig, ax = plt.subplots(1, 1, figsize=(6, 8), sharex=True)

# Set labels and titles for each subplot (optional)
file = os.path.join(data_dir, filename)
data = pd.read_csv(file)

#ax.text(0.98, 0.87, label, transform=ax.transAxes, ha='right', fontsize=fs2)
#ax.set_ylim(-0.5,0.5)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

pos = data['pos'].values + 0.5*np.ones(len(data['pos'].values))
entropies = data['SvN'].values
#label = r'$\vert\langle S_1^zS_j^z \rangle\vert$',

ax.plot(pos, entropies, marker=".", markersize=15)

cut = 10
ceff, ceff_err, const, const_err = perform_entropy_fit(pos[cut:-cut], entropies[cut:-cut])
xrange = np.linspace(pos[cut], pos[-cut])
ax.plot(xrange, fit_func(xrange, ceff, const), lw=3)

ax.set_ylabel("$S_{\\rm VN}$", fontsize=fs2)
ax.set_xlabel("$\\ell$", fontsize=fs2)

ax.text(0.3, 0.1, "$c_{\\rm eff}=~$" + f"${ceff:.3f}\\pm{ceff_err:.3f}$", transform=ax.transAxes, fontsize=fs1)

#plt.title(f"$D={D}\\sim D_c$", fontsize=fs1)
plt.title(f"$D={D}\\sim D_c+\\varepsilon$", fontsize=fs1)
fig.legend()
# Ensure that subplots are tightly packed vertically
plt.subplots_adjust(hspace=0.1, top=0.9)

## save figure
plt.savefig(output_file)

# Show the plot
plt.show()


