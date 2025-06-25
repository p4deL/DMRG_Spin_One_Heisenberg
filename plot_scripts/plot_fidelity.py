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



Ls = [60, 80, 100]  # system size
alpha = 'inf'
chi = 300

# output filename
output_file = f"../plots/fidelity_alpha{alpha}.pdf"

# directory and filename
data_dir = '../output/fidelity/Sz1/'

fs1 = 18
fs2 = 16

# Create the contour plot
plt.figure(figsize=(8, 6))

for L in Ls:

    filename = f'spinone_heisenberg_fss_obs_chi{chi}_alpha{alpha}_L{L}.csv'
    file = os.path.join(data_dir, filename)
    data = pd.read_csv(file)



    combined = list(zip(data["D"].values, data["fidelity"].values))
    sorted_combined = sorted(combined)
    Ds, fidelity = zip(*sorted_combined)

    plt.plot(Ds, fidelity)


# Label axes
plt.xlabel(r'$D$', fontsize=fs2)
plt.ylabel(r'$\chi_{\rm fidelity}$', fontsize=fs2)

#plt.ylim(0,2500)
#plt.legend(fontsize=fs2)

# title
#plt.title(f"$L={L}$", fontsize=fs1)

# save figure
plt.savefig(output_file)

# show plot
plt.show()
