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



L = 10  # system size
alpha = 10.0

# output filename
output_file = f"fidelity_alpha{alpha}_L{L}.pdf"

# directory and filename
data_dir = 'output/'
filename_dmrg = f'spinone_heisenberg_fidelity_alpha{alpha}_L{L}.csv'
filename_ed = f'ed_spinone_heisenberg_fidelity_alpha{alpha}_L{L}.csv'

fs1 = 18
fs2 = 16


file_dmrg = os.path.join(data_dir, filename_dmrg)
data_dmrg = pd.read_csv(file_dmrg)

file_ed = os.path.join(data_dir, filename_ed)
data_ed = pd.read_csv(file_ed)

# Create the contour plot
plt.figure(figsize=(8, 6))

combined = list(zip(data_dmrg["D"].values, data_dmrg["fidelity"].values))
sorted_combined = sorted(combined)
Ds_dmrg, fidelity_dmrg = zip(*sorted_combined)


combined = list(zip(data_ed["D"].values, data_ed["fidelity"].values))
sorted_combined = sorted(combined)
Ds_ed, fidelity_ed = zip(*sorted_combined)


plt.plot(Ds_dmrg, fidelity_dmrg, marker="x", label="dmrg")
plt.plot(Ds_ed, fidelity_ed, marker="+", label="ed")


# Label axes
plt.xlabel(r'$D$', fontsize=fs2)
plt.ylabel(r'$\chi_{\rm fidelity}$', fontsize=fs2)

#plt.ylim(0,2500)
#plt.legend(fontsize=fs2)

# title
plt.title(f"$L={L},~\\alpha={alpha}$", fontsize=fs1)
plt.legend()

# save figure
plt.savefig(output_file)

# show plot
plt.show()
