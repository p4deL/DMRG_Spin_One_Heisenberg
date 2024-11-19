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



L = 64  # system size
alpha = 4.0
chi = 150


# output filename
output_file = f"plots/fidelity_alpha{alpha}_chi{chi}_L{L}.pdf"

# directory and filename
data_dir = f"output/L{L}/"
#filename_dmrg_alg = f'spinone_heisenberg_fidelity_alpha{alpha}_L{L}.csv'
filename_dmrg_exp = f'spinone_heisenberg_exp_fidelity_alpha{alpha}_L{L}.csv'
#filename_ed = f'ed_spinone_heisenberg_fidelity_alpha{alpha}_L{L}.csv'

fs1 = 18
fs2 = 16


#file_dmrg_alg = os.path.join(data_dir, filename_dmrg_alg)
#data_dmrg_alg = pd.read_csv(file_dmrg_alg)

file_dmrg_exp = os.path.join(data_dir, filename_dmrg_exp)
data_dmrg_exp = pd.read_csv(file_dmrg_exp)

#file_ed = os.path.join(data_dir, filename_ed)
#data_ed = pd.read_csv(file_ed)

# Create the contour plot
plt.figure(figsize=(8, 6))

#combined = list(zip(data_dmrg_alg["D"].values, data_dmrg_alg["fidelity"].values))
#sorted_combined = sorted(combined)
#Ds_dmrg_alg, fidelity_dmrg_alg = zip(*sorted_combined)

combined = list(zip(data_dmrg_exp["D"].values, data_dmrg_exp["fidelity"].values))
sorted_combined = sorted(combined)
Ds_dmrg_exp, fidelity_dmrg_exp = zip(*sorted_combined)


#combined = list(zip(data_ed["D"].values, data_ed["fidelity"].values))
#sorted_combined = sorted(combined)
#Ds_ed, fidelity_ed = zip(*sorted_combined)


#plt.plot(Ds_ed, fidelity_ed, marker="o", label="ed")
#plt.plot(Ds_dmrg_alg, fidelity_dmrg_alg, marker="x", label="dmrg - power law")
plt.plot(Ds_dmrg_exp, fidelity_dmrg_exp, marker="+", label="dmrg - sum exps")



# Label axes
plt.xlabel(r'$D$', fontsize=fs2)
plt.ylabel(r'$\chi_{\rm fidelity}$', fontsize=fs2)

plt.ylim(-0.2,40)
#plt.legend(fontsize=fs2)

# title
plt.title(f"$L={L},~\\alpha={alpha},~\\chi={chi}$", fontsize=fs1)
plt.legend()

# save figure
plt.savefig(output_file)

# show plot
plt.show()
