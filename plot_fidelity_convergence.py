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
alpha = "Inf"

# output filename
output_file = "initstate_convergence_positiveD.pdf"

# directory and filename
data_dir = 'output/fidelity_convergence/positiveD/'
filename_largeD = f'fidelity_largeDinit_alpha{alpha}_L{L}.csv'
filename_af = f'fidelity_afinit_alpha{alpha}_L{L}.csv'
filename_haldane = f'fidelity_haldaneinit_alpha{alpha}_L{L}.csv'

fs1 = 18
fs2 = 16


file_largeD = os.path.join(data_dir, filename_largeD)
file_af = os.path.join(data_dir, filename_af)
file_haldane = os.path.join(data_dir, filename_haldane)

data_largeD = pd.read_csv(file_largeD)
data_af = pd.read_csv(file_af)
data_haldane = pd.read_csv(file_haldane)

# Create the contour plot
plt.figure(figsize=(8, 6))

Ds = data_largeD['D'].values
fidelity = data_largeD['fidelity'].values
plt.plot(Ds, fidelity, label="large D init state")

Ds = data_af['D'].values
fidelity = data_af['fidelity'].values
plt.plot(Ds, fidelity, label="AF init state")

Ds = data_haldane['D'].values
fidelity = data_haldane['fidelity'].values
plt.plot(Ds, fidelity, label="Haldane init state")

# Label axes
plt.xlabel(r'$D$', fontsize=fs2)
plt.ylabel(r'$\chi_{\rm fidelity}$', fontsize=fs2)

plt.ylim(0,20)
plt.legend(fontsize=fs2)

# title
plt.title(f"$L={L}$", fontsize=fs1)

# save figure
plt.savefig(output_file)

# show plot
plt.show()
