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
output_file = "initstate_gse_convergence.pdf"

# directory and filename
data_dir = 'output/gse_convergence/'
filename_largeD = f'gsenergy_largeDinit_alpha{alpha}_L{L}_D0.45.csv'
filename_af = f'gsenergy_afinit_alpha{alpha}_L{L}_D0.45.csv'

fs1 = 18
fs2 = 16


file_largeD = os.path.join(data_dir, filename_largeD)
file_af = os.path.join(data_dir, filename_af)

data_largeD = pd.read_csv(file_largeD)
data_af = pd.read_csv(file_af)

# Create the contour plot
plt.figure(figsize=(8, 6))

steps = data_largeD['step'].values
energy = data_largeD['energy'].values
plt.plot(steps, energy, label="large D init state")

steps = data_af['step'].values
energy = data_af['energy'].values
plt.plot(steps, energy, label="AF init state")


# Label axes
plt.xlabel(r'$step$', fontsize=fs2)
plt.ylabel(r'$varepsilon_{\rm GS}$', fontsize=fs2)


# title
plt.title("$L={L}$", fontsize=fs1)

# save figure
plt.savefig(output_file)

# show plot
plt.show()
