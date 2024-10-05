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
output_file = f"fidelity_alpha{alpha}_L{L}.pdf"

# directory and filename
data_dir = 'output/fidelity/'
filename = f'spinone_heisenberg_fidelity_alpha{alpha}_L{L}.csv'

fs1 = 18
fs2 = 16


file = os.path.join(data_dir, filename)
data = pd.read_csv(file)

# Create the contour plot
plt.figure(figsize=(8, 6))

Ds = data['D'].values
fidelity = data['fidelity'].values
plt.plot(Ds, fidelity)


# Label axes
plt.xlabel(r'$D$', fontsize=fs2)
plt.ylabel(r'$\chi_{\rm fidelity}$', fontsize=fs2)

plt.ylim(0,100)
#plt.legend(fontsize=fs2)

# title
plt.title(f"$L={L}$", fontsize=fs1)

# save figure
plt.savefig(output_file)

# show plot
plt.show()
