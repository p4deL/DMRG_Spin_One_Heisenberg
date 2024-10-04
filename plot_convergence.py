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


# Directory where your CSV files are stored
data_dir = 'output/gse_convergence/'


file = os.path.join(data_dir, 'gse_convergence_L100_noqn.csv')

data = pd.read_csv(file)

Ds = data['step'].values
energy = data['energy'].values

# Create the contour plot
plt.figure(figsize=(8, 6))

#plt.yscale('log')
plt.plot(Ds, abs(energy))

# Label axes
plt.xlabel(r'$D$')
plt.ylabel(r'$\chi$')

# title
plt.title("$L=100$")


# Show plot
plt.show()

