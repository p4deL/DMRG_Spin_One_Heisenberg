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
data_dir = 'output/fidelity_test/'


file = os.path.join(data_dir, 'spinone_heisenberg_fidelity_alphaInf_L100.csv')

data = pd.read_csv(file)

Ds = data['D'].values
fidelity = data['fidelity'].values

# Create the contour plot
plt.figure(figsize=(8, 6))

plt.plot(Ds, fidelity)

# Label axes
plt.xlabel(r'D')
plt.ylabel(r'$\chi$')

# title
plt.title("$L=100$")


# Show plot
plt.show()

