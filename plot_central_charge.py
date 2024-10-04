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
data_dir = 'output/ceff/'


file = os.path.join(data_dir, 'spinone_heisenberg_centralcharge_alpha10.0_1L100_2L110.csv')

data = pd.read_csv(file)

data = data.sort_values(by="D")



Ds = data['D'].values
ceff = data['ceff'].values




# Create the contour plot
plt.figure(figsize=(8, 6))

plt.plot(Ds, ceff)

# Label axes
plt.xlabel(r'$D$')
plt.ylabel(r'$c_{\rm eff}$')

# title
plt.title(r"$L_1=100 {\rm and} L_2=110$")


# Show plot
plt.show()

