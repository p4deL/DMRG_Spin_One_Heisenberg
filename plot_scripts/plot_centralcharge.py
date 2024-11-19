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



L1 = 100  # system size
L2 = 110
alpha = "Inf"

# output filename
output_file = f"initstate_central_charge_L{L}.pdf"

# directory and filename
data_dir = 'output/ceff/'
filename = f'spinone_heisenberg_centralcharge_alpha{alpha}_1L{L1}_2L{L2}.csv'

fs1 = 18
fs2 = 16


file = os.path.join(data_dir, filename)
data = pd.read_csv(file)

# Create the contour plot
plt.figure(figsize=(8, 6))

Ds = data['D'].values
ceff = data['ceff'].values
plt.plot(Ds, ceff)



# Label axes
plt.xlabel(r'$D$', fontsize=fs2)
plt.ylabel(r'$c_{\rm eff}$', fontsize=fs2)

plt.ylim(0,100)
plt.legend(fontsize=fs2)

# title
plt.title(f"$L_1={L1} {\\rm and} L_2={L2}$", fontsize=fs1)

# save figure
plt.savefig(output_file)

# show plot
plt.show()
