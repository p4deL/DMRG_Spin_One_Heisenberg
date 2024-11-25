import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



Ls = [60, 80, 100, 120, 140] # system size
alpha = "inf"
chi = 300

max_sweeps = 50

# output filename
output_file = f"../plots/svn_fss_chi{chi}_alpha{alpha}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"../output/"

files = [f"{data_dir}L{L}/spinone_heisenberg_fss_obs_chi{chi}_alpha{alpha}_L{L}.csv" for L in Ls]

quantity = "SvN"


#markers = ["+", "x"]
label = "$S_{\\rm VN}$"

# Set up the 4x1 plot layout
fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True)

for i, (file, L) in enumerate(zip(files, Ls)):
    
    # Read the data
    data = pd.read_csv(file)
    x = data["D"].values
    y = data[quantity].values

    combined = list(zip(x, y))
    sorted_combined = sorted(combined)
    x, y = zip(*sorted_combined)

    # Plot data on the corresponding subplot
    ax.plot(x, y, label=f"$L={L}$", marker="x", color=f"C{i}", lw=2)

# Label the x-axis on the last subplot only (shared x-axis)
ax.legend(loc="lower right")
#ax.set_ylim(0.0,0.4)
ax.set_xlabel("$D$")
ax.set_ylabel("$S_{\\rm VN}$")

# Adjust layout for readability
plt.tight_layout()
plt.savefig(output_file)

plt.show()



