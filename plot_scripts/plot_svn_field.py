import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 100  # system size
alpha = "inf"
chi = 500

max_sweeps = 50

# output filename
output_file = f"../plots/svn_field_chi{chi}_alpha{alpha}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"../output/auxfield_test/"


#dirs = ["B1e-0", "B1e-1", "B1e-2", "B1e-3", "B1e-4", "B1e-5", "B1e-6"]
dirs = ["B1e-0", "B1e-1", "B1e-2", "B1e-3", "B1e-4", "B1e1", "B1e2"]
markers = ["o", "o", "o", "+", "x", "x", "x"]
marker_sizes = [8, 8, 8, 10, 6, 6, 6]

quantity = "SvN"
label = "$S_{\\rm VN}$"

# Set up the 4x1 plot layout
fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True)

for i, (dir, marker, ms) in enumerate(zip(dirs, markers, marker_sizes)):
    
    # Read the data
    file = f'{data_dir}{dir}/spinone_heisenberg_obs_chi{chi}_alpha{alpha}_L{L}.csv'
    data = pd.read_csv(file)
    x = data["D"].values
    y = data[quantity].values

    if quantity == "fidelity" or quantity == "log_fidelity":
        y = y/L

    combined = list(zip(x, y))
    sorted_combined = sorted(combined)
    x, y = zip(*sorted_combined)

    # Plot data on the corresponding subplot
    ax.plot(x, y, label=f"{dir}", marker=marker, color=f"C{i}", ms=ms, mew=2, lw=2)

# Label the x-axis on the last subplot only (shared x-axis)
ax.legend(loc="upper right")
#ax.set_ylim(0.0,0.4)
ax.set_xlabel("$D$")
ax.set_ylabel("$S_{\\rm VN}$")

# Adjust layout for readability
plt.tight_layout()
plt.savefig(output_file)

plt.show()



