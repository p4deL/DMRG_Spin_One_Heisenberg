import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 100  # system size
alpha = "inf"
chi = 300

max_sweeps = 50

# output filename
output_file = f"plots/fidelity_field_chi{chi}_alpha{alpha}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"output/"


#dirs = ["B1e-0", "B1e-1", "B1e-2", "B1e-3", "B1e-4", "B1e-5", "B1e-6"]
dirs = ["B-1e-0", "B-1e-1", "B-1e-2", "B-1e-3", "B-1e-4", "B-1e-5", "B-1e-6"]

quantity = "fidelity"


#markers = ["+", "x"]
label = "$\\chi_{\\rm fidelity}$"

# Set up the 4x1 plot layout
fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True)

for i, dir in enumerate(dirs):
    
    

    # Read the data
    file =  f'{data_dir}{dir}/spinone_heisenberg_fidelity_obs_chi{chi}_alpha{alpha}_L{L}.csv'
    data = pd.read_csv(file)
    x = data["D"].values
    y = data[quantity].values

    if quantity == "fidelity" or quantity == "log_fidelity":
        y = y/L

    combined = list(zip(x, y))
    sorted_combined = sorted(combined)
    x, y = zip(*sorted_combined)

    # Plot data on the corresponding subplot
    ax.plot(x, y, label=f"{dir}", marker="x", color=f"C{i}", lw=2)

# Label the x-axis on the last subplot only (shared x-axis)
ax.legend(loc="upper right")
ax.set_ylim(0.0,0.4)
ax.set_xlabel("$D$")
ax.set_ylabel("$\\chi_{\\rm fidelity}$")

# Adjust layout for readability
plt.tight_layout()
plt.savefig(output_file)

plt.show()



