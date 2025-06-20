import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 100  # system size
alpha = "inf"
#alpha = 2.5

chi = 300

# output filename
output_file = f"../plots/energy_and_entanglement_alpha{alpha}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"../output/"


N=3
files_energies = [f"spinone_heisenberg_psi{i}_trackobs_chi{chi}_alpha{alpha}_L{L}.csv" for i in range(N)]
files_entropies = [f"spinone_heisenberg_psi{i}_obs_chi{chi}_alpha{alpha}_L{L}.csv" for i in range(N)]
files = [files_energies, files_entropies]
quantities = ["gs_energy", "SvN"]


markers = [".", "s", "*"]
markersizes = [14, 6, 12]
labels = ["$\\epsilon$", "$S_{\\rm VN}$"]

# Set up the 4x1 plot layout
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

for i, (files_quantity, quantity, label) in enumerate(zip(files, quantities, labels)):

    axs[i].set_ylabel(label)
    for j, file in enumerate(files_quantity):

        # Read the data
        data = pd.read_csv(data_dir + file)
        x = data["Jz"].values
        if i == 0:
            y = data[quantity].values/L
        else:
            y = data[quantity].values

        combined = list(zip(x, y))
        sorted_combined = sorted(combined)
        x, y = zip(*sorted_combined)

        # Plot data on the corresponding subplot
        axs[i].plot(x, y, marker=markers[j], ms=markersizes[j], lw=2.5, color=f"C{j}", label="$\\psi_{" + f"{j}" + "}$")

    axs[i].legend(loc="best")

# Label the x-axis on the last subplot only (shared x-axis)
axs[-1].set_xlabel("$J_z$")

#axs[0].set_ylim([1.1, 2.0])



# Adjust layout for readability
plt.tight_layout()
plt.savefig(output_file)

plt.show()


