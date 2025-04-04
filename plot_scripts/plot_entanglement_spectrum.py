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
output_file = f"../plots/entanglment_spectrum_alpha{alpha}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"../output/full/Sz1/"



files = [f"spinone_heisenberg_ee_spectrum_chi{chi}_alpha{alpha}_L{L}.csv", f"spinone_heisenberg_obs_chi{chi}_alpha{alpha}_L{L}.csv"]
all_quantities = [[f"chi{i}" for i in range(10)], ["SvN"]]


markers = ["+", "x"]
labels = ["$\\xi_{i}$", "$S_{\\rm VN}$"]

# Set up the 4x1 plot layout
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

for i, (file, quantities, label) in enumerate(zip(files, all_quantities, labels)):

    axs[i].set_ylabel(label)
    for j, quantity in enumerate(quantities):

        print(quantity)
        # Read the data
        data = pd.read_csv(data_dir + file)
        x = data["Jz"].values
        y = data[quantity].values

        combined = list(zip(x, y))
        sorted_combined = sorted(combined)
        x, y = zip(*sorted_combined)

        # Plot data on the corresponding subplot
        axs[i].plot(x, y, marker=markers[i], color=f"C{j}")

# Label the x-axis on the last subplot only (shared x-axis)
axs[-1].set_xlabel("$J_z$")
#axs[0].set_ylim([1.1, 2.0])



# Adjust layout for readability
plt.tight_layout()
plt.savefig(output_file)

plt.show()


