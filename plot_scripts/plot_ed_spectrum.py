import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 12  # system size
alpha = 1.25
#alpha = 2.5

crit_point = 0.096676
#crit_point = 0.384695

# output filename
output_file = f"../plots/ed_spectrum_alpha{alpha}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"../data/fss/largeD_U(1)CSB_transition/ed_spectrum/"



files = [f"spinone_heisenberg_ed_spectrum_alpha{alpha}_L{L}.csv", f"spinone_heisenberg_ed_m_long_alpha{alpha}_L{L}.csv"]
all_quantities = [[f"e{i}" for i in range(9,0,-1)], ["m_long"]]


markers = ["+", "x"]
labels = ["$\\epsilon_{i}-\\epsilon_0$", "$M_z$"]

# Set up the 4x1 plot layout
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

for i, (file, quantities, label) in enumerate(zip(files, all_quantities, labels)):

    axs[i].axvline(x=crit_point, color='r', linestyle='--')
    axs[i].set_ylabel(label)
    for j, quantity in enumerate(quantities):

        print(quantity)
        # Read the data
        data = pd.read_csv(data_dir + file)
        x = data["D"].values
        y = data[quantity].values

        x = np.reciprocal(x)

        combined = list(zip(x, y))
        sorted_combined = sorted(combined)
        x, y = zip(*sorted_combined)

        if quantity == "e9":
            e0 = y

        # Plot data on the corresponding subplot
        if i == 0:
            axs[i].plot(x, np.array(y)-np.array(e0), marker=markers[i], color=f"C{j}")
            axs[i].set_yscale("log")
        else:
            axs[i].plot(x, y, marker=markers[i], color=f"C{j}")
            #axs[i].set_yscale("log")

# Label the x-axis on the last subplot only (shared x-axis)
axs[-1].set_xlabel("$\\lambda$")
#axs[0].set_ylim([1.1, 2.0])



# Adjust layout for readability
plt.tight_layout()
plt.savefig(output_file)

plt.show()


