import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 10  # system size
alpha = 1.1
chi = 150

max_sweeps = 30

# output filename
output_file = f"plots/fidelity_info_alpha{alpha}_chi{chi}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"output/"


all_quantities = [["fidelity"], ["gs_energy", "gs_energy_eps"], ["gs_energy_diff"], ["parity_x", "parity_x_eps"], ["s_total", "s_total_eps"], ["nsweeps", "nsweeps_eps"]]

markers = ["+", "x"]
labels = ["$\\chi_{\\rm fidelity}$", "$\\epsilon_{\\rm gs}$", "$\\Delta \\epsilon_{\\rm gs}$", "$P_x$", "$S_{\\rm tot}$", "$N_{\\rm sweeps}$"]

# Set up the 4x1 plot layout
fig, axs = plt.subplots(6, 1, figsize=(8, 10), sharex=True)

for i, (quantities, label) in enumerate(zip(all_quantities, labels)):
    
    axs[i].set_ylabel(label)
    for j, quantity in enumerate(quantities):
        file =  f'spinone_heisenberg_{quantity}_alpha{alpha}_L{L}.csv'
        # Read the data
        data = pd.read_csv(data_dir + file)
        x = data["D"].values
        y = data[quantity].values

        combined = list(zip(x, y))
        sorted_combined = sorted(combined)
        x, y = zip(*sorted_combined)

        # don't forget to sort 

        # Plot data on the corresponding subplot
        if j == 0:
            axs[i].plot(x, y, label=f"$D$", marker=markers[j], color=f"C{j}")
        else:
            axs[i].plot(x, y, label=f"$D+\\epsilon$", marker=markers[j], color=f"C{j}")
        #axs[i].set_xlabel("$Quantity$")
        axs[i].legend(loc="upper right")

# Label the x-axis on the last subplot only (shared x-axis)
axs[3].set_ylim(-1.2,1.2)
axs[-1].plot([-2.5,1.5], [max_sweeps, max_sweeps], lw=2, c="C9")
axs[-1].text(0.05, 0.85, "max sweeps", color="gray", transform=axs[-1].transAxes)

axs[-1].set_xlabel("$D$")

# Adjust layout for readability
plt.tight_layout()
plt.show()


