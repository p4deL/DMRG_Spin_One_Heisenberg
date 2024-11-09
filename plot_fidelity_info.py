import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 100  # system size
alpha = 10.0
chi = 300

max_sweeps = 100

# output filename
output_file = f"plots/fidelity_info_chi{chi}_alpha{alpha}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"output/"


all_quantities = [["fidelity"], ["overlap"], ["gs_energy", "gs_energy_eps"], ["gs_energy_diff"], ["parity_x", "parity_x_eps"], ["s_total", "s_total_eps"], ["chi_max", "chi_max_eps"], ["nsweeps", "nsweeps_eps"]]

markers = ["+", "x"]
labels = ["$\\chi_{\\rm fidelity}$", "$\\langle \\psi(D) \\vert \\psi(D+\\epsilon) \\rangle$", "$\\epsilon_{\\rm gs}$", "$\\Delta \\epsilon_{\\rm gs}$", "$P_x$", "$S_{\\rm tot}$", "$\\chi_{\\rm bond}$", "$N_{\\rm sweeps}$"]

# Set up the 4x1 plot layout
fig, axs = plt.subplots(8, 1, figsize=(8, 10), sharex=True)

for i, (quantities, label) in enumerate(zip(all_quantities, labels)):
    
    axs[i].set_ylabel(label)
    for j, quantity in enumerate(quantities):
        file =  f'spinone_heisenberg_{quantity}_chi{chi}_alpha{alpha}_L{L}.csv'
        # Read the data
        data = pd.read_csv(data_dir + file)
        x = data["D"].values
        y = data[quantity].values

        combined = list(zip(x, y))
        sorted_combined = sorted(combined)
        x, y = zip(*sorted_combined)

        #if quantity == "overlap":   
            #print("Hello")
            #eps = 1e-4
            #y = -2*np.log(y)/eps**2

        # don't forget to sort 

        # Plot data on the corresponding subplot
        if j == 0:
            axs[i].plot(x, y, label=f"$D$", marker=markers[j], color=f"C{j}")
        else:
            axs[i].plot(x, y, label=f"$D+\\epsilon$", marker=markers[j], color=f"C{j}")
        #axs[i].set_xlabel("$Quantity$")
        axs[i].legend(loc="upper right")

# Label the x-axis on the last subplot only (shared x-axis)
axs[0].set_ylim(-0.5,80)
#axs[1].set_ylim(1.-2e-6,1.+2e-6)
axs[4].set_ylim(-1.2,1.2)
axs[-2].set_ylim(0,chi+50)
axs[-2].plot([x[0],x[-1]], [chi, chi], lw=2, c="C10")
axs[-2].text(x[0], chi+15, "max bond dim", color="gray")
axs[-1].set_ylim(0,max_sweeps+0.3*max_sweeps)
axs[-1].plot([x[0],x[-1]], [max_sweeps, max_sweeps], lw=2, c="C10")
axs[-1].text(0.05, 0.85, "max sweeps", color="gray", transform=axs[-1].transAxes)

axs[-1].set_xlabel("$D$")

# Adjust layout for readability
plt.tight_layout()
plt.savefig(output_file)

plt.show()


