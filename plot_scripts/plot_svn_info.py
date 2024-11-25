import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use, rc, rcParams


rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"



L = 60  # system size
alpha = "inf"
chi = 300

max_sweeps = 50

# output filename
output_file = f"../plots/svn_info_chi{chi}_alpha{alpha}_L{L}.pdf"

# directory and filename
#data_dir = f"output/L{L}/"
data_dir = f"../output/"


all_quantities = [["SvN"], ["fidelity", "log_fidelity"], ["overlap"], ["gs_energy", "gs_energy_eps"], ["gs_energy_diff"], ["parity_x", "parity_x_eps"], ["s_total", "s_total_eps"], ["chi_max", "chi_max_eps"], ["nsweeps", "nsweeps_eps"]]

markers = ["+", "x"]
labels = ["$S_{\\rm VN}$", "$\\chi_{\\rm fidelity}$", "$\\langle \\psi(D) \\vert \\psi(D+\\epsilon) \\rangle$", "$\\epsilon_{\\rm gs}$", "$\\Delta \\epsilon_{\\rm gs}$", "$P_x$", "$S_{\\rm tot}$", "$\\chi_{\\rm bond}$", "$N_{\\rm sweeps}$"]

# Set up the 4x1 plot layout
fig, axs = plt.subplots(9, 1, figsize=(8, 10), sharex=True)

for i, (quantities, label) in enumerate(zip(all_quantities, labels)):
    
    if i < 3:
        file =  f'spinone_heisenberg_fss_obs_chi{chi}_alpha{alpha}_L{L}.csv'
    else:
        file =  f'spinone_heisenberg_fss_trackobs_chi{chi}_alpha{alpha}_L{L}.csv'

    axs[i].set_ylabel(label)
    for j, quantity in enumerate(quantities):
        
        # Read the data
        data = pd.read_csv(data_dir + file)
        x = data["D"].values
        y = data[quantity].values

        if quantity == "fidelity" or quantity == "log_fidelity":
            y = y/L

        combined = list(zip(x, y))
        sorted_combined = sorted(combined)
        x, y = zip(*sorted_combined)

        #if quantity == "overlap":   
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
axs[1].set_ylim(0.0,0.7)
#axs[2].set_ylim(0.99,1.00000)
#axs[1].set_ylim(1.-2e-6,1.+2e-6)
axs[5].set_ylim(-1.2,1.2)
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


