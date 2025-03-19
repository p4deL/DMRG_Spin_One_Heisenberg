import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

from matplotlib import use, rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"


# system size 
L = 100
chi = 300

#quantity = ['SvN', r'$S_{\rm VN}$']
#quantity = ['str_order', r'$O^{\rm str}_{\frac{L}{4}, \frac{3L}{4}}$']
#quantity = ['eff_str_order', r'$O^{z,\rm str}_{\frac{L}{4}, \frac{3L}{4}}- \langle S^z_{\frac{L}{4}}S^z_{\frac{3L}{4}}\rangle$']
#quantity = ['m_long', r'$M_z$']  # TODO. different mag directions
quantity = ['m_trans', r'$M_{\perp}$']  # TODO. different mag directions
#quantity = ['fidelity', 'fidelity', r'$\chi_{\rm fidelity}$']

# output filename
output_file = f"../plots/colorplot_D_alpha_{quantity[0]}_L{L}.pdf"


# font sizes
fs1 = 18
fs2 = 16

def get_plot_data(csv_files, phase_diag):
    # Lists to hold the data
    D_values = []
    alpha_values = []
    z_values = []

    transition_D = []
    transition_alpha = []

    # Iterate through all the files
    for file in csv_files:
        # Extract alphaval from the filename
        alpha = float(file.split('_alpha')[-1].split(f'_L{L}.csv')[0])  # Extract alpha from filename
        #print(alpha)

        # Read the CSV file
        data = pd.read_csv(file)

        # Append data to lists
        #values = np.full_like(data["D"].values, alpha)
        # sort by D values
        if phase_diag == "lambda_alpha":
            combined = list(zip(np.reciprocal(data["D"].values), data[quantity[0]].values))
        else:
            combined = list(zip(data["D"].values, data[quantity[0]].values))

        sorted_combined = sorted(combined)
        d, z = zip(*sorted_combined)

        #d = d[2:]
        #z = z[2:]

        print(f"alpha={alpha}: len(z_values)={len(z)}")

        D_values.append(d)
        z_values.append(z)

        alpha_values.append(np.full_like(data["D"].values, 1./alpha))  # Create an array of alphaval for each D

    # sort colorplot vals
    combined = zip(alpha_values, D_values, z_values)
    sorted_combined = sorted(combined, key=lambda x: x[0][0])
    alpha_values, D_values, z_values = zip(*sorted_combined)

    # Convert lists to arrays for plotting
    D_values = np.concatenate(D_values)
    z_values = np.concatenate(z_values)
    alpha_values = np.concatenate(alpha_values)

    # Now create a grid of D and alphaval values for contouring
    D_grid, alpha_grid = np.meshgrid(np.unique(D_values), np.unique(alpha_values))
    print(z_values)

    return D_grid, alpha_grid, z_values.reshape(D_grid.shape)

# which phase diagram
phase_diag = "lambda_alpha"
#phase_diag = "Gamma_alpha"
#phase_diag = "Jz_alpha"


# Directory where your CSV files are stored
phase_diag = "D_alpha"
data_dir = f'../data/phase_diagram/{phase_diag}_observables/Sz0/B-1e-2/L{L}/'
file_pattern = os.path.join(data_dir, f'spinone_heisenberg_obs_chi{chi}_alpha*.csv')
csv_files = glob.glob(file_pattern)
D_grid, alpha_D_grid, z_D_values = get_plot_data(csv_files, phase_diag)

phase_diag = "lambda_alpha"
data_dir = f'../data/phase_diagram/{phase_diag}_observables/L{L}/'
file_pattern = os.path.join(data_dir, f'spinone_heisenberg_obs_chi{chi}_alpha*.csv')
csv_files = glob.glob(file_pattern)
lambda_grid, alpha_lambda_grid, z_lambda_values = get_plot_data(csv_files, phase_diag)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False, gridspec_kw={'wspace': 0})

# Define global color limits
vmin, vmax = min(z_D_values.min(), z_lambda_values.min()), max(z_D_values.max(), z_lambda_values.max())

# First plot (linear scale)
print(D_grid.shape, alpha_D_grid.shape, z_D_values.shape)
cp1 = ax1.pcolor(D_grid, alpha_D_grid, z_D_values, cmap='viridis', vmin=vmin, vmax=vmax) #cmap='rainbow') #cmap='viridis')

# Second plot (1/x scale)
cp2 = ax2.pcolor(lambda_grid, alpha_lambda_grid, z_lambda_values, cmap='viridis', vmin=vmin, vmax=vmax) #cmap='rainbow') #cmap='viridis')

# Remove extra space between subplots
ax2.set_yticklabels([])

# Add a single colorbar
cbar = fig.colorbar(cp1, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.02)
#cbar.set_label("Z Value")
cbar.set_label(quantity[1], fontsize=fs2)

# Label axes
#ax1.set_xlabel(r'$D$', fontsize=fs2)
ax2.set_xlim([1.0,0.02])
# Set custom tick positions and labels
tick_positions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
tick_labels = ["$1.0$", "$5/4$", "$5/3$", "$5/2$", "$5$", "$\infty$"]
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
#ax2.set_xlabel('$D$', fontsize=fs2)
ax1.set_ylabel('$1/\\alpha$', fontsize=fs2)

fig.text(0.5, 0.02, "$D$", ha='center', fontsize=fs2)  # Bottom middle

# save fig
plt.savefig(output_file)

# Show plot
plt.show()


