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

# which phase diagram
phase_diag = "lambda_alpha"
phase_diag = "Gamma_alpha"
phase_diag = "Jz_alpha"
#phase_diag = "D_alpha"

# Directory where your CSV files are stored
data_dir = f'../data/phase_diagram/{phase_diag}_observables/L{L}/'
print(data_dir)

#quantity = ['SvN', r'$S_{\rm VN}$']
#quantity = ['str_order', r'$O^{\rm str}_{\frac{L}{4}, \frac{3L}{4}}$']
#quantity = ['eff_str_order', r'$O^{z,\rm str}_{\frac{L}{4}, \frac{3L}{4}}- \langle S^z_{\frac{L}{4}}S^z_{\frac{3L}{4}}\rangle$']
quantity = ['m_long', r'$M_z$']  # TODO. different mag directions
#quantity = ['m_trans', r'$M_{\perp}$']  # TODO. different mag directions
#quantity = ['fidelity', 'fidelity', r'$\chi_{\rm fidelity}$']

# output filename
output_file = f"../plots/colorplot_{phase_diag}_{quantity[0]}_L{L}.pdf"


def find_first_above_half_max(arr, max_val_input=float("inf")):
    """
    Find the index of the first entry in the array that is greater than half the maximum value.

    Parameters:
        arr (numpy.ndarray): Input array of float values.

    Returns:
        int: Index of the first entry greater than half the max value.
             Returns -1 if no such entry is found.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if arr.size == 0:
        return -1  # Handle empty array

    if math.isinf(max_val_input):
        max_val = np.max(arr)
    else:
        max_val = max_val_input

    target_val = max_val / 2.0

    # Find the first index where the value is greater than half the max value
    indices = np.where(arr > target_val)[0]
    return indices[0] if indices.size > 0 else -1


# Use glob to find all csv files that match the pattern
file_pattern = os.path.join(data_dir, f'spinone_heisenberg_obs_chi{chi}_alpha*.csv')

# font sizes
fs1 = 18
fs2 = 16

csv_files = glob.glob(file_pattern)

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
    elif phase_diag == "Gamma_alpha":
        combined = list(zip(data["Gamma"].values, data[quantity[0]].values))
    elif phase_diag == "Jz_alpha":
        combined = list(zip(data["Jz"].values, data[quantity[0]].values))
    else:
        combined = list(zip(data["D"].values, data[quantity[0]].values))

    sorted_combined = sorted(combined)
    d, z = zip(*sorted_combined)

    idx = find_first_above_half_max(np.array(z), max_val_input=0.95)
    if idx > -1:
        transition_D.append(d[idx])
        transition_alpha.append(1./alpha)

    D_values.append(d)
    z_values.append(z)

    if phase_diag == "Gamma_alpha":
        alpha_values.append(np.full_like(data["Gamma"].values, 1./alpha))  # Create an array of alphaval for each Gamma
    else:
        alpha_values.append(np.full_like(data["D"].values, 1./alpha))  # Create an array of alphaval for each D

# sort by alpha values
#if phase_diag == "lambda_alpha":
#    D_values = np.reciprocal(D_values)


# sort colorplot vals
combined = zip(alpha_values, D_values, z_values)
sorted_combined = sorted(combined, key=lambda x: x[0][0])
alpha_values, D_values, z_values = zip(*sorted_combined)

# Convert lists to arrays for plotting
D_values = np.concatenate(D_values)
z_values = np.concatenate(z_values)
alpha_values = np.concatenate(alpha_values)

# sort transition values
combined = zip(transition_alpha, transition_D)
sorted_combined = sorted(combined, key=lambda x: x[0])
#transition_alpha , transition_D = zip(*sorted_combined)

for Dc, alphac in zip(transition_D, np.reciprocal(transition_alpha)):
    print(f"alphac={alphac}, lambdac={Dc}")

#print(D_values)
#print(alpha_values)
#print(z_values)

# Now create a grid of D and alphaval values for contouring
D_grid, alpha_grid = np.meshgrid(np.unique(D_values), np.unique(alpha_values))
print(z_values)

# Create the contour plot
plt.figure(figsize=(8, 6))

colorplot = plt.pcolor(D_grid, alpha_grid, z_values.reshape(D_grid.shape), cmap='viridis')
#plt.plot(transition_D, transition_alpha, color='k')
#plt.scatter([-1.5, 0, 0, 0.5, 1.5], [0, 0, 0.5, 0.5, 0.], color='red', marker='o', s=100)

# Add colorbar
cbar = plt.colorbar(colorplot)
cbar.set_label(quantity[1], fontsize=fs2)

# Label axes
if phase_diag == "D_alpha":
    plt.xlabel(r'$D$', fontsize=fs2)
elif phase_diag == "Gamma_alpha":
    plt.xlabel('$\\Gamma$', fontsize=fs2)
elif phase_diag == "Jz_alpha":
    plt.xlabel("$J_z$", fontsize=fs2)
else:
    plt.xlabel('$\\lambda$', fontsize=fs2)
plt.ylabel('$1/\\alpha$', fontsize=fs2)

# title
plt.title(f"$L={L}$", fontsize=fs1)

# save fig
plt.savefig(output_file)

# Show plot
plt.show()


