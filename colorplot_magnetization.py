import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from matplotlib import use, rc, rcParams

#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"


# Directory where your CSV files are stored
data_dir = 'output/'

# Use glob to find all csv files that match the pattern
file_pattern = os.path.join(data_dir, 'spinone_heisenberg_fidelity_alpha*.csv')
csv_files = glob.glob(file_pattern)

# Lists to hold the data
D_values = []
alpha_values = []
mag_values = []

# Iterate through all the files
for file in csv_files:
    # Extract alphaval from the filename
    alpha = float(file.split('_alpha')[-1].split('_L100.csv')[0]) # Extract alpha from filename

    # Read the CSV file
    #data = pd.read_csv(file, header=None, names=["D", "mag"])
    data = pd.read_csv(file)

    # Append data to lists
    D_values.append(data["D"].values)
    mag_values.append(data["fidelity"].values)
    alpha_values.append(np.full_like(data["D"].values, 1./alpha))  # Create an array of alphaval for each D

# Convert lists to arrays for plotting
D_values = np.concatenate(D_values)
mag_values = np.concatenate(mag_values)
alpha_values = np.concatenate(alpha_values)


#print(D_values)
print(alpha_values)
#print(mag_values)

# Now create a grid of D and alphaval values for contouring
D_grid, alpha_grid = np.meshgrid(np.unique(D_values), np.unique(alpha_values))

# Interpolate z_values onto the grid
from scipy.interpolate import griddata
mag_grid = griddata((D_values, alpha_values), mag_values, (D_grid, alpha_grid), method='cubic')

# Create the contour plot
plt.figure(figsize=(8, 6))
#contour = plt.contourf(D_grid, alpha_grid, mag_grid, cmap='viridis', levels=20)
contour = plt.contourf(D_grid, alpha_grid, mag_grid, cmap='viridis', levels=20)

# Add colorbar
#plt.colorbar(contour, label='squared stag. magnetization')
#plt.colorbar(contour, label='string order')
#plt.colorbar(contour, label='entanglement entropy')
plt.colorbar(contour, label='fidelity')

# Label axes
plt.xlabel(r'$D$')
plt.ylabel(r'$1/\alpha$')

# title
plt.title("$L=100$")


# Show plot
plt.show()

