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
data_dir = 'output/L200/'

# Use glob to find all csv files that match the pattern
#file_pattern = os.path.join(data_dir, 'spinone_heisenberg_fidelity_alpha*.csv')
file_pattern = os.path.join(data_dir, 'spinone_heisenberg_svn_alpha*.csv')
#file_pattern = os.path.join(data_dir, 'spinone_heisenberg_stringorder_alpha*.csv')
#file_pattern = os.path.join(data_dir, 'spinone_heisenberg_magnetization_alpha*.csv')


csv_files = glob.glob(file_pattern)

# Lists to hold the data
D_values = []
alpha_values = []
z_values = []

# Iterate through all the files
for file in csv_files:
    # Extract alphaval from the filename
    alpha = float(file.split('_alpha')[-1].split('_L200.csv')[0]) # Extract alpha from filename
    print(alpha)
    # Read the CSV file
    #data = pd.read_csv(file, header=None, names=["D", "mag"])
    data = pd.read_csv(file)

    # Append data to lists
    #values = np.full_like(data["D"].values, alpha)
    #combined = list(zip(data["D"].values, values))
    #combined = list(zip(data["D"].values, data["fidelity"].values))
    combined = list(zip(data["D"].values, data["SvN"].values))
    #combined = list(zip(data["D"].values, data["str_order"].values))
    #combined = list(zip(data["D"].values, data["mag"].values))
    sorted_combined = sorted(combined)
    d, z = zip(*sorted_combined)

    D_values.append(d)
    #mag_values.append(data["fidelity"].values)
    #mag_values.append(data["SvN"].values)
    #mag_values.append(data["str_order"].values)
    #mag_values.append(data["mag"].values)
    z_values.append(z)
    alpha_values.append(np.full_like(data["D"].values, 1./alpha))  # Create an array of alphaval for each D

# sort values
#combined = list(zip(D_values, mag_values))
#print(combined)
#sorted_combined = sorted(combined)
#D_values, mag_values = zip(*sorted_combined)

combined = zip(alpha_values, D_values, z_values)
sorted_combined = sorted(combined, key=lambda x: x[0][0])
alpha_values, D_values, z_values = zip(*sorted_combined)


# Convert lists to arrays for plotting
D_values = np.concatenate(D_values)
z_values = np.concatenate(z_values)
alpha_values = np.concatenate(alpha_values)


#print(D_values)
print(alpha_values)
print(z_values)

# Now create a grid of D and alphaval values for contouring
D_grid, alpha_grid = np.meshgrid(np.unique(D_values), np.unique(alpha_values))
#mag_grid = np.meshgrid(np.unique(mag_values))


# Interpolate z_values onto the grid
#from scipy.interpolate import griddata
#mag_grid = griddata((D_values, alpha_values), mag_values, (D_grid, alpha_grid), method='linear')



# Create the contour plot
plt.figure(figsize=(8, 6))
#contour = plt.contourf(D_grid, alpha_grid, mag_grid, cmap='viridis', levels=20)
#print(z_values.reshape((9,16)))
#print(mag_grid)


#contour = plt.contourf(D_grid, alpha_grid, z_values.reshape((9,16)), cmap='viridis')
#contour = plt.contourf(D_grid, alpha_grid, mag_grid, cmap='viridis', levels=20)
#contour = plt.contourf(D_grid, alpha_grid, mag_grid, cmap='viridis')
colorplot = plt.pcolor(D_grid, alpha_grid, z_values.reshape(D_grid.shape), cmap='viridis')

# Add colorbar
#plt.colorbar(contour, label='fidelity')
#plt.colorbar(contour, label='entanglement entropy')
plt.colorbar(colorplot, label='string order')
#plt.colorbar(colorpolt, label='squared stag. magnetization')

# Label axes
plt.xlabel(r'$D$')
plt.ylabel(r'$1/\alpha$')

# title
plt.title("$L=200$")


# Show plot
plt.show()

