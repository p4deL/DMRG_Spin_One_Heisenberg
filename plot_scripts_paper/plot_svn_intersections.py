import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec
from matplotlib import rc, rcParams
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


#matplotlib.rcParams['text.usetex'] = True
rc('text', usetex=True)
rc('text.latex', preamble = r'\usepackage{amssymb}')
rcParams['pgf.preamble'] = r"\usepackage{amssymb}"

phase_diag = "D_alpha_observables"
#phase_diag = "lambda_alpha_observables"
#phase_diag = "Jz_alpha_observables"

probe_idx = 1
chi = 500


variable_str = "D"
fixed_str = "alpha"
#fixed_val = float("inf")
#fixed_val = 4.0
#fixed_val = float("inf")
fixed_val = 5.0

D_c = 2.5


# font sizes
fs1 = 16
fs2 = 12

xlabels_dict = {
    "D": "$D$",
    "alpha": "$\\alpha$"
}

# directory and filename
data_dir = f'../data/fss/gaussian_transition/'
#data_dir = f'../output/svn_min/'

file_pattern_sz0_dict = {
    "D": os.path.join(f"{data_dir}/alpha{fixed_val}/Sz0/", f'spinone_heisenberg_obs_chi{chi}_alpha{fixed_val}_*.csv'),
    "alpha": os.path.join(f"{data_dir}/D{fixed_val}/Sz0/", f'spinone_heisenberg_obs_chi{chi}_D{fixed_val}_*.csv')
}
file_pattern_sz1_dict = {
    "D": os.path.join(f"{data_dir}/alpha{fixed_val}/Sz1/", f'spinone_heisenberg_obs_chi{chi}_alpha{fixed_val}_*.csv'),
    "alpha" : os.path.join(f"{data_dir}/D{fixed_val}/Sz1/", f'spinone_heisenberg_obs_chi{chi}_D{fixed_val}_*.csv')
}

output_file_dict = {
    "D" : f"../plots/paper/SvN_mins_alpha{fixed_val}.pdf",
    "alpha" : f"../plots/paper/SvN_mins_D{fixed_val}.pdf"
}

xlabel = xlabels_dict.get(variable_str)
file_pattern_sz0 = file_pattern_sz0_dict.get(variable_str)
file_pattern_sz1 = file_pattern_sz1_dict.get(variable_str)
output_file = output_file_dict.get(variable_str)


def read_data(file_pattern, variable_str):

    svn_list = []

    csv_files = glob.glob(file_pattern)

    print(file_pattern)

    # Lists to hold the data
    D_values = []
    L_values = []
    z_values = []

    # Iterate through all the files
    for file in csv_files:

        # Read the CSV file
        data = pd.read_csv(file)

        L = float(file.split('_L')[-1].split(f'.csv')[0])  # Extract alpha from filename

        # Append data to lists
        combined = list(zip(data[variable_str].values, data["SvN"].values))

        sorted_combined = sorted(combined)
        d, z = zip(*sorted_combined)

        D_values.append(d)
        z_values.append(z)
        L_values.append(L)

        # sort by alpha values
        #if phase_diag == "lambda_alpha_observables":
        #    D_values = np.reciprocal(D_values)

    # sort colorplot vals
    combined = zip(L_values, D_values, z_values)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    L_values, D_values, z_values = zip(*sorted_combined)

    # Convert lists to arrays for plotting
    D_values = np.array(D_values)
    L_values = np.array(L_values)
    z_values = np.array(z_values)

    return L_values, D_values, z_values

def get_intersections_old(x1, y1, x2, y2):
    num_systems = np.shape(x1)[0]

    intersections = np.full(num_systems, np.nan)

    for i in range(num_systems):
        x1_i, y1_i = x1[i], y1[i]
        x2_i, y2_i = x2[i], y2[i]

        # Define common x-grid over overlapping region
        x_min = max(np.min(x1_i), np.min(x2_i))
        x_max = min(np.max(x1_i), np.max(x2_i))
        x_common = np.linspace(x_min, x_max, 1000)

        # Interpolate y1 and y2 to common x
        y1_interp = np.interp(x_common, x1_i, y1_i)
        y2_interp = np.interp(x_common, x2_i, y2_i)

        # Find difference and where it changes sign
        diff = y1_interp - y2_interp
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) == 0:
            # No intersection
            continue

        idx = sign_changes[-1]  # Take first crossing
        x0, x1_ = x_common[idx], x_common[idx + 1]
        y0, y1_ = diff[idx], diff[idx + 1]

        # Linear interpolation for root of y1 - y2 = 0
        x_cross = x0 - y0 * (x1_ - x0) / (y1_ - y0)
        intersections[i] = x_cross


    print("Intersection x-values:", intersections)
    return intersections


def get_intersections(x1, y1, x2, y2):
    num_systems = np.shape(x1)[0]

    intersections_x = np.full(num_systems, np.nan)
    intersections_y = np.full(num_systems, np.nan)

    for i in range(num_systems):
        x1_i, y1_i = x1[i], y1[i]
        x2_i, y2_i = x2[i], y2[i]

        # Define common x-grid over overlapping region
        x_min = max(np.min(x1_i), np.min(x2_i))
        x_max = min(np.max(x1_i), np.max(x2_i))
        x_common = np.linspace(x_min, x_max, 1000)

        # Interpolate y1 and y2 to common x
        y1_interp = np.interp(x_common, x1_i, y1_i)
        y2_interp = np.interp(x_common, x2_i, y2_i)

        # Find where y1 - y2 changes sign
        diff = y1_interp - y2_interp
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) == 0:
            continue  # No intersection

        idx = sign_changes[-1]  # Take last crossing (change if you want first)
        x0, x1_ = x_common[idx], x_common[idx + 1]
        y0, y1_ = diff[idx], diff[idx + 1]

        # Linear interpolation for root of y1 - y2 = 0
        x_cross = x0 - y0 * (x1_ - x0) / (y1_ - y0)
        y_cross = np.interp(x_cross, x1_i, y1_i)  # or y2_i

        intersections_x[i] = x_cross
        intersections_y[i] = y_cross

    print("Intersection x-values:", intersections_x)
    print("Intersection y-values:", intersections_y)
    return intersections_x, intersections_y


if __name__ == "__main__":
    L_values_sz0, D_values_sz0, svn_values_sz0 = read_data(file_pattern_sz0, variable_str)
    L_values_sz1, D_values_sz1, svn_values_sz1 = read_data(file_pattern_sz1, variable_str)
    print(f"Ls: {np.shape(L_values_sz0)}, Ds: {np.shape(D_values_sz0)}, svns: {np.shape(svn_values_sz0)}")
    print(f"Ls: {np.shape(L_values_sz1)}, Ds: {np.shape(D_values_sz1)}, svns: {np.shape(svn_values_sz1)}")

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    ax1, ax2 = axs

    intersections_x, intersections_y = get_intersections(D_values_sz0, svn_values_sz0, D_values_sz1, svn_values_sz1)

    for idx, (Ds, svn) in enumerate(zip(D_values_sz0, svn_values_sz0)):
        ax1.plot(Ds, svn, marker='o', color='#457b9d', alpha=(idx+1)/len(L_values_sz0))

    cmap = plt.cm.winter
    n_lines = len(L_values_sz1)
    colors_sz1 = cmap(np.linspace(0, 1, n_lines))

    for idx, (Ds, svn) in enumerate(zip(D_values_sz1, svn_values_sz1)):
        ax1.plot(Ds, svn, marker='o', color='#81b29a', alpha=(idx+1)/len(L_values_sz1))

    #ax1.set_xlim(0.99,1.04)
    #ax1.set_ylim(1.5,1.6)
    ax1.scatter(intersections_x, intersections_y, marker='x', s=50, lw=2.5, color='#bc4749', zorder=3)
    ax1.set_xlabel(xlabel, fontsize=fs1)

    #ax2.set_xlabel('$\\log(L)$', fontsize=fs1)
    ax1.set_ylabel('$S_{\\rm VN}$', fontsize=fs1)

    inv_Ls = np.reciprocal(L_values_sz0)
    ax2.scatter(inv_Ls, intersections_x, marker='x', s=50, lw=2.5, color='#bc4749')
    ax2.set_xlim([0, 0.02])
    ax2.set_xlabel('$1/L$', fontsize=fs1)
    ax2.set_ylabel('$D_c$', fontsize=fs1)

    popt, pcov = curve_fit(lambda L, a, b, c: a + b*L**(-c), L_values_sz0[0:], intersections_x[0:], p0=[1.0,-1.0, 1.5])
    #popt, pcov = curve_fit(lambda x, a, b: a - x**b, inv_Ls, intersections, p0=[1.0, 1.5])
    a, b, c = popt  # coefficients
    #a, b = popt  # coefficients
    da = np.sqrt(pcov[0,0])  # standard deviations (errors)
    db = np.sqrt(pcov[1,1])  # standard deviations (errors)
    dc = np.sqrt(pcov[2,2])  # standard deviations (errors)
    Lmin_fit = 60
    x_fit = np.linspace(1 / Lmin_fit, 0, 200)
    y_fit = a + b*x_fit**c
    #y_fit = a - x_fit**b
    ax2.plot(x_fit, y_fit, linestyle='--', color="gray")
    print("alpha,x_c,dx_c")
    print(f"{fixed_val},{a},{da}")

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax1.text(0.87, 0.95, f"$\\alpha={fixed_val}$", transform=ax1.transAxes, fontsize=12, verticalalignment='center', bbox=props)

    resstr = xlabel + '$_c=%.6f$' % (a, ) + '$\\pm %.6f$' % (da, )
    ax2.text(0.02, 0.05, resstr, transform=ax2.transAxes, fontsize=12, verticalalignment='center', bbox=props)


    # title
    #ax1.set_title(f"$L_1={L1}$,~$L_2={L2}$", fontsize=fs2)

    # save fig
    plt.tight_layout()
    plt.savefig(output_file)

    # Show plot
    plt.show()
