import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import use, rc, rcParams
rc('text', usetex=True)
#rc('text.latex', preamble = r"\usepackage[greek, english]{babel}")
#rcParams['pgf.preamble'] = r"\usepackage[greek, english]{babel}"
#rcParams['pgf.preamble'] = r"\usepackage[polutonikogreek]{babel}"
#rc('text', usetex=True)
#rc('text.latex', preamble = r"\usepackage[greek, english]{babel}\usepackage{amsmath}")
#rcParams['pgf.preamble'] = r"\usepackage[greek, english]{babel}"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble = r"\usepackage[greek, english]{babel}")
rcParams['pgf.preamble'] = r"\usepackage[greek, english]{babel}"
from matplotlib import pyplot as plt
from scipy import optimize
import sys
import os

sys.path.insert(0, os.path.abspath('../'))
import include.data_io as data_io

alpha = 2.0
chi = 300
#sigma = float(fixed_sigma)
#koppa = np.maximum(1, 2./(3*sigma))
koppa = 1.
#print(koppa)
L_min = 0

# global xc and nu guess
#obs_string = "fidelity"
#obs_string = "m_long"
obs_string = "m_trans"

cutoff_left = 0
cutoff_right = 0

if obs_string == "fidelity":
    ylabel = "$\\chi_{\\rm fidelity}$"
elif obs_string == "m_long":
    ylabel = "$M_{z}$"
else:
    ylabel = "$M_{\\rm \\perp}$"

xlabel = "$\\Gamma$"

labels = (xlabel, ylabel)

data_path = f"../data/fss/haldane_SU(2)CSB_transition/alpha{alpha}/"
#data_path = f"../data/fss/ising_transition/alpha{alpha}/"
out_file = f"../plots/fss/fss_effects_{obs_string}_alpha{alpha}.pdf"

def plot_data_collapse(out_file, data, dim, labels):

    #fix, ax = plt.subplots()
    #ins_ax = ax.inset_axes([.3, .1, .45, .35])  # [x, y, width, height] w.r.t. ax
    fix, ax = plt.subplots(1,1, figsize=(6, 4))
    L = data[0,:]
    x = data[1,:]
    obs = data[2,:]

    total_dim = len(data[0,:])
    n = total_dim//dim
    for i in range(1,n+1):
        start = (i-1)*dim
        end = i*dim
        ax.plot(x[start:end], obs[start:end], label=f'$L={int(float(L[start]))}$')

    xlabel, ylabel = labels

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="lower right")
    ax.legend()

    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


def main(argv):
    data, dim = data_io.read_fss_data(data_path, obs_string, alpha, chi, L_min=L_min, cutoff_l=cutoff_left, cutoff_r=cutoff_right, reciprocal=False)
    plot_data_collapse(out_file, data, dim, labels)


if __name__ == "__main__":
    main(sys.argv[1:])
