import os
import re
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
import pandas as pd
import sys


alpha = "inf"
chi = 300
#sigma = float(fixed_sigma)
#koppa = np.maximum(1, 2./(3*sigma))
koppa = 1.
#print(koppa)

# global xc and nu guess

data_path = f"output/fss/alpha{alpha}/"
out_file = f"plots/fss_fidelity_alpha{alpha}.pdf"
cutoff_left = 0
cutoff_right = 0


def read_data(path, obs_string, alpha, chi, cutoff_l=0, cutoff_r=0):
    """read data for given observable (as string) and return prepared data"""

    data_L = []
    data_tuning_param = []
    data_obs = []
    # iterate over files in path
    for f in os.listdir(path):
        # check if indeed is file
        print(f)
        if os.path.isfile(os.path.join(path, f)):
            # check if filename contains string of observable
            if obs_string in f:
                # extract system size from filename
                # FIXME: TRACKOBS doesn't work if there are other files...
                print(f"spinone_heisenberg_{obs_string}_obs_chi{chi}_alpha{alpha}_L(.*).csv")
                system_size = int(re.search(f"spinone_heisenberg_{obs_string}_obs_chi{chi}_alpha{alpha}_L(.*).csv", f).group(1))

                # import csv with panda
                df = pd.read_csv(path + f)
                data_array = df.to_numpy()
                #sigmas = data_array[:,0]
                # FIXME
                hs = data_array[:,0]
                obs = data_array[:,3] # FIXME do i need to multiply with L again?
                
                # TODO: only if necessary
                # prepare list with j as control fixed_param
                tmp = list(zip(hs, obs))
                tmp.sort(key=lambda x: x[0])
                sorted_hs = [tuples[0] for tuples in tmp]
                sorted_obs = [tuples[1] for tuples in tmp]
                l = len(sorted_hs)

                L = list(np.ones(len(hs[cutoff_l:l-cutoff_r]))*system_size)
                data_L += L
                data_tuning_param += sorted_hs[cutoff_l:l-cutoff_r]
                data_obs += sorted_obs[cutoff_l:l-cutoff_r]

                # TODO: WRITE DOWN WHAT'S GOING ON
                #L = list(np.ones(len(hs))*system_size)
                #data_L += L
                #data_tuning_param += sorted_hs
                #data_obs += sorted_obs

                print(len(data_L))
                print(len(data_tuning_param))

    dim = len(sorted_hs[cutoff_l:l-cutoff_r])
    data = np.stack((np.array(data_L), np.array(data_tuning_param), np.array(data_obs)))

  
    return data, dim


def fss_fid_suscept_fit_func(data, x_c, invnu, *coefs):
    L = data[0,:]
    x = data[1,:]

    poly = 0.
    #poly += coefs[0]*(L**(1./nu)*(x-x_c))**1
    for power, coef in enumerate(coefs):
        poly += coef*(L**(invnu)*(x-x_c))**power

    return L**(2*invnu)*poly



def perform_data_collapse(data):

    tuning_param_guess = -0.31
    invnu_guess = 1.
    #print(fss_mag_fit_func(data[:2,:], tuning_param_guess, beta_guess, nu_guess, 1,1,1,1))

    params, params_covariance = optimize.curve_fit(fss_fid_suscept_fit_func, data[:2,:], data[2,:], p0=[tuning_param_guess, invnu_guess, 1, 1, 1, 1, 1, 1], maxfev=500000)
    #print(params)
    #ax.plot(L_range, params[1]*L_range**params[0] + params[2], c='C1')
    
    return params, params_covariance

def plot_data_collapse(out_file, data, dim, params, params_covariance):

    fix, ax = plt.subplots()
    ins_ax = ax.inset_axes([.3, .1, .45, .35])  # [x, y, width, height] w.r.t. ax
    L = data[0,:]
    x = data[1,:]
    obs = data[2,:]
    x_c = params[0]
    invnu = params[1]
    nu = 1/params[1]
    print(f"koppa={koppa}")
    print(f"invnu={invnu}")
    print(f"nu={nu}")

    #ax.scatter(L**(1./nu)*(x-params[0]), L**(2*beta/nu)*obs, s=0.5)
    total_dim = len(data[0,:])
    n = total_dim//dim
    for i in range(1,n+1):
        start = (i-1)*dim
        end = i*dim
        #print(L[start:end])
        invnu=1.0
        #nu = 1.0
        #beta = 0.05
        #x_c = 1.0
        ax.scatter(L[start:end]**(invnu)*(x[start:end]-x_c), L[start:end]**(-2*invnu)*obs[start:end], s=14, label=f'$L={int(L[start])}$')
        ins_ax.plot(x[start:end], obs[start:end])


    print("x_c: ", x_c, np.sqrt(params_covariance[0,0]))
    #print("beta: ", beta, np.sqrt(params_covariance[1,1]))
    print("nu: ", nu, np.sqrt(params_covariance[2,2]))
    print("FIXME: covriance for nu")

    resstr = '\n'.join(('$h_c=%.4f$' % (x_c, ), '$\\nu=%.4f$' % (nu, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.025, 0.9, resstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)

    #ax.set_xlabel('$L^{ {\\selectlanguage{greek}\\qoppa}/\\nu}(h-h_c)$' , fontsize=16)
    #ax.set_ylabel('$L^{2\\beta\\selectlanguage{greek}\\qoppa/\\nu}\\left\\langle M^2 \\right\\rangle_L$', fontsize=16)
    ax.set_xlabel('$L^{1/\\nu}(D-D_c)$' , fontsize=16)
    ax.set_ylabel('$L^{-2/\\nu}\\chi_L$', fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="upper right")
    #ax.legend()

    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


def main(argv):
    data, dim = read_data(data_path, "fidelity", alpha, chi, cutoff_l=cutoff_left, cutoff_r=cutoff_right)
    params, params_covariance = perform_data_collapse(data)
    plot_data_collapse(out_file, data, dim, params, params_covariance)


if __name__ == "__main__":
    main(sys.argv[1:])
