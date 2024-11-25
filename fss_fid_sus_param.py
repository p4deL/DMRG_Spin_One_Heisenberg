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

import include.data_io as data_io

alpha = 2.5
chi = 300
#sigma = float(fixed_sigma)
#koppa = np.maximum(1, 2./(3*sigma))
koppa = 1.
#print(koppa)

reciprocal_lambda = True

# global xc and nu guess
#obs_string = "fidelity"
#obs_string = "m_long"
obs_string = "m_trans"

cutoff_left = 0
cutoff_right = 0

if obs_string == "fidelity":
    ylabels = ("$\\chi_{\\rm fidelity}$", '$L^{-2/\\nu}\\chi_{\\rm fidelity}$')  # FIXME
elif obs_string == "m_long":
    ylabels = ("$M_{z}$", "$L^{\\beta/\\nu}M_{z}$")
else:
    ylabels = ("$M_{\\rm \\perp}$", "$L^{\\beta/\\nu}M_{\\rm \\perp}$")

if reciprocal_lambda:
    xlabels = ("$\\lambda$", "$L^{1/\\nu}(\\lambda-\\lambda_c)$")
else:
    xlabels = ("$D$", "$L^{1/\\nu}(D-D_c)$")

labels = (xlabels, ylabels)

data_path = f"data/fss/largeD_U(1)CSB_transition/alpha{alpha}/"
out_file = f"plots/fss_{obs_string}_alpha{alpha}.pdf"



def fss_fid_suscept_fit_func(data, x_c, invnu, exponent, *coefs):
    L = data[0,:]
    x = data[1,:]

    # "exponent" is dummy
    poly = 0.
    for power, coef in enumerate(coefs):
        poly += coef*(L**(invnu)*(x-x_c))**power

    return L**(2*invnu)*poly

def fss_mag_fit_func(data, x_c, koppanu, beta, *coefs):
    L = data[0,:]
    x = data[1,:]


    poly = 0.
    #poly += coefs[0]*(L**(1./nu)*(x-x_c))**1
    for power, coef in enumerate(coefs):
        poly += coef*(L**(koppanu)*(x-x_c))**power

    return L**(-1.*beta*koppanu)*poly


def perform_data_collapse(data, fit_func):

    tuning_param_guess = -0.31
    invnu_guess = 1.
    exponent_guess = 0.125
    #print(fss_mag_fit_func(data[:2,:], tuning_param_guess, beta_guess, nu_guess, 1,1,1,1))

    params, params_covariance = optimize.curve_fit(fit_func, data[:2,:], data[2,:], p0=[tuning_param_guess, invnu_guess, exponent_guess, 1, 1, 1, 1, 1, 1], maxfev=500000)
    #print(params)
    #ax.plot(L_range, params[1]*L_range**params[0] + params[2], c='C1')
    
    return params, params_covariance

def plot_data_collapse(out_file, data, dim, params, params_covariance, obs_string, labels):

    #fix, ax = plt.subplots()
    #ins_ax = ax.inset_axes([.3, .1, .45, .35])  # [x, y, width, height] w.r.t. ax
    fix, axs = plt.subplots(2,1, figsize = (8,10))
    ax = axs[0]
    ins_ax = axs[1]
    L = data[0,:]
    x = data[1,:]
    obs = data[2,:]
    x_c = params[0]
    invnu = params[1]
    beta = params[2]
    nu = 1/params[1]
    #print(f"koppa={koppa}")
    #print(f"invnu={invnu}")
    #print(f"beta={beta}")
    #print(f"nu={nu}")

    #exp = 0.5

    #ax.scatter(L**(1./nu)*(x-params[0]), L**(2*beta/nu)*obs, s=0.5)
    total_dim = len(data[0,:])
    n = total_dim//dim
    for i in range(1,n+1):
        start = (i-1)*dim
        end = i*dim
        #print(L[start:end])
        #invnu=1.0
        #nu = 1.0
        #beta = 0.125
        #x_c = 1.0
        if obs_string == "fidelity":
            ax.scatter(L[start:end]**(invnu)*(x[start:end]-x_c), L[start:end]**(-2*invnu)*obs[start:end], s=14, label=f'$L={int(L[start])}$')
        else:
            ax.scatter(L[start:end]**(invnu)*(x[start:end]-x_c), L[start:end]**(1*beta*invnu)*obs[start:end], s=14, label=f'$L={int(L[start])}$')

        ins_ax.plot(x[start:end], obs[start:end])


    (xlabel, xlabel_scaling), (ylabel, ylabel_scaling) = labels

    print("x_c: ", x_c, np.sqrt(params_covariance[0,0]))
    print("nu: ", nu, np.sqrt(params_covariance[1,1]))
    print("FIXME: covriance for nu")
    if obs_string != "fidelity":
        print("beta: ", beta, np.sqrt(params_covariance[2,2]))
        resstr = '\n'.join((xlabel + '$\\hspace{-0.5em}\\phantom{x}_c=%.4f$' % (x_c, ), '$\\nu=%.4f$' % (nu, ), '$\\beta=%.4f$' % beta, ))
    else:
        resstr = '\n'.join((xlabel +'$\\hspace{-0.5em}\\phantom{x}_c=%.4f$' % (x_c, ), '$\\nu=%.4f$' % (nu, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.025, 0.9, resstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)
    #ax.text(0.025, 0.9, resstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)


    #ax.set_xlabel('$L^{ {\\selectlanguage{greek}\\qoppa}/\\nu}(h-h_c)$' , fontsize=16)
    #ax.set_ylabel('$L^{2\\beta\\selectlanguage{greek}\\qoppa/\\nu}\\left\\langle M^2 \\right\\rangle_L$', fontsize=16)
    ax.set_xlabel(xlabel_scaling, fontsize=16)
    ax.set_ylabel(ylabel_scaling, fontsize=16)

    ins_ax.set_xlabel(xlabel, fontsize=16)
    ins_ax.set_ylabel(ylabel, fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="lower right")
    #ax.legend()

    plt.savefig(out_file, bbox_inches='tight')
    plt.show()


def main(argv):
    data, dim = data_io.read_fss_data(data_path, obs_string, alpha, chi, cutoff_l=cutoff_left, cutoff_r=cutoff_right, reciprocal=reciprocal_lambda)
    if obs_string == "fidelity":
        params, params_covariance = perform_data_collapse(data, fss_fid_suscept_fit_func)
    else:
        params, params_covariance = perform_data_collapse(data, fss_mag_fit_func)

    plot_data_collapse(out_file, data, dim, params, params_covariance, obs_string, labels)


if __name__ == "__main__":
    main(sys.argv[1:])
