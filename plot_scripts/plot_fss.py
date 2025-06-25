import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import use, rc, rcParams
rc('text', usetex=True)
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

alpha = 2.75
chi = 500
#sigma = float(fixed_sigma)
#koppa = np.maximum(1, 2./(3*sigma))
koppa = 1.
#print(koppa)
L_min = 100
L_mins = [60, 100, 140, 180, 200, 220, 240, 260]

variable_str = "D"
loop = True

# global xc and nu guess
#obs_string = "fidelity"
#obs_string = "m_long"
obs_string = "m_trans"

# data collapse guess ####
tuning_param_guess = 0.488
#x_c = tuning_param_guess
#dx_c = 1.e-12
#1.25,0.05628459055566769,2.113150896893074e-6,0
#1.5,0.10794923331584483,0.00007044664706561662,
#1.75,0.16182594579522097,0.00014840898296116826
#2.0,0.22192787344511145,0.00011423184989786802,
#2.25,0.29204226094403873,0.0006668923477267245,
#2.5,0.3794484555760642,0.00021543873275218614,1
#2.75,0.48809099656271715,0.005777484463677801,1
invnu_guess = 1. / 1.3
exponent_guess = 0.25
guess = (tuning_param_guess, invnu_guess, exponent_guess)

red_n_points = 0
cutoff_left = red_n_points//2
cutoff_right = red_n_points//2

ylabels_dict = {
    "fidelity": ("$\\chi_{\\rm fidelity}$", "$L^{-\\mu}\\chi_{\\rm fidelity}$"),
    "m_long": ("$M_{z}$", "$L^{\\beta/\\nu}M_{z}$"),
    "m_trans":  ("$M_{\\rm \\perp}$", "$L^{\\beta/\\nu}M_{\\rm \\perp}$"),
}

xlabels_dict = {
    "lambda": ("$\\lambda$", "$L^{1/\\nu}(\\lambda-\\lambda_c)$"),
    "D": ("$D$", "$L^{1/\\nu}(D-D_c)$"),
    "Gamma": ("$\\Gamma$", "$L^{1/\\nu}(\\Gamma-\\Gamma_c)$"),
    "alpha": ("$M_{\\rm \\perp}$", "$L^{\\beta/\\nu}M_{\\rm \\perp}$"),
}

ylabels = ylabels_dict.get(obs_string)
xlabels = xlabels_dict.get(variable_str)
labels = (xlabels, ylabels)

data_path = f"../data/fss/largeD_U(1)CSB_transition/alpha{alpha}/"
#data_path = f"../data/fss/ising_transition/alpha{alpha}/"
out_file = f"../plots/fss/fss_{obs_string}_alpha{alpha}.pdf"
out_data_file = f"../data/fss/largeD_U(1)CSB_transition/alpha{alpha}/data_collapse_{obs_string}_alpha{alpha}.csv"

print(data_path)

def fss_fid_suscept_fit_func(data, x_c, invnu, mu, *coefs):
    L = data[0,:]
    x = data[1,:]

    # "exponent" is dummy
    poly = 0.
    for power, coef in enumerate(coefs):
        poly += coef*(L**(invnu)*(x-x_c))**power

    return L**(mu)*poly

#def fss_mag_fit_func(data, x_c, koppanu, beta, *coefs):
def fss_mag_fit_func(data, x_c, koppanu, beta, *coefs):

    L = data[0,:]
    x = data[1,:]
    poly = 0.
    #poly += coefs[0]*(L**(1./nu)*(x-x_c))**1
    for power, coef in enumerate(coefs):
        poly += coef*(L**(koppanu)*(x-x_c))**power

    return L**(-1.*beta*koppanu)*poly


def perform_data_collapse(data, fit_func, guess):
    tuning_param_guess, invnu_guess, exponent_guess = guess

    params, params_covariance = optimize.curve_fit(fit_func, data[:2,:], data[2,:], p0=[tuning_param_guess, invnu_guess, exponent_guess, 1, 1, 1, 1, 1, 1], maxfev=500000)
    #params, params_covariance = optimize.curve_fit(fit_func, data[:2,:], data[2,:], p0=[invnu_guess, exponent_guess, 1, 1, 1, 1, 1, 1], maxfev=500000)

    #print(fss_mag_fit_func(data[:2,:], tuning_param_guess, beta_guess, nu_guess, 1,1,1,1))
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
    dx_c = np.sqrt(params_covariance[0,0])
    invnu = params[1]
    dinvnu = np.sqrt(params_covariance[1,1])
    #invnu = params[0]
    #dinvnu = np.sqrt(params_covariance[0,0])
    #exp = params[1]
    #dexp = np.sqrt(params_covariance[1,1])
    exp = params[2]
    dexp = np.sqrt(params_covariance[2,2])
    nu = 1/params[1]
    dnu = dinvnu/nu**2
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
            ax.scatter(L[start:end]**(invnu)*(x[start:end]-x_c), L[start:end]**(-exp)*obs[start:end], s=14, label=f'$L={int(L[start])}$')
        else:
            ax.scatter(L[start:end]**(invnu)*(x[start:end]-x_c), L[start:end]**(exp*invnu)*obs[start:end], s=14, label=f'$L={int(L[start])}$')

        ins_ax.plot(x[start:end], obs[start:end])


    (xlabel, xlabel_scaling), (ylabel, ylabel_scaling) = labels

    print(f"x_c = {x_c:.6f}±{dx_c:.6f}")
    print(f"nu = {nu:.6f}±{dnu:.6f}")
    if obs_string != "fidelity":
        print(f"beta = {exp:.6f}±{dexp:.6f}")
        resstr = '\n'.join((xlabel + '$\\hspace{-0.5em}\\phantom{x}_c=%.4f$' % (x_c, ), '$\\nu=%.4f$' % (nu, ), '$\\beta=%.4f$' % exp, ))
    else:
        print(f"mu = {exp:.6f}±{dexp:.6f}")
        resstr = '\n'.join((xlabel +'$\\hspace{-0.5em}\\phantom{x}_c=%.4f$' % (x_c, ), '$\\nu=%.4f$' % (nu, ), '$\\mu=%.4f$' % exp, ))

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

    return x_c, dx_c, nu, dnu, exp, dexp


def main(argv):
    if loop:
        for L in L_mins:
            data, dim = data_io.read_fss_data(data_path, obs_string, variable_str, alpha, chi, L_min=L, cutoff_l=cutoff_left,
                                          cutoff_r=cutoff_right)
            if obs_string == "fidelity":
                params, params_covariance = perform_data_collapse(data, fss_fid_suscept_fit_func, guess)
            else:
                params, params_covariance = perform_data_collapse(data, fss_mag_fit_func, guess)

            x_c, dx_c, nu, dnu, exp, dexp = plot_data_collapse(out_file, data, dim, params, params_covariance, obs_string, labels)
            data_io.write_data_collapse_to_file(out_data_file, red_n_points, L, x_c, dx_c, nu, dnu, exp, dexp)


    else:
        data, dim = data_io.read_fss_data(data_path, obs_string, variable_str, alpha, chi, L_min=L_min, cutoff_l=cutoff_left,
                                          cutoff_r=cutoff_right)
        if obs_string == "fidelity":
            params, params_covariance = perform_data_collapse(data, fss_fid_suscept_fit_func, guess)
        else:
            params, params_covariance = perform_data_collapse(data, fss_mag_fit_func, guess)

        x_c, dx_c, nu, dnu, exp, dexp = plot_data_collapse(out_file, data, dim, params, params_covariance, obs_string, labels)
        data_io.write_data_collapse_to_file(out_data_file, red_n_points, L_min, x_c, dx_c, nu, dnu, exp, dexp)

if __name__ == "__main__":
    main(sys.argv[1:])
