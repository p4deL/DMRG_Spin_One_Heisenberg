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


# fixed val_ Either alpha or D
loop = False
variable_str = "alpha"
fixed_val = 0.4
#fixed_val = 1.6666666666666667
chi = 500

# global xc and nu guess
#obs_string = "fidelity"
#obs_string = "m_long"
obs_string = "m_trans"


#data_path = f"../data/fss/ising_transition/alpha{fixed_val}/"
#out_file = f"../plots/paper/fss_{obs_string}_alpha{fixed_val}.pdf"

#data_path = f"../data/fss/largeD_U(1)CSB_transition/alpha{fixed_val}/"
#out_file = f"../plots/paper/fss_{obs_string}_alpha{fixed_val}.pdf"

#data_path = f"../data/fss/largeD_U(1)CSB_transition/D{fixed_val}/"
#out_file = f"../plots/paper/fss_{obs_string}_D{fixed_val}.pdf"

data_path = f"../data/fss/haldane_U(1)CSB_transition/D{fixed_val}/"
out_file = f"../plots/paper/fss_{obs_string}_D{fixed_val}.pdf"

#data_path = f"../output/"
#out_file = f"../plots/paper/fss_{obs_string}_D{fixed_val}.pdf"

koppa = 1.
L_min = 60

red_n_points = 12
cutoff_left = red_n_points//2
cutoff_right = red_n_points//2

# data collapse guess ####
tuning_param_guess = 2.9
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

ylabels_dict = {
    "fidelity": ("$\\chi_{\\rm fidelity}$", "$L^{-\\mu}\\chi_{\\rm fidelity}$"),
    "m_long": ("$M_{z}$", "$L^{\\beta/\\nu}M_{z}$"),
    "m_trans":  ("$M_{\\rm \\perp}$", "$L^{\\beta/\\nu}M_{\\rm \\perp}$"),
}

xlabels_dict = {
    "lambda": ("$\\lambda$", "$L^{1/\\nu}(\\lambda-\\lambda_c)$"),
    "D": ("$D$", "$L^{1/\\nu}(D-D_c)$"),
    "Gamma": ("$\\Gamma$", "$L^{1/\\nu}(\\Gamma-\\Gamma_c)$"),
    "alpha": ("$\\alpha$", "$L^{1/\\nu}(\\alpha-\\alpha_c)$"),
}

ylabels = ylabels_dict.get(obs_string)
xlabels = xlabels_dict.get(variable_str)
labels = (xlabels, ylabels)

def hex_to_RGB(hex_str):
  """ #FFFFFF -> [255,255,255]"""
  #Pass 16 to the integer function for change of base
  return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


def fss_fid_suscept_fit_func(data, x_c, invnu, mu, *coefs):
    L = data[0,:]
    x = data[1,:]

    # "exponent" is dummy
    poly = 0.
    for power, coef in enumerate(coefs):
        poly += coef*(L**(invnu)*(x-x_c))**power

    return L**(mu)*poly

#def fss_mag_fit_func(data, koppanu, beta, *coefs):
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
    # params, params_covariance = optimize.curve_fit(fit_func, data[:2,:], data[2,:], p0=[invnu_guess, exponent_guess, 1, 1, 1, 1, 1, 1], maxfev=500000)

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

    # unbiased
    x_c = params[0]
    dx_c = np.sqrt(params_covariance[0,0])
    invnu = params[1]
    dinvnu = np.sqrt(params_covariance[1,1])
    exp = params[2]
    dexp = np.sqrt(params_covariance[2,2])
    # biased
    #invnu = params[0]
    #dinvnu = np.sqrt(params_covariance[0,0])
    #exp = params[1]
    #dexp = np.sqrt(params_covariance[1,1])

    nu = 1./invnu
    dnu = dinvnu/nu**2
    #print(f"koppa={koppa}")
    #print(f"invnu={invnu}")
    #print(f"beta={beta}")
    #print(f"nu={nu}")

    #exp = 0.5


    #ax.scatter(L**(1./nu)*(x-params[0]), L**(2*beta/nu)*obs, s=0.5)
    total_dim = len(data[0,:])
    n = total_dim//dim
    colors = get_color_gradient("#457b9d", "#81b29a", n)
    for i in range(1,n+1):
        start = (i-1)*dim
        end = i*dim
        #print(L[start:end])
        #invnu=1.0
        #nu = 1.0
        #beta = 0.125
        #x_c = 1.0
        if i == 1:
            label = '$L_{\\rm min}' + f'={int(L[start])}$'
        elif i == n:
            label = '$L_{\\rm max}' + f'={int(L[start])}$'
        else:
            label = None  # no legend entry

        ax.scatter(L[start:end]**(invnu)*(x[start:end]-x_c), L[start:end]**(exp*invnu)*obs[start:end], s=40, color=colors[i-1], alpha=0.5, label=label)

        ins_ax.plot(x[start:end], obs[start:end], color=colors[i-1], lw=3, alpha=0.75)

    (xlabel, xlabel_scaling), (ylabel, ylabel_scaling) = labels

    print(f"x_c = {x_c:.6f}±{dx_c:.6f}")
    print(f"nu = {nu:.6f}±{dnu:.6f}")
    print(f"beta = {exp:.6f}±{dexp:.6f}")

    resstr = '\n'.join((xlabel + '$_c=%.5f$' % (x_c, ) + '$\\pm %.5f$' % (dx_c, ), '$\\nu=%.4f$' % (nu, ) + '$\\pm %.4f$' % (dnu, ), '$\\beta=%.4f$' % (exp, ) + '$\\pm %.4f$' % (dexp, )))

    #resstr = xlabel + '$_c=%.6f$' % (a, ) + '$\\pm %.6f$' % (da, )
    #ax2.text(0.02, 0.05, resstr, transform=ax2.transAxes, fontsize=12, verticalalignment='center', bbox=props)


    #resstr = '\n'.join((xlabel + '$\\hspace{-0.5em}\\phantom{x}_c=%.4f$' % (x_c, ), '$\\nu=%.4f$' % (nu, ), '$\\beta=%.4f$' % exp, ))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.025, 0.1, resstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)
    #ax.text(0.025, 0.9, resstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)


    #ax.set_xlabel('$L^{ {\\selectlanguage{greek}\\qoppa}/\\nu}(h-h_c)$' , fontsize=16)
    #ax.set_ylabel('$L^{2\\beta\\selectlanguage{greek}\\qoppa/\\nu}\\left\\langle M^2 \\right\\rangle_L$', fontsize=16)
    ax.set_xlabel(xlabel_scaling, fontsize=16)
    ax.set_ylabel(ylabel_scaling, fontsize=16)

    ins_ax.set_xlabel(xlabel, fontsize=16)
    ins_ax.set_ylabel(ylabel, fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="upper right")
    #ax.legend()

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    return x_c, dx_c, nu, dnu, exp, dexp


def main(argv):
    data, dim = data_io.read_fss_data(data_path, obs_string, variable_str, fixed_val, chi, L_min=L_min, cutoff_l=cutoff_left,
                                      cutoff_r=cutoff_right)

    #data = data[data[:, 0].argsort()]
    data = data[:, data[0].argsort(kind="stable")] # sort by entries of the first row
    #print(np.shape(data))
    #print(data[:,0])

    params, params_covariance = perform_data_collapse(data, fss_mag_fit_func, guess)

    x_c, dx_c, nu, dnu, exp, dexp = plot_data_collapse(out_file, data, dim, params, params_covariance, obs_string, labels)
    print(x_c, dx_c, nu, dnu, exp, dexp)


if __name__ == "__main__":
    main(sys.argv[1:])
