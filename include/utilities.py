import matplotlib.pyplot as plt
import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.tools.fit import sum_of_exp


def power_law_decay(dist, alpha):
    return 1. / dist ** alpha


def fit_with_sum_of_exp(f, alpha, n, N=50):
    r"""Approximate a decaying function `f` with a sum of exponentials.

    MPOs can naturally represent long-range interactions with an exponential decay.
    A common technique for other (e.g. powerlaw) long-range interactions is to approximate them
    by sums of exponentials and to include them into the MPOs.
    This function allows to do that.

    The algorithm/implementation follows the appendix of :cite:`murg2010`.

    Parameters
    ----------
    f : function
        Decaying function to be approximated. Needs to accept a 1D numpy array `x`
    n : int
        Number of exponentials to be used.
    N : int
        Number of points at which to evaluate/fit `f`;
        we evaluate and fit `f` at the points ``x = np.arange(1, N+1)``.

    Returns
    -------
    lambdas, prefactors: 1D arrays
        Such that :math:`f(k) \approx \sum_i x_i \lambda_i^k` for (integer) 1 <= `k` <= `N`.
        The function :func:`sum_of_exp` evaluates this for given `x`.
    """
    assert n < N
    x = np.arange(1, N + 1)
    f_x = f(x, alpha)
    F = np.zeros([N - n + 1, n])
    for i in range(n):
        F[:, i] = f_x[i:i + N - n + 1]

    U, V = np.linalg.qr(F)
    U1 = U[:-1, :]
    U2 = U[1:, :]
    M = np.dot(np.linalg.pinv(U1), U2)
    lam = np.linalg.eigvals(M)
    lam = np.sort(lam)[::-1]
    # least-square fit
    W = np.power.outer(lam, x).T
    pref, res, rank, s = np.linalg.lstsq(W, f_x, None)
    return lam, pref


def plot_fit(x_vals, y_powerlaw, y_sumexp):
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_powerlaw, marker='o', label='Power law')
    plt.plot(x_vals, y_sumexp, marker='x', label='sum exp')
    plt.legend(loc='best')
    plt.show()


def determine_n_exp(n_exp_min=1, err_tol=1e-10, L=1000, alpha=100.0, plot=False):
    n_exp = n_exp_min
    n_exp_max = L//2
    fit_range = L

    while n_exp_min < n_exp_max:
        lam, pref = fit_with_sum_of_exp(power_law_decay, alpha, n_exp, L)
        x = np.arange(1, fit_range + 1)
        err = np.sum(np.abs(power_law_decay(x, alpha) - sum_of_exp(lam, pref, x)))

        if err < err_tol:
            print("*"*100)
            print(f"alpha = {alpha}, n_exp = {n_exp}")
            print('error in fit: {0:.3e}'.format(err))
            if plot:
                plot_fit(x, power_law_decay(x, alpha), sum_of_exp(lam, pref, x) )
            break

        n_exp += 1

    return n_exp


def calc_tracking_quantities(psi, info, dmrg_params):
    # sweeps
    nsweeps = len(info['sweep_statistics']['sweep'])
    #max_sweeps = dmrg_params['max_sweeps']

    # bond dimensions
    chi_max = max(psi.chi)
    _ , chi_limit = list(dmrg_params['chi_list'].items())[-1]  # FIXME this only works if there is indeed a chi list
    chi = (chi_limit, chi_max)

    # calculate x- and z-parity
    id = psi.sites[0].Id
    Sz2 = psi.sites[0].multiply_operators(['Sz','Sz'])
    rotz = id - 2*Sz2
    Px = psi.expectation_value_multi_sites([rotz]*psi.L, 0)

    # Calculate S_tot^2
    corrzz = psi.correlation_function('Sz','Sz')
    corrpm = psi.correlation_function('Sp','Sm')
    corrmp = psi.correlation_function('Sm','Sp')
    Stot_sq = np.real(np.sum(np.triu(corrpm,k=1))+np.sum(np.triu(corrmp,k=1))+2*np.sum(np.triu(corrzz,k=1))) + 2*psi.L

    return nsweeps, chi, Px, Stot_sq


def calc_observables(psi):
    L = psi.L

    # von Neumann entanglement entropy
    SvN = psi.entanglement_entropy()[(L-1)//2]

    # transverse magnetization
    corr_pm = psi.correlation_function('Sp','Sm')
    corr_pm_stag = np.array([[(-1) ** (i + j) * corr_pm[i][j] for j in range(L)] for i in range(L)])
    mag_pm_stag = np.sqrt(np.mean(corr_pm_stag))
    mag_pm_stag_test = np.sqrt(1/L**2 * np.sum(corr_pm_stag))
    print(mag_pm_stag, mag_pm_stag_test)

    # longitudinal magnetization
    corr_zz = psi.correlation_function('Sz','Sz')
    corr_zz_stag = np.array([[(-1) ** (i + j) * corr_zz[i][j] for j in range(L)] for i in range(L)])
    mag_zz_stag = np.sqrt(np.mean(corr_zz_stag))

    # string_order parameter
    Sz = psi.sites[0].Sz
    exp = npc.expm(1.j * np.pi * Sz)
    corr_str_order = -1*psi.correlation_function("Sz", "Sz", opstr=exp, str_on_first=False)
    i = L//4
    j = i + L//2
    str_order = corr_str_order[i, j]
    eff_str_order = corr_str_order[i, j] - corr_zz[i, j]

    return SvN, mag_pm_stag, mag_zz_stag, str_order, eff_str_order

def calc_correlations(psi):
    corr_pm = psi.correlation_function('Sp','Sm')
    corr_mp = psi.correlation_function('Sm','Sp')
    corr_zz = psi.correlation_function('Sz','Sz')
    Sz = psi.sites[0].Sz
    exp = npc.expm(1.j * np.pi * Sz)
    corr_str_order = -1*psi.correlation_function("Sz", "Sz", opstr=exp, str_on_first=False)

    return corr_pm[0,:], corr_mp[0,:], corr_zz[0,:], corr_str_order[0,:]

def calc_entropies(psi):
    # von Neumann entanglement entropy
    SvN_full = psi.entanglement_entropy()
    return SvN_full

def calc_log_fidelity(psi, psi_eps, eps):
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions
    return -2 * np.log(overlap) / (eps ** 2)  # fidelity susceptiblity


def calc_fidelity(psi, psi_eps, eps):
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions
    constant = 2/ (eps**2)
    return constant * (1-overlap)  # fidelity susceptiblity

