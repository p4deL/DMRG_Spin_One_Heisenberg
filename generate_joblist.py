import numpy as np
from tenpy.tools.fit import sum_of_exp, plot_alg_decay_fit #, fit_with_sum_of_exp
import matplotlib.pyplot as plt
import math
import os


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


def write_joblist_files(basename, script, L, alpha, Ds, n_exp):
    # Writing to file
    filename = f"{basename}_L{L}_alpha{alpha}.txt"
    with open(filename, "w") as file:
        for D in Ds:
            line = f"python {script} -L {L} -D {D} -a {alpha} -e {n_exp}\n"
            file.write(line)

    print(f"Output written to {filename}")


def write_one_joblist_file(basename, script, L, alpha, Ds, n_exp):
    # Writing to file
    filename = f"{basename}_L{L}.txt"
    with open(filename, "a") as file:
        for D in Ds:
            line = f"python {script} -L {L} -D {D} -a {alpha} -e {n_exp}\n"
            file.write(line)

    print(f"Output written to {filename}")

if __name__ == "__main__":

    #######################################################
    # Parameters
    script = "dmrg_spinone_heisenberg.py"
    basename = "joblist_dmrg"
    plotflag = False
    L = 100
    Ds = np.arange(-1.0, 0.5, 0.02)
    err_tol = 1e-9
    alphas = np.reciprocal(np.arange(0.0, 0.82, 0.02))
    print(alphas)
    #######################################################

    n_exp_min = n_exp = 1
    for alpha in alphas:
        print(alpha)
        if math.isinf(alpha):
            write_joblist_files(basename, script, L, alpha, Ds, n_exp)
        else:
            n_exp = determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
            n_exp_min = n_exp
            write_joblist_files(basename, script, L, alpha, Ds, n_exp)


        write_one_joblist_file(basename, script, L, alpha, Ds, n_exp)

