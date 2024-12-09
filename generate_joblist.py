import math
import numpy as np

import include.utilities as utilities
import include.data_io as data_io


if __name__ == "__main__":

    #######################################################
    # Parameters
    script = "dmrg_spinone_heisenberg.py"
    basename = "joblist_dmrg"
    plotflag = False
    L = 12
    #################
    # D phase diagram
    #Gammas = np.array([1.0])
    #Ds = np.arange(-0.4, 1.0, 0.02)
    #alphas = np.reciprocal(np.arange(0.0, 0.82, 0.02))
    # adaptive D phase diagram
    #Gammas = np.array([1.0])
    #Ds = np.arange(-0.4, 1.0, 0.02)
    #alphas1 = np.reciprocal(np.arange(0.0, 0.42, 0.02))
    #alphas2 = np.reciprocal(np.arange(0.45, 0.85, 0.05))
    #alphas = np.concatenate((alphas1, alphas2))
    # lambda phase diagram
    Gammas = np.array([1.0])
    lambdas = np.arange(0.02, 1.02, 0.02)
    Ds = np.reciprocal(lambdas)
    alphas = np.reciprocal(np.arange(0.0, 0.82, 0.02))
    #lambdas = np.arange(0.01, 1.01, 0.01)
    #Ds = np.reciprocal(lambdas)
    #alphas = np.array([1.25,2.5])
    # Gamma phase diagram
    #Ds = np.array([0.0])
    #Gammas = np.arange(0.0, 2.0, 0.05)
    #alphas = np.reciprocal(np.arange(0.0, 0.82, 0.02))
    #################

    err_tol = 1e-9
    #######################################################

    n_exp_min = n_exp = 1
    for i, alpha in enumerate(alphas):
        print(alpha)
        if math.isinf(alpha):
            data_io.write_joblist_files(basename, script, L, alpha, Ds, Gammas, n_exp)
        else:
            n_exp = utilities.determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
            n_exp_min = n_exp
            data_io.write_joblist_files(basename, script, L, alpha, Ds, Gammas, n_exp)


        data_io.write_one_joblist_file(f"{basename}_all_L{L}.txt", script, L, alpha, Ds, Gammas, n_exp, append=i)


