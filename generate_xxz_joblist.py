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
    L = 20
    sz1_flag = False
    #################
    # D phase diagram
    Jzs = np.arange(-0.95, 3.05, 0.05)
    #Ds = np.arange(-0.4, 1.0, 0.02)
    alphas = np.reciprocal(np.arange(0.0, 0.61, 0.01))
    #################

    err_tol = 1e-9
    #######################################################

    n_exp_min = n_exp = 1
    for i, alpha in enumerate(alphas):
        print(alpha)
        if math.isinf(alpha):
            data_io.write_xxz_joblist_files(basename, script, L, alpha, Jzs, n_exp, sz1_flag)
        else:
            n_exp = utilities.determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
            n_exp_min = n_exp
            data_io.write_xxz_joblist_files(basename, script, L, alpha, Jzs, n_exp, sz1_flag)


        data_io.write_one_xxz_joblist_file(f"{basename}_all_L{L}.txt", script, L, alpha, Jzs, n_exp, sz1_flag, append=i)


