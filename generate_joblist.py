import math
import numpy as np

import include.utilities as utilities
import include.data_io as data_io


if __name__ == "__main__":

    #######################################################
    # Parameters
    script = "dmrg_spinone_heisenberg_all.py"
    basename = "joblist_dmrg"
    plotflag = False
    L = 100
    Ds = np.arange(-1.0, 0.5, 0.02)
    err_tol = 1e-9
    alphas = np.reciprocal(np.arange(0.0, 0.82, 0.02))
    print(alphas)
    #######################################################

    n_exp_min = n_exp = 1
    for i, alpha in enumerate(alphas):
        print(alpha)
        if math.isinf(alpha):
            data_io.write_joblist_files(basename, script, L, alpha, Ds, n_exp)
        else:
            n_exp = utilities.determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
            n_exp_min = n_exp
            data_io.write_joblist_files(basename, script, L, alpha, Ds, n_exp)


        data_io.write_one_joblist_file(f"{basename}_all_L{L}.txt", script, L, alpha, Ds, n_exp, append=i)


