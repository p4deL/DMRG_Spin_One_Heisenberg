import math
import numpy as np

import include.utilities as utilities
import include.data_io as data_io


if __name__ == "__main__":

    #######################################################
    # Parameters
    script = "dmrg_spinone_heisenberg_fss.py"
    basename = "joblist_dmrg_fss"
    sz1_flag = False
    plotflag = False
    lambdaflag = True
    alpha = float('inf')
    Ls = [40, 50, 60]
    #Ls = [40, 60, 80, 100, 120, 140, 100, 240, 280, 320]
    #xcmin = 0.38
    xcinf = 0.385  # guess for phase transition point
    koppa = 1.
    nu = 1.
    nonuni_pref_range = 10. # 4.
    # should locate peak for min and max length, then scale appropriately with expected exponent
    #nonuni_pref_xc = (xcmin-xcinf)*Ls[0]**(koppa/nu)  # choose such that it coincides with Dcmin
    #xcs = [nonuni_pref_xc*L**(-koppa/nu) + xcinf for L in Ls]
    #print(xcs)
    n_datapoints = 80
    err_tol = 1e-9
    #######################################################

    n_exp_min = n_exp = 1
    #for i, (L, xc) in enumerate(zip(Ls, xcs)):
    for i, L in enumerate(Ls):
        print(f"L={L}")
        xmin = xcinf - nonuni_pref_range*L**(-koppa/nu)
        xmax = xcinf + nonuni_pref_range*L**(-koppa/nu)
        xs = np.linspace(xmin, xmax, n_datapoints)
        if lambdaflag:
            xs = np.reciprocal(xs)  # convert lambdas into Ds

        #print(f"xc={xc}")
        print(f"xmin={xmin}, xmax={xmax}")

        if math.isinf(alpha):
            data_io.write_joblist_files(basename, script, L, alpha, xs, n_exp, sz1_flag)
        else:
            n_exp = utilities.determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
            n_exp_min = n_exp
            data_io.write_joblist_files(basename, script, L, alpha, xs, n_exp, sz1_flag)

        data_io.write_one_joblist_file(f"{basename}_all_alpha{alpha}.txt", script, L, alpha, xs, n_exp, sz1_flag, append=i)




