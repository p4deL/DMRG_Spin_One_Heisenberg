import math
import numpy as np

import include.utilities as utilities
import include.data_io as data_io


if __name__ == "__main__":

    #######################################################
    # Parameters
    script = "dmrg_spinone_heisenberg_fss.py"
    basename = "joblist_dmrg_fss"
    plotflag = False
    alpha = float('inf')
    Ls = [60, 80, 100, 120, 140, 160, 180, 200]
    Dcmin = -0.25
    Dcinf = -0.31 # guess for phase transition point
    #Dcmax = -0.28
    koppa = 1.
    nu = 1.
    nonuni_pref_Dc = (Dcmin-Dcinf)*Ls[0]**(koppa/nu)  # choose such that it coincides with Dcmin
    nonuni_pref_range = 4.
    # should locate peak for min and max length, then scale appropriately with expected exponent
    Dcs = [nonuni_pref_Dc*L**(-koppa/nu) + Dcinf for L in Ls]
    #Dcs = np.linspace(Dcmin, Dcmax, len(Ls))  # Can also be used if nu=1
    n_datapoints = 25
    err_tol = 1e-9
    #alphas = np.reciprocal(np.arange(0.0, 0.82, 0.02))
    #print(alphas)
    #######################################################

    n_exp_min = n_exp = 1
    for i, (L, Dc) in enumerate(zip(Ls, Dcs)):
        print(f"L={L}")
        Dmin = Dc - nonuni_pref_range*L**(-koppa/nu)
        Dmax = Dc + nonuni_pref_range*L**(-koppa/nu)
        Ds = np.linspace(Dmin, Dmax, n_datapoints)
        print(f"Dc={Dc}")
        print(f"Dmin={Dmin}, Dmax={Dmax}")

        if math.isinf(alpha):
            data_io.write_joblist_files(basename, script, L, alpha, Ds, n_exp)
        else:
            n_exp = utilities.determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
            n_exp_min = n_exp
            data_io.write_joblist_files(basename, script, L, alpha, Ds, n_exp)

        data_io.write_one_joblist_file(f"{basename}_all_alpha{alpha}.txt", script, L, alpha, Ds, n_exp, append=i)




