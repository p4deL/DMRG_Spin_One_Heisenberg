import math
import numpy as np

import include.utilities as utilities
import include.data_io as data_io


if __name__ == "__main__":

    #######################################################
    # Parameters
    script = "dmrg_spinone_heisenberg.py"
    basename = "joblist_dmrg_fss"
    sz1_flag = False
    plotflag = False
    lambdaflag = False
    alpha = float('inf')
    D = 2.75
    #Ls = [60, 100, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    Ls = [16,32,48,64]
    alphacinf = 2.75  # guess for phase transition point
    koppa = 1.
    nu = 1.8
    nonuni_pref_range = 10. # 4.
    n_datapoints = 30
    err_tol = 1e-9
    #######################################################

    if lambdaflag:
        D = 1./D  # convert lambdas into Ds

    for i, L in enumerate(Ls):
        print(f"L={L}")
        alphamin = alphacinf - nonuni_pref_range*L**(-koppa/nu)
        alphamax = alphacinf + nonuni_pref_range*L**(-koppa/nu)
        alphamin = 1.1
        alphamax = 6.0
        alphas = np.linspace(alphamax, alphamin, n_datapoints)

        #print(f"xc={xc}")
        print(f"alphamin={alphamin}, alphamax={alphamax}")

        n_exp_min = n_exp = 1
        n_exps = []
        for alpha in alphas:
            n_exp = utilities.determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
            n_exp_min = n_exp
            n_exps.append(n_exp)

        data_io.write_alpha_joblist_files(basename, script, L, alphas, D, n_exps, sz1_flag)

        data_io.write_alpha_one_joblist_file(f"{basename}_all_D{D}.txt", script, L, alphas, D, n_exps, sz1_flag, append=i)
        data_io.write_alpha_fss_meta_data(D, alphacinf, nonuni_pref_range, koppa, nu, lambdaflag, sz1_flag)


