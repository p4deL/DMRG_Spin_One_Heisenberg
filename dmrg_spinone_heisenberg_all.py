import numpy as np
import sys
import time

from tenpy.algorithms import dmrg
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS

import include.data_io as data_io
from include.long_range_exp_spinone_heisenberg_chain import LongRangeSpinOneChain


# Set the number of BLAS threads
#os.environ["OMP_NUM_THREADS"] = "1"    # For OpenMP (used by some BLAS implementations)
#os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS
#os.environ["MKL_NUM_THREADS"] = "1"       # For MKL
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For Accelerate (macOS)
#os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr, if used


def calc_tracking_quantities(psi, info, dmrg_params):
    # sweeps
    sweeps = len(info['sweep_statistics']['sweep'])
    #max_sweeps = dmrg_params['max_sweeps']

    # bond dimensions
    chi_max = max(psi.chi)
    _ , chi_limit = list(dmrg_params['chi_list'].items())[-1]
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
    Stot_sq = (np.sum(np.triu(corrpm,k=1))+np.sum(np.triu(corrmp,k=1))+2*np.sum(np.triu(corrzz,k=1))) + 2*psi.L

    return sweeps, chi, Px, Stot_sq


def calc_observables(psi):
    L = psi.L

    # von Neumann entanglement entropy
    SvN = psi.entanglement_entropy()[(L-1)//2]

    # transverse magnetization
    corr_pm = psi.correlation_function('Sp','Sm')
    corr_pm_stag = np.array([[(-1) ** (i + j) * corr_pm[i][j] for j in range(L)] for i in range(L)])
    mag_pm_stag = np.sqrt(np.mean(corr_pm_stag))

    # longitudinal magnetization
    corr_zz = psi.correlation_function('Sz','Sz')
    corr_zz_stag = np.array([[(-1) ** (i + j) * corr_zz[i][j] for j in range(L)] for i in range(L)])
    mag_zz_stag = np.sqrt(np.mean(corr_zz_stag))

    # string_order parameter
    Sz = psi.sites[0].Sz
    Bk = npc.expm(1.j * np.pi * Sz)
    str_order = psi.correlation_function("Sz", "Sz", opstr=Bk, str_on_first=False)

    return SvN, mag_pm_stag, mag_zz_stag, str_order


def dmrg_lr_spinone_heisenberg_finite(L=10, alpha=10.0, D=0.0, n_exp=2, conserve='best'):
    model_params = dict(
        L=L,
        D=D,  # couplings
        B=0.0, # FIXME use parameterlist
        alpha=alpha,
        n_exp=n_exp,
        bc_MPS='finite',
        conserve=conserve)
    dmrg_params = {
        'mixer': False,  # TODO: Turn off mixer for large alpha!? For small alpha it may be worth a try increasing
        #'mixer_params': {
        #    'amplitude': 1.e-4,
        #    'decay': 2.0
        #    'disable_after': 12,
        #},
        'trunc_params': {
            'svd_min': 1.e-5,
        },
        #'chi_max': 150,
        'chi_list': {
            1: 10,
            2: 20,
            3: 80,
            4: 100,
            8: 200,
            10: 300,
            #8: 200,
            #    12: 400,
            #    16: 600,
        },
        'max_E_err': 1.e-9,
        'max_S_err': 1.e-7,
        'norm_tol': 5.e-7,
        'max_sweeps': 100,
    }


    # create spine one model
    M = LongRangeSpinOneChain(model_params)

    # create initial state
    if D <= 0.0 or alpha <= 3.0:  # FIXME: Check if boundary should be changed
        product_state = [0, 2] * (L//2)  # initial state down = 0, 0 = 1, up = 2
    else:
        product_state = [1] * L

    # initial guess mps
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    # run dmrg
    info = dmrg.run(psi, M, dmrg_params)
    data_io.log_sweep_statistics(L, alpha, D, info['sweep_statistics'])
    E = info['E']

    # calc observables for tracking convergence
    tracking_obs = calc_tracking_quantities(psi, info, dmrg_params)

    # calculate observables
    obs = calc_observables(psi)

    # save everything to a hdf5 file
    filename = f"output/data/dmrg_data_observables_alpha{alpha}_D{D}_L{L}.h5"
    data_io.save_results_obs(filename,  model_params=model_params,
                                    init_state=product_state,
                                    dmrg_params=dmrg_params,
                                    dmrg_info=info,
                                    mpo=M,
                                    mps=psi,
                                    observables=obs,
                                    tracking_observables=tracking_obs
                         )

    # output to check sanity
    print("E = {E:.13f}".format(E=info['E']))
    print("final bond dimensions psi: ", psi.chi)

    return E, tracking_obs, obs


def main(argv):

    ######################
    # read terminal inputs
    L, D, alpha, n_exp = data_io.param_use(argv)

    ##########
    # run dmrg
    start_time = time.time()
    E, tracking_obs, obs = dmrg_lr_spinone_heisenberg_finite(L=L, D=D, alpha=alpha, n_exp=n_exp)
    print("--- %s seconds ---" % (time.time() - start_time))

    ###########################
    # writing the data to files
    # unpack
    sweeps, chi, Px, Stot_sq = tracking_obs
    chi_limit, chi_max = chi
    # save tracking obs
    str_tracking_obs = ["gs_energy", "parity_x", "s_total", "nsweeps", "chi_max"]
    tracking_obs = [E, Px, Stot_sq, chi_max, chi_max]
    data_io.write_observables_to_file(str_tracking_obs, tracking_obs, L, alpha, D, chi_limit)
    # save observables
    str_observables = ["SvN", "m_trans", "m_long", "str_order"]
    data_io.write_observables_to_file(str_observables, list(obs), L, alpha, D, chi_limit)



if __name__ == "__main__":
   import logging
   logging.basicConfig(level=logging.INFO)
   main(sys.argv[1:])
