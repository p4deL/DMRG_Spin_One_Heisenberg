import numpy as np
import sys
import time

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

import include.data_io as data_io
from include.long_range_exp_spinone_anisotropy_heisenberg_chain import LongRangeSpinOneChain
#from include.long_range_frust_spinone_anisotropy_heisenberg_chain import LongRangeSpinOneChain
#from include.long_range_exp_spinone_heisenberg_chain import LongRangeSpinOneChain
import include.utilities as utilities


import tenpy.tools.cache

# Set the number of BLAS threads
#os.environ["OMP_NUM_THREADS"] = "1"    # For OpenMP (used by some BLAS implementations)
#os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS
#os.environ["MKL_NUM_THREADS"] = "1"       # For MKL
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For Accelerate (macOS)
#os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr, if used
cache_dir = "tmp/cache"

def dmrg_lr_spinone_heisenberg_finite(L=10, alpha=10.0, D=0.0, Jz=1.0, Gamma=1.0, B=0.0, n_exp=2, sz1_flag=False, conserve='best'):
    model_params = dict(
        L=L,
        D=D,
        Jz=Jz,
        Gamma=Gamma,
        B=B,
        alpha=alpha,
        n_exp=n_exp,
        bc_MPS='finite',
        conserve=conserve)
    dmrg_params = {
        'mixer': False,  # TODO: Don't turn on if not stuck in Minima
        #'mixer_params': {
        #    'amplitude': 1.e-4,  # FIXME: should be chosen larger than smallest svd vals kept
        #    'decay': 2.0,
        #    'disable_after': 10,
        #},
        'trunc_params': {
            'svd_min': 1e-5,
        },
        #'chi_max': 300,
        'chi_list': {
            1: 10,
            2: 20,
            3: 80,
            4: 100,
            6: 200,
            8: 300,
            10: 500,
        },
        'max_E_err': 1.e-8,
        'max_S_err': 1.e-6,
        'norm_tol': 1.e-6,
        'max_sweeps': 25,
    }
    #cache_params = {
    #        'storage_class': 'PickleStorage',
    #        'use_threading': True,  # reduce the OMP_NUM_THREADS if you use this!
    #        'cache_threshold_chi': 500,  # use cache for chi larger than that
    #        'directory': 'tmp/cache',
    #        # specify tmpdir on the clusters node's local file system
    #    }

    print(f"L={L}, D={D}, Jz={Jz}, Gamma={Gamma}, B={B}, n_exp={n_exp}, sz={1 if sz1_flag else 0}")

    # create spine one model
    M = LongRangeSpinOneChain(model_params)

    # create initial state
    if D <= 0.0 or alpha <= 3.0:  # FIXME: Check if boundary should be changed
        product_state = [0, 2] * (L//2)  # initial state down = 0, 0 = 1, up = 2
        if sz1_flag:
            product_state[L//2-1] = 1
    else:
        product_state = [1] * L
        if sz1_flag:
            product_state[L//2-1] = 2


    # initial guess mps
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    # run dmrg
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']

    # calc observables for tracking convergence
    tracking_obs = utilities.calc_tracking_quantities(psi, info, dmrg_params)

    # calculate observables
    obs = utilities.calc_observables(psi)

    # save everything to a hdf5 file
    #filename = f"output/data/dmrg_data_observables_alpha{alpha}_D{D}_L{L}.h5"
    #data_io.save_results_obs(filename,  model_params=model_params,
    #                                init_state=product_state,
    #                                dmrg_params=dmrg_params,
    #                                dmrg_info=info,
    #                                mpo=M,
    #                                mps=psi,
    #                                observables=obs,
    #                                tracking_observables=tracking_obs
    #                     )

    # output to check sanity
    print("E = {E:.13f}".format(E=info['E']))
    print("final bond dimensions psi: ", psi.chi)

    return E, tracking_obs, obs, psi


def main(argv):

    ######################
    # read terminal inputs
    L, D, Jz, Gamma, alpha, n_exp, sz1_flag = data_io.param_use(argv)

    # AUXFIELD ?
    #B = 1.e-2  # FIXME
    B = 0.  # FIXME
    #if alpha > 3.0:
    #    B = 1e-2
    #else:
    #    B = 0.

    ##########################################################################################
    ### TODO:
    ### TODO: Benchmark different parts of the code with a timer for a large MPS
    ### TODO: It seems like saving the data to the file takes forever after DMRG is done
    ### TODO: Could also be other issue; maybe calculating obs or something...
    ### TODO:
    ##########################################################################################

    ##########
    # run dmrg
    start_time = time.time()
    E, tracking_obs, obs, psi = dmrg_lr_spinone_heisenberg_finite(L=L, D=D, Jz=Jz, Gamma=Gamma, alpha=alpha, B=B, n_exp=n_exp, sz1_flag=sz1_flag)
    print("--- %s seconds ---" % (time.time() - start_time))

    ###########################
    # writing the data to files
    # unpack
    nsweeps, chi, Px, Stot_sq = tracking_obs
    chi_limit, chi_max = chi
    # save tracking obs
    str_tracking_obs = ["gs_energy", "parity_x", "s_total", "chi_max", "nsweeps"]
    tracking_obs = [E, Px, Stot_sq, chi_max, nsweeps]
    data_io.write_observables_to_file("spinone_heisenberg_trackobs",str_tracking_obs, tracking_obs, L, alpha, D, Gamma, Jz, chi_limit)
    #data_io.write_observables_to_file_fix_D("spinone_heisenberg_trackobs",str_tracking_obs, tracking_obs, L, alpha, D, Gamma, Jz, chi_limit)
    # save observables
    str_observables = ["SvN", "m_trans", "m_long", "str_order", "eff_str_order"]
    data_io.write_observables_to_file("spinone_heisenberg_obs", str_observables, list(obs), L, alpha, D, Gamma, Jz, chi_limit)
    #data_io.write_observables_to_file_fix_D("spinone_heisenberg_obs", str_observables, list(obs), L, alpha, D, Gamma, Jz, chi_limit)

    # save entanglement spectrum
    #ee_spectrum = psi.entanglement_spectrum()[(L-1)//2][:10]
    #str_ee_spectrum = [f"chi{i}" for i in range(len(ee_spectrum))]
    #data_io.write_observables_to_file("spinone_heisenberg_ee_spectrum", str_ee_spectrum, list(ee_spectrum), L, alpha, D, Gamma, Jz, chi_limit)


if __name__ == "__main__":
   import logging
   logging.basicConfig(level=logging.INFO)
   main(sys.argv[1:])
