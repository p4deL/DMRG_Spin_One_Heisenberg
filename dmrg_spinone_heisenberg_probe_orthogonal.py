import numpy as np
import sys
import time

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

import include.data_io as data_io
#from include.long_range_exp_spinone_anisotropy_heisenberg_chain import LongRangeSpinOneChain
from include.long_range_frust_spinone_anisotropy_heisenberg_chain import LongRangeSpinOneChain
#from include.long_range_exp_spinone_heisenberg_chain import LongRangeSpinOneChain
import include.utilities as utilities

from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

# Set the number of BLAS threads
#os.environ["OMP_NUM_THREADS"] = "1"    # For OpenMP (used by some BLAS implementations)
#os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS
#os.environ["MKL_NUM_THREADS"] = "1"       # For MKL
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For Accelerate (macOS)
#os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr, if used


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
        'mixer_params': {
            'amplitude': 1.e-4,  # FIXME: should be chosen larger than smallest svd vals kept
            'decay': 2.0,
            'disable_after': 10,
        },
        'trunc_params': {
            'svd_min': 1.e-6,
        },
        #'chi_max': 150,
        'chi_list': {
            1: 10,
            2: 20,
            3: 80,
            4: 100,
            6: 200,
            8: 300,
        },
        'max_E_err': 1.e-8,
        'max_S_err': 1.e-6,
        'norm_tol': 1.e-6,
        'max_sweeps': 30,
    }

    print(f"L={L}, D={D}, Jz={Jz}, Gamma={Gamma}, B={B}, n_exp={n_exp}, sz={1 if sz1_flag else 0}")

    psi_list = []
    energy_list = []
    obs_list = []
    tracking_obs_list = []
    svn_list = []


    # create spine one model
    M = LongRangeSpinOneChain(model_params)

    # create initial state
    product_state = [0, 2] * (L//2)  # initial state down = 0, 0 = 1, up = 2
    if sz1_flag:
        product_state[L//2-1] = 1

    # initial guess mps
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    # run dmrg
    #info = dmrg.run(psi, M, dmrg_params)
    #E = info['E']
    engine = TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = engine.run()
    info = {'sweep_statistics' : engine.sweep_stats}

    # calc observables for tracking convergence
    tracking_obs = utilities.calc_tracking_quantities(psi, info, dmrg_params)

    # calculate observables
    obs = utilities.calc_observables(psi)

    SvN = psi.entanglement_entropy()[(L-1)//2]

    psi_list.append(psi)
    energy_list.append(E)
    obs_list.append(obs)
    tracking_obs_list.append(tracking_obs)
    svn_list.append(SvN)


    N = 3
    psi_perp_list = []
    psi_perp_list.append(psi)
    for i in range(1,N):
        psi_perp = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        engine_perp = TwoSiteDMRGEngine(psi_perp, M, dmrg_params)
        engine_perp.init_env(orthogonal_to=psi_perp_list)
        E_perp, psi_perp = engine_perp.run()
        info_perp = {'sweep_statistics' : engine.sweep_stats}

        # calc observables for tracking convergence
        tracking_obs_perp = utilities.calc_tracking_quantities(psi_perp, info_perp, dmrg_params)
        # calculate observables
        obs_perp = utilities.calc_observables(psi_perp)

        SvN_perp = psi_perp.entanglement_entropy()[(L-1)//2]

        psi_list.append(psi_perp)
        energy_list.append(E_perp)
        obs_list.append(obs_perp)
        tracking_obs_list.append(tracking_obs_perp)
        svn_list.append(SvN_perp)

        psi_perp_list.append(psi_perp)


    print(f"E={np.array(energy_list)/L}")
    print(f"SvN={np.array(svn_list)}")
    #print(f"Delta={E_perp-E}")
    #print(f"overlap={np.abs(psi.overlap(psi_perp))}")

    return (energy_list, tracking_obs_list, obs_list, psi_list)


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
    info_list = dmrg_lr_spinone_heisenberg_finite(L=L, D=D, Jz=Jz, Gamma=Gamma, alpha=alpha, B=B, n_exp=n_exp, sz1_flag=sz1_flag)
    print("--- %s seconds ---" % (time.time() - start_time))

    ###########################
    # writing the data to files
    # unpack
    def save_stuff(base_name, info_psi):
        (E, tracking_obs, obs, psi) = info_psi
        nsweeps, chi, Px, Stot_sq = tracking_obs
        chi_limit, chi_max = chi
        # save tracking obs
        str_tracking_obs = ["gs_energy", "parity_x", "s_total", "chi_max", "nsweeps"]
        tracking_obs = [E, Px, Stot_sq, chi_max, nsweeps]
        data_io.write_observables_to_file(f"{base_name}_trackobs",str_tracking_obs, tracking_obs, L, alpha, D, Gamma, Jz, chi_limit)
        # save observables
        str_observables = ["SvN", "m_trans", "m_long", "str_order", "eff_str_order"]
        data_io.write_observables_to_file(f"{base_name}_obs", str_observables, list(obs), L, alpha, D, Gamma, Jz, chi_limit)

        # save correlators
        corr_pm, corr_mp, corr_zz, corr_str_order = utilities.calc_correlations(psi)
        str_correlators = ["pos", "corr_pm", "corr_mp", "corr_zz", "corr_str_order"]
        pos = np.arange(len(corr_pm))
        correlators = [pos, corr_pm, corr_mp, corr_zz, corr_str_order]
        data_io.write_correlations_to_file(base_name, str_correlators, correlators, L, alpha, D, Gamma, Jz, chi_limit)

        # save entropies
        entropies = utilities.calc_entropies(psi)
        str_entropies = ["pos", "SvN"]
        pos = np.arange(len(entropies))
        entropies = [pos, entropies]
        data_io.write_entropies_to_file(base_name, str_entropies, entropies, L, alpha, D, Gamma, Jz, chi_limit)

        # Mz as a function of sites
        mz = psi.expectation_value('Sz')
        str_mz = ["pos", "m_long"]
        pos = np.arange(len(mz))
        mzs = [pos, mz]
        data_io.write_mz_to_file(base_name, str_mz, mzs, L, alpha, D, Gamma, Jz, chi_limit)

        print(base_name)
        print(f"gs_energy={E}")
        print(f"SvN={obs[0]}")


    energy_list, tracking_obs_list, obs_list, psi_list = info_list


    for i, info in enumerate(zip(energy_list, tracking_obs_list, obs_list, psi_list)):
        base_name = f"spinone_heisenberg_psi{i}"
        save_stuff(base_name, info)

if __name__ == "__main__":
   import logging
   logging.basicConfig(level=logging.INFO)
   main(sys.argv[1:])
