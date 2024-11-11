# padelhardt
import numpy as np
import sys
import time

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

import include.data_io as data_io
from include.long_range_exp_spinone_heisenberg_chain import LongRangeSpinOneChain
import include.utilities as utilities


# Set the number of BLAS threads
#os.environ["OMP_NUM_THREADS"] = "1"    # For OpenMP (used by some BLAS implementations)
#os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS
#os.environ["MKL_NUM_THREADS"] = "1"       # For MKL
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For Accelerate (macOS)
#os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr, if used




def dmrg_lr_spinone_heisenberg_finite_fidelity(L=10, alpha=10.0, D=0.0, n_exp=2, eps=1e-3, conserve='best'):
    model_params = dict(
        L=L,
        D=D,  # couplings
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
        'max_E_err': 1.e-8,
        'max_S_err': 1.e-7,
        'norm_tol': 5.e-7,
        'max_sweeps': 100,
    }

    # TODO: Save METADATA!!!!!!!


    # create spine one model
    M = LongRangeSpinOneChain(model_params)
    model_params['D'] = D+eps
    M_eps = LongRangeSpinOneChain(model_params)

    # TODO: Do not delete me
    #print("."*100)
    #print(M.all_onsite_terms().to_TermList())
    #print(M.all_coupling_terms().to_TermList())
    #print(M.exp_decaying_terms.to_TermList())
    #print(alternating_power_law_decay(np.arange(1,7), alpha))
    #print("."*100)

    # create initial state
    if D <= 0.0 or alpha <= 3.0:  # FIXME: Check if boundary should be changed
        product_state = [0, 2] * (L//2)  # initial state down = 0, 0 = 1, up = 2
    else:
        product_state = [1] * L

    # initial guess mps
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    psi_eps = MPS.from_product_state(M_eps.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    # run dmrg
    info = dmrg.run(psi, M, dmrg_params)
    info_eps = dmrg.run(psi_eps, M_eps, dmrg_params)

    # TODO: save ground_state and results
    # TODO: save in subdirectory wavefunctions observables/ logs/ ~ think about good directory structure

    # ground-state energy
    E_psi = info['E']
    E_psi_eps = info_eps['E']
    delta_E = E_psi - E_psi_eps

    E = (E_psi, E_psi_eps)

    # log data
    data_io.log_sweep_statistics(L, alpha, D, info['sweep_statistics'])
    data_io.log_sweep_statistics(L, alpha, D+eps, info_eps['sweep_statistics'])

    # sweeps
    n_sweep_psi = len(info['sweep_statistics']['sweep'])
    n_sweep_psi_eps = len(info_eps['sweep_statistics']['sweep'])
    #max_sweeps = dmrg_params['max_sweeps']

    sweeps = (n_sweep_psi, n_sweep_psi_eps)

    # TODO: also plot max bond dimension
    # choose smaller svd cutoff to control max bond dimension in practice
    # make plots as function of max bond dimension vs max cutoff
    # lastly try slower ramp up of bond dimension
    # Should I plot bond dimension versus system size, Check area law for sufficiently short ranged interactions?
    # TODO: first
    # I should rather go back to TFIM -> Frust Spin-one chain and compare results again!

    # plot bond dimensions
    chi_max_psi = max(psi.chi)
    chi_max_psi_eps = max(psi_eps.chi)
    _ , chi_limit = list(dmrg_params['chi_list'].items())[-1]
    chi = (chi_limit, chi_max_psi, chi_max_psi_eps)

    # calculate x- and z-parity
    id = psi.sites[0].Id
    Sz2 = psi.sites[0].multiply_operators(['Sz','Sz'])
    rotz = id - 2*Sz2
    Px_psi = psi.expectation_value_multi_sites([rotz]*L, 0)
    Px_psi_eps = psi_eps.expectation_value_multi_sites([rotz]*L, 0)
    Px = (Px_psi, Px_psi_eps)

    # Calculate S_tot^2
    corrzz_psi = psi.correlation_function('Sz','Sz')
    corrpm_psi = psi.correlation_function('Sp','Sm')
    corrmp_psi = psi.correlation_function('Sm','Sp')
    Stot_sq_psi = (np.sum(np.triu(corrpm_psi,k=1))+np.sum(np.triu(corrmp_psi,k=1))+2*np.sum(np.triu(corrzz_psi,k=1))) + 2*L

    corrzz_psi_eps = psi_eps.correlation_function('Sz','Sz')
    corrpm_psi_eps = psi_eps.correlation_function('Sp','Sm')
    corrmp_psi_eps = psi_eps.correlation_function('Sm','Sp')
    Stot_sq_psi_eps = (np.sum(np.triu(corrpm_psi_eps,k=1))+np.sum(np.triu(corrmp_psi_eps,k=1))+2*np.sum(np.triu(corrzz_psi_eps,k=1))) + 2*L

    Stot_sq = (Stot_sq_psi, Stot_sq_psi_eps)

    # calculate fidelity
    fidelity = utilities.calc_fidelity(psi, psi_eps, eps)

    # calcualte overlap of wavefunctions
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions

    # output to check sanity
    print("E = {E:.13f}".format(E=info['E']))
    print("final bond dimensions psi: ", psi.chi)
    print("E_eps = {E:.13f}".format(E=info_eps['E']))
    print("final bond dimensions psi_eps: ", psi_eps.chi)

    return chi, overlap, fidelity, E, delta_E, Px, Stot_sq, sweeps



def dmrg_lr_spinone_heisenberg_finite(L=10, alpha=10.0, D=0.0, B=0.0, n_exp=2, conserve='best'):
    model_params = dict(
        L=L,
        D=D,
        B=B,
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
            'svd_min': 1.e-9,
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
        'max_S_err': 1.e-9,
        'norm_tol': 1.e-9,
        'max_sweeps': 50,
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
    E = info['E']

    # calc observables for tracking convergence
    tracking_obs = utilities.calc_tracking_quantities(psi, info, dmrg_params)

    # calculate observables
    #obs = utilities.calc_observables(psi)

    # save everything to a hdf5 file
    # TODO: add empty data dir to repo
    #filename = f"output/data/dmrg_data_observables_alpha{alpha}_D{D}_L{L}.h5"
    #data_io.save_results_obs(filename,  model_params=model_params,
    #                                init_state=product_state,
    #                                dmrg_params=dmrg_params,
    #                                dmrg_info=info,
    #                                mpo=M,
    #                                mps=psi,
    #                                observables={},
    #                                tracking_observables=tracking_obs
    #                     )

    # output to check sanity
    print("E = {E:.13f}".format(E=info['E']))
    print("final bond dimensions psi: ", psi.chi)

    return E, tracking_obs, psi


def main(argv):

    ######################
    # read terminal inputs
    L, D, alpha, n_exp = data_io.param_use(argv)
    eps = 1e-3

    ##########
    # AUXFIELD ?
    #B = 0.
    if alpha > 3.0:
        B = 1e-6
    else:
        B = 0.

    ##########
    # run dmrg
    start_time = time.time()
    E, tracking_obs, psi = dmrg_lr_spinone_heisenberg_finite(L=L, D=D, alpha=alpha, B=B, n_exp=n_exp)
    E_eps, tracking_obs_eps, psi_eps = dmrg_lr_spinone_heisenberg_finite(L=L, D=D+eps, alpha=alpha, B=B, n_exp=n_exp)
    print("--- %s seconds ---" % (time.time() - start_time))

    ###########################
    # writing the data to files
    # unpack and prepare
    nsweeps, chi, Px, Stot_sq = tracking_obs
    chi_limit, chi_max = chi
    nsweeps_eps, chi_eps, Px_eps, Stot_sq_eps = tracking_obs_eps
    chi_limit, chi_max_eps = chi_eps
    delta_E = E_eps - E
    # save tracking obs
    str_tracking_obs = ["gs_energy", "gs_energy_eps", "gs_energy_diff", "parity_x", "parity_x_eps", "s_total", "s_total_eps", "chi_max", "chi_max_eps", "nsweeps", "nsweeps_eps"]
    tracking_obs = [E, E_eps, delta_E, Px, Px_eps, Stot_sq, Stot_sq_eps, chi_max, chi_max_eps, nsweeps, nsweeps_eps]
    data_io.write_observables_to_file("spinone_heisenberg_fidelity_trackobs", str_tracking_obs, tracking_obs, L, alpha, D, chi_limit)
    # save observables
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions
    fidelity = utilities.calc_fidelity(psi, psi_eps, eps)
    log_fidelity = utilities.calc_log_fidelity(psi, psi_eps, eps)
    obs = [eps, overlap, fidelity, log_fidelity]
    str_observables = ["eps", "overlap", "fidelity", "log_fidelity"]
    data_io.write_observables_to_file("spinone_heisenberg_fidelity_obs", str_observables, obs, L, alpha, D, chi_limit)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
