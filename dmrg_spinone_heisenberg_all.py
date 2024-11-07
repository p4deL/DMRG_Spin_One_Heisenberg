import getopt
import numpy as np
import sys
import time

import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

import include.data_io as data_io

# Set the number of BLAS threads
#os.environ["OMP_NUM_THREADS"] = "1"    # For OpenMP (used by some BLAS implementations)
#os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS
#os.environ["MKL_NUM_THREADS"] = "1"       # For MKL
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For Accelerate (macOS)
#os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr, if used

def usage():
    print("Usage: dmrg_spinone_heisenberg_lr_coupling.py -L <length of chain> -D <single ion anisotropy strength> -a <decay exponent alpha> -e <number of exp terms to fit power law>")


def param_use(argv):
    L = 0
    D = 1.
    alpha = 10.
    n_exp = 0
    found_l = found_D = found_a = found_exp = False

    try:
        opts, args = getopt.getopt(argv, "L:D:a:e:h", ["Length=", "D=", "alpha=", "nexp=", "help"])
    except getopt.GetoptError:
      usage()
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("-h", "--help"):
         usage()
      elif opt in ("-L", "--Length"):
        L = int(arg)
        found_l = True
      elif opt in ("-D", "--D"):
        D = float(arg)
        found_D = True
      elif opt in ("-a", "--alpha"):
        alpha = float(arg)
        found_a = True
      elif opt in ("-e", "--nexp"):
          n_exp = int(arg)
          found_exp = True

    if not found_l:
     print("Length of ladder (system size) not given.")
     usage()
     sys.exit(2)
    if not found_D:
      print("single-ion anisotropy strength not given.")
      usage()
      sys.exit(2)
    if not found_a:
      print("decay exponent not given.")
      usage()
      sys.exit(2)
    if not found_exp:
        print("number of exponential terms not given.")
        usage()
        sys.exit(2)

    return L, D, alpha, n_exp



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

    # TODO: Save METADATA!!!!!!!


    # create spine one model
    M = LongRangeSpin1ChainExp(model_params)

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

    # TODO: save ground_state and results
    # TODO: save in subdirectory wavefunctions observables/ logs/ ~ think about good directory structure

    # log data
    data_io.log_sweep_statistics(L, alpha, D, info['sweep_statistics'])

    # calc observables for tracking convergence
    tracking_obs = calc_tracking_quantities(psi, info, dmrg_params)

    # TODO: Calculate the following quantities
    # - ground-state energy
    # - Mag_|| mag_perp staggered!
    # - string order parameter
    # - entanglement entropy
    obs = calc_observables(psi)


    # output to check sanity
    print("E = {E:.13f}".format(E=info['E']))
    print("final bond dimensions psi: ", psi.chi)

    return E, tracking_obs, obs


def main(argv):

    # read terminal inputs
    L, D, alpha, n_exp = param_use(argv)

    start_time = time.time()
    E, tracking_obs, obs = dmrg_lr_spinone_heisenberg_finite(L=L, D=D, alpha=alpha, n_exp=n_exp)
    sweeps, chi, Px, Stot_sq = tracking_obs
    chi_limit, chi_max = chi
    print("--- %s seconds ---" % (time.time() - start_time))

    # print observables
    #print(f"fidelity susceptibility: {fidelity}")  # TODO

    # write results to files
    # TODO: Generalize function such that it takes multiple strings
    #data_io.write_quantity_to_file("gs_energy", E, chi_limit, alpha, D, L)
    #data_io.write_quantity_to_file("parity_x", Px, chi_limit, alpha, D, L)
    #data_io.write_quantity_to_file("s_total", Stot_sq, chi_limit, alpha, D, L)
    #data_io.write_quantity_to_file("nsweeps", sweeps, chi_limit, alpha, D, L)
    #data_io.write_quantity_to_file("chi_max", chi_max, chi_limit, alpha, D, L)

    # writing the data to files
    str_tracking_obs = ["gs_energy", "parity_x", "s_total", "nsweeps", "chi_max"]
    tracking_obs = [E, Px, Stot_sq, chi_max, chi_max]
    data_io.write_observables_to_file(str_tracking_obs, tracking_obs, L, alpha, D, chi_limit)

    str_observables = ["SvN", "m_trans", "m_long", "str_order"]
    data_io.write_observables_to_file(str_observables, list(obs), L, alpha, D, chi_limit)



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
