# padelhardt

import sys
import time
import csv
import numpy as np
import getopt
import os
import math
from filelock import FileLock
from scipy.optimize import curve_fit
from tenpy.tools.fit import sum_of_exp, plot_alg_decay_fit #, fit_with_sum_of_exp
import matplotlib.pyplot as plt

import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite
from tenpy.models.model import CouplingModel, MPOModel, CouplingMPOModel
from tenpy.models.spins import SpinChain
from tenpy.models.lattice import Lattice, Chain, Ladder
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.linalg.np_conserved import Array, LegCharge

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


def log_sweep_statistics(L, alpha, D, sweep_info):
    # global log number of sweeps
    # TODO: I could also print other info here like max bond dimension
    with open(f"logs/0_spinone_heisenberg_L{L}_alpha{alpha}_nsweeps.log", 'a') as file:
        # If the file is empty, write the header first
        if file.tell() == 0:
            file.write("D,nsweeps\n")


        nsweeps = len(sweep_info['sweep'])
        # Write the two values in CSV format
        file.write(f"{D},{nsweeps}\n")

    # TODO: Whould be nice to also depict convergence criteria values in plots. Where and how to include?
    # write detailed info in seperate file
    with open(f"logs/1_spinone_heisenberg_L{L}_alpha{alpha}_D{D}_info.log", 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        header = [ key for key in sweep_info ]
        info_lists = [ list for list in sweep_info.values()]

        writer.writerow(header)

        # Write the data row by row
        for row in zip(*info_lists):
            writer.writerow(row)


def write_quantity_to_file(quantity_string, quantity, chi, alpha, D, L):

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_{quantity_string}_chi{chi}_alpha{alpha}_L{L}.csv'  # FIXME

    # lock files when writing (necessary when using a joblist)
    with FileLock(filename + ".lock"):
        if os.path.isfile(filename):
            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([D, quantity])  # Append D and fidelity
        else:
            with open(filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(["D", quantity_string])
                writer.writerow([D, quantity])  # Append D and fidelity


def write_quantity_to_file2(quantity_string, quantity, alpha, D, eps, L):

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_{quantity_string}_alpha{alpha}_L{L}.csv'  # FIXME

    # lock files when writing (necessary when using a joblist)
    with FileLock(filename + ".lock"):
        if os.path.isfile(filename):
            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([D, quantity])  # Append D and fidelity
        else:
            with open(filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(["D", "fidelity"])
                writer.writerow([D, quantity])  # Append D and fidelity



def alternating_power_law_decay(dist, alpha):
    return (-1) ** (dist + 1) / dist ** alpha

def power_law_decay(dist, alpha):
    return 1. / dist ** alpha

def fit_with_sum_of_exp(f, alpha, n, N=50):
    r"""Approximate a decaying function `f` with a sum of exponentials.

    MPOs can naturally represent long-range interactions with an exponential decay.
    A common technique for other (e.g. powerlaw) long-range interactions is to approximate them
    by sums of exponentials and to include them into the MPOs.
    This function allows to do that.

    The algorithm/implementation follows the appendix of :cite:`murg2010`.

    Parameters
    ----------
    f : function
        Decaying function to be approximated. Needs to accept a 1D numpy array `x`
    n : int
        Number of exponentials to be used.
    N : int
        Number of points at which to evaluate/fit `f`;
        we evaluate and fit `f` at the points ``x = np.arange(1, N+1)``.

    Returns
    -------
    lambdas, prefactors: 1D arrays
        Such that :math:`f(k) \approx \sum_i x_i \lambda_i^k` for (integer) 1 <= `k` <= `N`.
        The function :func:`sum_of_exp` evaluates this for given `x`.
    """
    assert n < N
    x = np.arange(1, N + 1)
    f_x = f(x, alpha)
    F = np.zeros([N - n + 1, n])
    for i in range(n):
        F[:, i] = f_x[i:i + N - n + 1]

    U, V = np.linalg.qr(F)
    U1 = U[:-1, :]
    U2 = U[1:, :]
    M = np.dot(np.linalg.pinv(U1), U2)
    lam = np.linalg.eigvals(M)
    lam = np.sort(lam)[::-1]
    # least-square fit
    W = np.power.outer(lam, x).T
    pref, res, rank, s = np.linalg.lstsq(W, f_x, None)
    return lam, pref

def plot_fit(x_vals, y_powerlaw, y_sumexp):
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_powerlaw, marker='o', label='Power law')
    plt.plot(x_vals, y_sumexp, marker='x', label='sum exp')
    plt.legend(loc='best')
    plt.show()


class LongRangeSpin1ChainExp(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            spin_site = SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        elif conserve == 'parity':
            spin_site = SpinSite(S=1., conserve='parity', sort_charge=sort_charge)
        else:
            spin_site = SpinSite(S=1., conserve=None, sort_charge=sort_charge)

        return spin_site

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)
        alpha = model_params.get('alpha', float('inf'))
        n_exp = model_params.get('n_exp', 2)  # Number of exponentials in fit
        fit_range = model_params.get('fit_range', self.lat.N_sites)  # Range of fit for decay


        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, 'Sx')
            self.add_onsite(D, u, 'Sz Sz')

        if math.isinf(alpha):
            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
                self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)
        else:
            # fit power-law decay with sum of exponentials
            lam, pref = fit_with_sum_of_exp(power_law_decay, alpha, n_exp, fit_range)
            x = np.arange(1, fit_range + 1)
            print("*" * 100)
            print(f'n_exp = {n_exp}')
            print('error in fit: {0:.3e}'.format(np.sum(np.abs(power_law_decay(x, alpha) - sum_of_exp(lam, pref, x)))))
            #print(lam, pref)
            #plot_fit(x, power_law_decay(x, alpha), sum_of_exp(lam, pref, x) )
            print("*" * 100)

            sys.exit()

            # add exponentially_decaying terms
            for pr, la in zip(pref, lam):
                self.add_exponentially_decaying_coupling(0.5*pr, la, 'Sp', 'Sm', plus_hc=True)
                self.add_exponentially_decaying_coupling(pr, la, 'Sz', 'Sz')
                # change sign of Ferro couplings
                prprime = -2*pr
                laprime = la**2
                # couplings on even sites
                even_sites = list(range(0,self.lat.N_sites,2))
                self.add_exponentially_decaying_coupling(0.5*prprime, laprime, 'Sp', 'Sm', subsites=even_sites, plus_hc=True)
                self.add_exponentially_decaying_coupling(prprime, laprime, 'Sz', 'Sz', subsites=even_sites)
                # couplings on odd sites
                odd_sites = list(range(1,self.lat.N_sites,2))
                self.add_exponentially_decaying_coupling(0.5*prprime, laprime, 'Sp', 'Sm', subsites=odd_sites, plus_hc=True)
                self.add_exponentially_decaying_coupling(prprime, laprime, 'Sz', 'Sz', subsites=odd_sites)


class LongRangeSpin1ChainFrustExp(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            spin_site = SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        else:
            spin_site = SpinSite(S=1., conserve=None, sort_charge=sort_charge)

        return spin_site

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)
        alpha = model_params.get('alpha', 100.)  # FIXME no need for alpha anymore
        n_exp = model_params.get('n_exp', 10)  # Number of exponentials in fit
        fit_range = model_params.get('fit_range', 2*self.lat.N_sites)  # Range of fit for decay


        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, 'Sx')
            self.add_onsite(D, u, 'Sz Sz')

        #for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
        #    print(f"dx={dx}")
        #    print(f"u1={u1}, u2={u2}")
        #    self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
        #    self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)


        # fit power-law decay with sum of exponentials
        lam, pref = fit_with_sum_of_exp(power_law_decay, alpha, n_exp, fit_range)
        x = np.arange(1, fit_range + 1)
        print("*" * 100)
        #print(lam, pref)
        print(f'n_exp: {n_exp}')
        print('error in fit: {0:.3e}'.format(np.sum(np.abs(power_law_decay(x, alpha) - sum_of_exp(lam, pref, x)))))
        #x_vals = np.arange(1, 0.02, fit_range+0.02)
        #plot_fit(x, power_law_decay(x, alpha), sum_of_exp(lam, pref, x) )
        print("*" * 100)

        # add expontially_decaying terms
        for pr, la in zip(pref, lam):
            self.add_exponentially_decaying_coupling(0.5*pr, la, 'Sp', 'Sm', plus_hc=True)
            self.add_exponentially_decaying_coupling(pr, la, 'Sz', 'Sz')


class LongRangeSpin1ChainFrust(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    #def init_sites(self, model_params):
    #    B = model_params.get('B', 0.)
    #    conserve = model_params.get('conserve', 'best')
    #    if conserve == 'best':
    #        conserve = 'Sz' if not model_params.any_nonzero(['B']) else None
    #        self.logger.info("%s: set conserve to %s", self.name, conserve)
    #    sort_charge = model_params.get('sort_charge', True)
    #    return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            return SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        else:
            return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)
        alpha = model_params.get('alpha', 100.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, 'Sx')
            self.add_onsite(D, u, 'Sz Sz')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            print(f"dx={dx}")
            print(f"u1={u1}, u2={u2}")
            self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)

        for dist in range(2, self.lat.N_sites):  # Only add for j > i to avoid double counting
            # print(dist)
            strength = 1. / (dist ** alpha)  # Long-range decay
            self.add_coupling(strength, 0, "Sz", 0, "Sz", dx=dist)
            self.add_coupling(0.5 * strength, 0, "Sp", 0, "Sm", dx=dist, plus_hc=True)



class LongRangeSpin1Chain(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    #def init_sites(self, model_params):
    #    B = model_params.get('B', 0.)
    #    conserve = model_params.get('conserve', 'best')
    #    if conserve == 'best':
    #        conserve = 'Sz' if not model_params.any_nonzero(['B']) else None
    #        self.logger.info("%s: set conserve to %s", self.name, conserve)
    #    sort_charge = model_params.get('sort_charge', True)
    #    return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            return SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        else:
            return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)
        alpha = model_params.get('alpha', 100.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, 'Sx')
            self.add_onsite(D, u, 'Sz Sz')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)

        for dist in range(2, self.lat.N_sites):  # Only add for j > i to avoid double counting
            strength = (-1) ** (dist + 1) / (dist ** alpha)  # Long-range decay
            self.add_coupling(strength, 0, "Sz", 0, "Sz", dx=dist)
            self.add_coupling(0.5 * strength, 0, "Sp", 0, "Sm", dx=dist, plus_hc=True)


class LongRangeSpin1ChainNN(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            return SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        else:
            return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, 'Sx')
            self.add_onsite(D, u, 'Sz Sz')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)

def calc_fidelity(psi, psi_eps, eps):
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions
    return -2 * np.log(overlap) / (eps ** 2)  # fidelity susceptiblity


def calc_fidelity_ed(psi, psi_eps, eps):
    overlap = npc.inner(psi, psi_eps, axes='range', do_conj=True)
    print(f"ed_overlap: {overlap}")
    return -2 * np.log(overlap) / (eps ** 2)  # fidelity susceptiblity


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
        'max_E_err': 1.e-9,
        'max_S_err': 1.e-7,
        'norm_tol': 5.e-7,
        'max_sweeps': 100,
    }

    # TODO: Save METADATA!!!!!!!


    # create spine one model
    M = LongRangeSpin1ChainExp(model_params)
    model_params['D'] = D+eps
    M_eps = LongRangeSpin1ChainExp(model_params)

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
    log_sweep_statistics(L, alpha, D, info['sweep_statistics'])
    log_sweep_statistics(L, alpha, D+eps, info_eps['sweep_statistics'])

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
    fidelity = calc_fidelity(psi, psi_eps, eps)

    # calcualte overlap of wavefunctions
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions

    # output to check sanity
    print("E = {E:.13f}".format(E=info['E']))
    print("final bond dimensions psi: ", psi.chi)
    print("E_eps = {E:.13f}".format(E=info_eps['E']))
    print("final bond dimensions psi_eps: ", psi_eps.chi)

    return chi, overlap, fidelity, E, delta_E, Px, Stot_sq, sweeps


def main(argv):

    # read terminal inputs
    L, D, alpha, n_exp = param_use(argv)
    eps = 2e-3

    start_time = time.time()
    chi, overlap, fidelity, E, delta_E, Px, Stot_sq, sweeps = dmrg_lr_spinone_heisenberg_finite_fidelity(L=L, D=D, alpha=alpha, n_exp=n_exp, eps=eps)
    chi_limit, chi_max, chi_max_eps = chi
    print("--- %s seconds ---" % (time.time() - start_time))

    # print observables
    print(f"fidelity susceptibility: {fidelity}")

    # write results to files
    write_quantity_to_file("fidelity", fidelity, chi_limit, alpha, D, L)
    write_quantity_to_file("overlap", overlap, chi_limit, alpha, D, L)
    write_quantity_to_file("gs_energy", E[0], chi_limit, alpha, D, L)
    write_quantity_to_file("gs_energy_eps", E[1], chi_limit, alpha, D+eps, L)
    write_quantity_to_file("gs_energy_diff", delta_E, chi_limit, alpha, D, L)
    write_quantity_to_file("parity_x", Px[0], chi_limit, alpha, D, L)
    write_quantity_to_file("parity_x_eps", Px[1], chi_limit, alpha, D, L)
    write_quantity_to_file("s_total", Stot_sq[0], chi_limit, alpha, D, L)
    write_quantity_to_file("s_total_eps", Stot_sq[1], chi_limit, alpha, D+eps, L)
    write_quantity_to_file("nsweeps", sweeps[0], chi_limit, alpha, D, L)
    write_quantity_to_file("nsweeps_eps", sweeps[1], chi_limit, alpha, D+eps, L)
    write_quantity_to_file("chi_max", chi_max, chi_limit, alpha, D, L)
    write_quantity_to_file("chi_max_eps", chi_max_eps, chi_limit, alpha, D+eps, L)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
