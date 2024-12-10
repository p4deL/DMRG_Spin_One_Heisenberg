from __future__ import print_function, division
from quspin.operators import hamiltonian, quantum_operator, exp_op   # Hamiltonians and exp_op
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
import numpy as np  # generic math functions
import sys
import getopt
import itertools
import os
import csv
import threading

from filelock import FileLock


def usage():
    print("Usage: ed_spectrum.py -L <length of chain> -D <single ion anisotropy strength> -a <decay exponent alpha>")


def param_use(argv):
    L = 0
    D = 1.
    alpha = 10.
    found_l = found_D = found_a = False

    try:
        opts, args = getopt.getopt(argv, "L:D:a:h", ["Length=", "D=", "alpha=", "help"])
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

    return L, D, alpha


def write_quantity_to_file(quantity_string : str, quantity : float, L : int, alpha : float, D : float, Gamma : float):

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_ed_{quantity_string}_alpha{alpha}_L{L}.csv'  # FIXME

    # lock files when writing (necessary when using a joblist)
    with FileLock(filename + ".lock"):
        if os.path.isfile(filename):
            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([D, Gamma, quantity])  # Append D and fidelity
        else:
            with open(filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(["D", "Gamma", quantity_string])
                writer.writerow([D, Gamma, quantity])  # Append D and fidelity


def write_observables_to_file(str_base : str, str_observables : list, observables : list, L : int, alpha : float, D : float, Gamma : float):
    if len(str_observables) != len(observables):
        print("Length of str_observables does not match length of observables.")
        print(str_observables)
        print(observables)
        return

    # Open a file in write mode
    filename = f'output/{str_base}_alpha{alpha}_L{L}.csv'  # FIXME

    observables.insert(0, Gamma)
    str_observables.insert(0, "Gamma")
    observables.insert(0, D)
    str_observables.insert(0, "D")

    # lock files when writing (necessary when using a joblist)
    with FileLock(filename + ".lock"):
        if os.path.isfile(filename):
            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(observables)  # Append D and fidelity
        else:
            with open(filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(str_observables)
                writer.writerow(observables)  # Append D and fidelity



def calc_coherence(basis, ham_list):
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

    # diagonalize hamiltonian, get the ground-state and ground-state energy
    H = hamiltonian(ham_list, [], basis=basis, dtype=np.float64, **no_checks)
    Es, psis = H.eigh()

    es = Es/basis.L

    rho = np.outer(es,es)
    dim = len(Es)
    print(np.shape(psis))
    #print(psis)

    #coherences = np.sum([np.sum([np.dot(psis[:,i].T, psis[:,i+m]) for i in range(dim)]) for m in range(-dim,dim)])
    m=1
    coherences = [np.dot(psis[:,i].T, psis[:,i+m]) for i in range(dim-m)]
    print(coherences)

    return Es/basis.L

def calc_stag_mag(basis, psi, component='z'):
    # set up the observable
    mag_int = [f"{component}{component}", [[(-1) ** (i + j), i, j] for i, j in itertools.product(range(basis.L), repeat=2)]]
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    stag_mag_obs = hamiltonian([mag_int], [], basis=basis, dtype=np.float64, **no_checks)

    # calc expectation value and normalize according to stat average
    stag_mag = stag_mag_obs.expt_value(psi) / basis.L ** 2

    return stag_mag[0]


def calc_fidelity(basis, ham_lists, eps):
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

    # diagonalize hamiltonian, get the ground-state and ground-state energy
    H = hamiltonian(ham_lists[0], [], basis=basis, dtype=np.float64, **no_checks)
    E, psi = H.eigsh(k=1, which='SA')

    H_eps = hamiltonian(ham_lists[1], [], basis=basis, dtype=np.float64, **no_checks)
    E_eps, psi_eps = H_eps.eigsh(k=1, which='SA')

    overlap = np.abs(np.dot(psi.T,psi_eps)[0,0])
    print(overlap)

    return -2. * np.log(overlap)/eps**2


def main(argv):
    
    # read terminal inputs
    L, D, alpha = param_use(argv)

    # set up basis_1d
    # OBC
    #basis = spin_basis_1d(L, pauli=False, S='1', Nup=L, pblock=1)
    # PBC
    #basis = spin_basis_1d(L, pauli=False, S='1', Nup=L, kblock=0, pblock=1)
    basis = spin_basis_1d(L, pauli=False, S='1')

    # init static interactions
    # OBC
    #J = [[(-1) ** (delta - 1) / delta ** alpha, i, (i + delta)] for i in range(L) for delta in range(1, L - i)]
    #Jxy = [[1./2*(-1) ** (delta - 1) / delta ** alpha, i, (i + delta)] for i in range(L) for delta in range(1, L - i)]
    # NN
    #J = [[1.,i,i+1] for i in range(L-1)]
    #Jxy = [[1./2, i,i+1] for i in range(L-1)]

    # PBC
    J = [[(-1)**(delta-1) / delta**alpha, i, (i+delta)%L] for i,delta in itertools.product(range(L), range(1,L))]
    Jxy = [[1./2*(-1)**(delta-1) / delta**alpha, i, (i+delta)%L] for i,delta in itertools.product(range(L), range(1,L))]

    Ds = [[D, i, i] for i in range(L)]
    ham_list = [["+-", Jxy], ["-+", Jxy], ["zz", J], ["zz", Ds]]

    # calculate observables
    Es = calc_coherence(basis, ham_list)

    #print(np.size(Es))


    # calculate magnetization
    # diagonalize hamiltonian, get the ground-state and ground-state energy
    #no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    #H = hamiltonian(ham_list, [], basis=basis, dtype=np.float64, **no_checks)
    #_, psi = H.eigsh(k=1, which='SA')

    #m_z = calc_stag_mag(basis, psi, component='z')


    # write results to files
    #write_quantity_to_file("m_long", m_z, L, alpha, D, 1.0)
    #energy_strings = [f"e{i}" for i in range(len(Es))]
    #write_observables_to_file("spinone_heisenberg_ed_spectrum", energy_strings, list(Es), L, alpha, D, 1.0)


if __name__ == "__main__":
    main(sys.argv[1:])
