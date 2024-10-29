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
    print("Usage: ed_fidelity.py -L <length of chain> -D <single ion anisotropy strength> -a <decay exponent alpha>")


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


#def calc_susceptibility(basis, psi, psi_epsilon, epsilon, L):
#    # init static interaction list
#    mag_int = ["zz" , [[(-1)**(i+j), i, j] for i, j in itertools.product(range(L), repeat=2)]]
#
#    # set up the observable
#    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
#    stag_mag_obs = hamiltonian([mag_int], [], basis=basis, dtype=np.float64, **no_checks)
#
#    # calc expectation value and normalize according to stat average
#    stag_mag_epsilon = 3*stag_mag_obs.expt_value(psi_epsilon)[0]/L**2
#    stag_mag = 3*stag_mag_obs.expt_value(psi)[0]/L**2
#
#    return (stag_mag-stag_mag_epsilon)/epsilon  # suscept defined with minus



def calc_fidelity(basis, ham_lists, L, eps):
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

    # diagonalize hamiltonian, get the ground-state and ground-state energy
    H = hamiltonian(ham_lists[0], [], basis=basis, dtype=np.float64, **no_checks)
    E, psi = H.eigsh(k=1, which='SA')

    H_eps = hamiltonian(ham_lists[1], [], basis=basis, dtype=np.float64, **no_checks)
    E_eps, psi_eps = H_eps.eigsh(k=1, which='SA')

    overlap = np.abs(np.dot(psi.T,psi_eps)[0,0])
    print(overlap)

    return -2. * np.log(overlap)/eps**2


def write_quantity_to_file(quantity_string, quantity, alpha, D, L):

    # Open a file in write mode
    filename = f'output/ed_spinone_heisenberg_{quantity_string}_alpha{alpha}_L{L}.csv'

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


def main(argv):
    
    # read terminal inputs
    L, D, alpha = param_use(argv)
    eps = 1e-4

    # set up basis_1d
    # OBC
    basis = spin_basis_1d(L, pauli=False, S='1', Nup=L, pblock=1)
    # PBC
    #basis = spin_basis_1d(L, pauli=False, S='1', Nup=L, kblock=0, pblock=1)

    # init static interactionsl
    # OBC
    #J = [[(-1) ** (delta - 1) / delta ** alpha, i, (i + delta)] for i in range(L) for delta in range(1, L - i)]
    #Jxy = [[1./2*(-1) ** (delta - 1) / delta ** alpha, i, (i + delta)] for i in range(L) for delta in range(1, L - i)]
    # Frust
    J = [[1. / delta ** alpha, i, (i + delta)] for i in range(L) for delta in range(1, L - i)]
    Jxy = [[0.5 / delta ** alpha, i, (i + delta)] for i in range(L) for delta in range(1, L - i)]
    # NN
    #J = [[1.,i,i+1] for i in range(L-1)]
    #Jxy = [[1./2, i,i+1] for i in range(L-1)]

    # PBC
    #J = [[(-1)**(delta-1) / delta**alpha, i, (i+delta)%L] for i,delta in itertools.product(range(L), range(1,L))]
    #Jxy = [[1./2*(-1)**(delta-1) / delta**alpha, i, (i+delta)%L] for i,delta in itertools.product(range(L), range(1,L))]

    Ds = [[D, i, i] for i in range(L)]
    Ds_eps = [[D+eps, i, i] for i in range(L)]

    ham_list = [["+-", Jxy], ["-+", Jxy], ["zz", J], ["zz", Ds]]
    ham_eps_list = [["+-", Jxy], ["-+", Jxy], ["zz", J], ["zz", Ds_eps]]

    # calculate observables
    fidelity = calc_fidelity(basis, [ham_list,ham_eps_list], L, eps)

    # print observables
    print(f"fidelity susceptibility: {fidelity}")

    # write results to files
    write_quantity_to_file("fidelity", fidelity, alpha, D, L)


if __name__ == "__main__":
    main(sys.argv[1:])
