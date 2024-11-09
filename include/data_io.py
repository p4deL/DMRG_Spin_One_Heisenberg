import csv
from filelock import FileLock
import getopt
import h5py
import os
import sys
from tenpy.tools import hdf5_io


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


def write_quantity_to_file(quantity_string : str, quantity : float, L : int, alpha, D, chi : int):

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


def write_observables_to_file(str_base : str, str_observables : list, observables : list, L : int, alpha : float, D : list, chi : int):
    if len(str_observables) != len(observables):
        print("Length of str_observables does not match length of observables.")
        print(str_observables)
        print(observables)
        return

    # Open a file in write mode
    filename = f'output/{str_base}_chi{chi}_alpha{alpha}_L{L}.csv'  # FIXME

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


def save_results_obs(filename,  model_params={},
                                init_state={},
                                dmrg_params={},
                                dmrg_info={},
                                mpo=None,
                                mps=None,
                                observables=(),
                                tracking_observables=()
                     ):

    #data structure
    # - Model
    #    L Hamiltonian sting?
    #    L model parameters
    # - DMRG
    #    L initial product state
    #    L dmrg parameters
    #    L info (split this?) Exclude final E
    # - Results
    #    L MPO
    #    L MPS
    #    L observables

    # TODO: what about exp fit parameters? I could save some storage, if I save the parameters instead of entire MPO
    data = {
        "Model" : {
            "Hamiltonian" : r"H = J \sum_i \vec {S}_i \cdot \vec {S}_{i + 1} + B \sum_i(-1) ^ i(S ^ x_i + S ^ y_i) + D \sum_i(S ^ z_i) ^ 2",
            "model parameters" : model_params,
       },
       "DMRG": {
           "initial product state" : init_state,
           "dmrg parameters" : dmrg_params,
           "dmrg info" : dmrg_info
       },
       "Results": {
           "MPO" : mpo,
           "MPS" : mps,
           "observables" : observables,
           "tracking_observables" : tracking_observables
       }
    }

    with h5py.File(filename, 'w') as f:
        hdf5_io.save_to_hdf5(f, data)


