import csv
from filelock import FileLock
import getopt
import h5py
import numpy as np
import os
import pandas as pd
import re
import sys
from tenpy.tools import hdf5_io


def usage():
    print("Usage: dmrg_spinone_heisenberg.py -L <length of chain> -D <single ion anisotropy strength> -G <Beyond-NN coupling parameter> -a <decay exponent alpha> -e <number of exp terms to fit power law>")


def param_use(argv):
    L = 0
    D = 1.
    Gamma = 1.
    Jz = 1.
    alpha = 10.
    n_exp = 0
    found_l = found_a = found_exp = False

    try:
        opts, args = getopt.getopt(argv, "L:D:J:G:a:e:h", ["Length=", "D=", "Jz", "Gamma=", "alpha=", "nexp=", "help"])
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
        elif opt in ("-J", "--Jz"):
            D = float(arg)
        elif opt in ("-G", "--Gamma"):
            Gamma = float(arg)
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
    if not found_a:
        print("decay exponent not given.")
        usage()
        sys.exit(2)
    if not found_exp:
        print("number of exponential terms not given.")
        usage()
        sys.exit(2)

    return L, D, Jz, Gamma, alpha, n_exp


def write_joblist_files(basename, script, L, alpha, Ds, Gammas, n_exp):
    # Writing to file
    filename = f"{basename}_L{L}_alpha{alpha}.txt"
    with open(filename, "w") as file:
        for D in Ds:
            for Gamma in Gammas:
                line = f"python {script} -L {L} -D {D} -G {Gamma} -a {alpha} -e {n_exp}\n"
                file.write(line)

    print(f"Output written to {filename}")


def write_xxz_joblist_files(basename, script, L, alpha, Jzs, n_exp):
    # Writing to file
    filename = f"{basename}_L{L}_alpha{alpha}.txt"
    with open(filename, "w") as file:
        for Jz in Jzs:
            line = f"python {script} -L {L} -J {Jz} -a {alpha} -e {n_exp}\n"
            file.write(line)

    print(f"Output written to {filename}")


def write_one_joblist_file(filename, script, L, alpha, Ds, Gammas, n_exp, append=True):
    write_flag = 'w'
    if append:
        write_flag = 'a'

    # Writing to file
    with open(filename, write_flag) as file:
        for D in Ds:
            for Gamma in Gammas:
                line = f"python {script} -L {L} -D {D} -G {Gamma} -a {alpha} -e {n_exp}\n"
                file.write(line)

    print(f"Output written to {filename}")


def write_one_xxz_joblist_file(filename, script, L, alpha, Jzs, n_exp, append=True):
    write_flag = 'w'
    if append:
        write_flag = 'a'

    # Writing to file
    with open(filename, write_flag) as file:
        for Jz in Jzs:
            line = f"python {script} -L {L} -J {Jz} -a {alpha} -e {n_exp}\n"
            file.write(line)

    print(f"Output written to {filename}")


def read_fss_data(path, obs_string, alpha, chi, cutoff_l=0, cutoff_r=0, reciprocal=False):
    """read data for given observable (as string) and return prepared data"""

    data_L = []
    data_tuning_param = []
    data_obs = []
    # iterate over files in path
    for f in os.listdir(path):
        # check if indeed is file
        if os.path.isfile(os.path.join(path, f)):
            # check if filename contains string of observable
            if "fss_obs" in f:
                print(f)
                # extract system size from filename
                # FIXME: TRACKOBS doesn't work if there are other files...
                system_size = int(re.search(f"spinone_heisenberg_fss_obs_chi{chi}_alpha{alpha}_L(.*).csv", f).group(1))

                # import csv with panda
                df = pd.read_csv(path + f)
                # data_array = df.to_numpy()
                # sigmas = data_array[:,0]
                # FIXME
                # hs = data_array[:,0]
                # obs = data_array[:,3] # FIXME do i need to multiply with L again?

                # FIXME: What if I want to read in Gamma as parameter? add input parameter
                Ds = df["D"].values
                obs = df[obs_string].values

                # TODO: only if necessary
                # prepare list with j as control fixed_param
                tmp = list(zip(Ds, obs))
                tmp.sort(key=lambda x: x[0])
                sorted_Ds = [tuples[0] for tuples in tmp]
                sorted_obs = [tuples[1] for tuples in tmp]
                l = len(sorted_Ds)

                L = list(np.ones(len(Ds[cutoff_l:l - cutoff_r])) * system_size)
                data_L += L
                data_tuning_param += sorted_Ds[cutoff_l:l - cutoff_r]
                data_obs += sorted_obs[cutoff_l:l - cutoff_r]

                # TODO: WRITE DOWN WHAT'S GOING ON
                # L = list(np.ones(len(hs))*system_size)
                # data_L += L
                # data_tuning_param += sorted_hs
                # data_obs += sorted_obs

                print(len(data_L))
                print(len(data_tuning_param))

    dim = len(sorted_Ds[cutoff_l:l - cutoff_r])
    if reciprocal:
        data = np.stack((np.array(data_L), np.reciprocal(data_tuning_param), np.array(data_obs)))
    else:
        data = np.stack((np.array(data_L), np.array(data_tuning_param), np.array(data_obs)))

    return data, dim


def log_sweep_statistics(L, alpha, D, Gamma, sweep_info):
    # global log number of sweeps
    # TODO: I could also print other info here like max bond dimension
    with open(f"logs/0_spinone_heisenberg_L{L}_alpha{alpha}_nsweeps.log", 'a') as file:
        # If the file is empty, write the header first
        if file.tell() == 0:
            file.write("D,Gamma,nsweeps\n")


        nsweeps = len(sweep_info['sweep'])
        # Write the two values in CSV format
        file.write(f"{D},{Gamma},{nsweeps}\n")

    # TODO: Would be nice to also depict convergence criteria values in plots. Where and how to include?
    # write detailed info in seperate file
    with open(f"logs/1_spinone_heisenberg_L{L}_alpha{alpha}_D{D}_Gamma{Gamma}_info.log", 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        header = [ key for key in sweep_info ]
        info_lists = [ list for list in sweep_info.values()]

        writer.writerow(header)

        # Write the data row by row
        for row in zip(*info_lists):
            writer.writerow(row)


def write_quantity_to_file(quantity_string : str, quantity : float, L : int, alpha : float, D : float, Gamma : float, chi : int):

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_{quantity_string}_chi{chi}_alpha{alpha}_L{L}.csv'  # FIXME

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


def write_observables_to_file(str_base : str, str_observables : list, observables : list, L : int, alpha : float, D : float, Gamma : float, chi : int):
    if len(str_observables) != len(observables):
        print("Length of str_observables does not match length of observables.")
        print(str_observables)
        print(observables)
        return

    # Open a file in write mode
    filename = f'output/{str_base}_chi{chi}_alpha{alpha}_L{L}.csv'  # FIXME

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


def write_correlations_to_file(correlator_strings : list, correlators : list, L : int, alpha : float, D : float, Gamma : float, chi : int):

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_correlations_chi{chi}_D{D}_Gamma{Gamma}_alpha{alpha}_L{L}.csv'

    correlator_array = np.vstack(correlators).T
    np.savetxt(filename, correlator_array, fmt='%f', delimiter=',', header=",".join(correlator_strings), comments='')


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
            "Hamiltonian" : r"H = \sum_i \vec{S}_i \cdot \vec{S}_{j} + \Gamma \sum_{j>i+1} (-1)^{i-j+1}/|i-j|^{\alpha}} \vec{S}_i \cdot \vec{S}_{j} + B S^z_0 + + D \sum_i (S^z_i)^2",
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


