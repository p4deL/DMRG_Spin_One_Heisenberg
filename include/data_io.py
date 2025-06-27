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
    L = 2
    D = 0.
    Gamma = 1.
    Jz = 1.
    alpha = 10.
    n_exp = 1
    found_l = found_a = found_exp = False
    sz1_flag = False

    try:
        opts, args = getopt.getopt(argv, "L:D:J:G:a:e:01h", ["Length=", "D=", "Jz=", "Gamma=", "alpha=", "nexp=", "sz0", "sz1", "help"])
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
            Jz = float(arg)
        elif opt in ("-G", "--Gamma"):
            Gamma = float(arg)
        elif opt in ("-a", "--alpha"):
            alpha = float(arg)
            found_a = True
        elif opt in ("-e", "--nexp"):
            n_exp = int(arg)
            found_exp = True
        elif opt in ("-0", "--sz0"):
            sz1_flag = False
        elif opt in ("-1", "--sz1"):
            sz1_flag = True


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

    return L, D, Jz, Gamma, alpha, n_exp, sz1_flag


def write_joblist_files(basename, script, L, alpha, Ds, Gammas, n_exp, sz1_flag):
    # Writing to file
    filename = f"{basename}_L{L}_alpha{alpha}.txt"
    with open(filename, "w") as file:
        for D in Ds:
            for Gamma in Gammas:
                line = f"python {script} -L {L} -D {D} -G {Gamma} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
                file.write(line)

    print(f"Output written to {filename}")

def write_alpha_joblist_files(basename, script, L, alphas, D, n_exps, sz1_flag):
    # Writing to file
    filename = f"{basename}_L{L}_D{D}.txt"
    with open(filename, "w") as file:
        for alpha, n_exp in zip(alphas, n_exps):
            line = f"python {script} -L {L} -D {D} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
            file.write(line)

    print(f"Output written to {filename}")

def write_xxz_joblist_files(basename, script, L, alpha, Jzs, n_exp, sz1_flag):
    # Writing to file
    filename = f"{basename}_L{L}_alpha{alpha}.txt"
    with open(filename, "w") as file:
        for Jz in Jzs:
            line = f"python {script} -L {L} -J {Jz} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
            file.write(line)

    print(f"Output written to {filename}")


def write_one_joblist_file(filename, script, L, alpha, Ds, Gammas, n_exp, sz1_flag, append=True):
    write_flag = 'w'
    if append:
        write_flag = 'a'

    # Writing to file
    with open(filename, write_flag) as file:
        for D in Ds:
            for Gamma in Gammas:
                line = f"python {script} -L {L} -D {D} -G {Gamma} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
                file.write(line)

    print(f"Output written to {filename}")


def write_alpha_one_joblist_file(filename, script, L, alphas, D, n_exps, sz1_flag, append=True):
    write_flag = 'w'
    if append:
        write_flag = 'a'

    # Writing to file
    with open(filename, write_flag) as file:
        for alpha, n_exp in zip(alphas, n_exps):
            line = f"python {script} -L {L} -D {D} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
            file.write(line)

    print(f"Output written to {filename}")


def write_fss_meta_data(alpha, xcinf, nonuni_pref_range, koppa, nu, lambdaflag, sz1_flag):
    filename = f"output/spinone_heisenberg_fss_metadata_alpha{alpha}.csv"

    meta_data_str = ["alpha", "xcinf", "nonuni_pref_range", "koppa", "nu", "lambda", "sz1_flag"]
    meta_data = [[alpha, xcinf, nonuni_pref_range, koppa, nu, lambdaflag, sz1_flag]]
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(meta_data_str)
        writer.writerows(meta_data)


def write_alpha_fss_meta_data(D, alphacinf, nonuni_pref_range, koppa, nu, lambdaflag, sz1_flag):
    filename = f"output/spinone_heisenberg_fss_metadata_D{D}.csv"

    meta_data_str = ["D", "alphacinf", "nonuni_pref_range", "koppa", "nu", "lambda", "sz1_flag"]
    meta_data = [[D, alphacinf, nonuni_pref_range, koppa, nu, lambdaflag, sz1_flag]]
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(meta_data_str)
        writer.writerows(meta_data)

def write_one_xxz_joblist_file(filename, script, L, alpha, Jzs, n_exp, sz1_flag, append=True):
    write_flag = 'w'
    if append:
        write_flag = 'a'

    # Writing to file
    with open(filename, write_flag) as file:
        for Jz in Jzs:
            line = f"python {script} -L {L} -J {Jz} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
            file.write(line)

    print(f"Output written to {filename}")


#def read_fss_data(path, obs_string, alpha, chi, L_min=0, cutoff_l=0, cutoff_r=0, reciprocal=False):
def read_fss_data(path, obs_string, variable_string, fixed_val, chi, L_min=0, cutoff_l=0, cutoff_r=0):
    """read data for given observable (as string) and return prepared data"""
    data_L = []
    data_tuning_param = []
    data_obs = []
    sorted_vars = []
    # iterate over files in path
    for f in os.listdir(path):
        # check if indeed is file
        if os.path.isfile(os.path.join(path, f)):
            # check if filename contains string of observable
            if "_obs_" in f:
                print(f)
                # extract system size from filename
                # FIXME: TRACKOBS doesn't work if there are other files...
                if variable_string == "alpha":
                    system_size = int(re.search(f"spinone_heisenberg_obs_chi{chi}_D{fixed_val}_L(.*).csv", f).group(1))
                else:
                    system_size = int(re.search(f"spinone_heisenberg_obs_chi{chi}_alpha{fixed_val}_L(.*).csv", f).group(1))

                if system_size >= L_min:

                    # import csv with panda
                    df = pd.read_csv(path + f)
                    if variable_string == "lambda":
                        vars = df["D"].values
                    else:
                        vars = df[variable_string].values
                    obs = df[obs_string].values
                    print(vars)

                    # TODO: only if necessary
                    # prepare list with j as control fixed_param
                    tmp = list(zip(vars, obs))
                    tmp.sort(key=lambda x: x[0])
                    sorted_vars = [tuples[0] for tuples in tmp]
                    sorted_obs = [tuples[1] for tuples in tmp]
                    l = len(sorted_vars)

                    L = list(np.ones(len(vars[cutoff_l:l - cutoff_r])) * system_size)
                    data_L += L
                    data_tuning_param += sorted_vars[cutoff_l:l - cutoff_r]
                    data_obs += sorted_obs[cutoff_l:l - cutoff_r]

                    # TODO: WRITE DOWN WHAT'S GOING ON
                    # L = list(np.ones(len(hs))*system_size)
                    # data_L += L
                    # data_tuning_param += sorted_hs
                    # data_obs += sorted_obs

                    print(len(data_L))
                    print(len(data_tuning_param))

    dim = len(sorted_vars[cutoff_l:l - cutoff_r])
    if variable_string == "lambda":
        data = np.stack((np.array(data_L), np.reciprocal(data_tuning_param), np.array(data_obs)))
    else:
        data = np.stack((np.array(data_L), np.array(data_tuning_param), np.array(data_obs)))

    return data, dim


def write_data_collapse_to_file(filename, red_n_points, L_min, x_c, dx_c, nu, dnu, exp, dexp):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if file is new
        if not file_exists:
            writer.writerow(['red_n_points', 'L_min', 'x_c', 'dx_c', 'nu', 'dnu', 'exp', 'dexp'])

        writer.writerow([red_n_points, L_min, x_c, dx_c, nu, dnu, exp, dexp])


def log_sweep_statistics(L, alpha, D, Gamma, Jz, sweep_info):
    # global log number of sweeps
    # TODO: I could also print other info here like max bond dimension
    with open(f"logs/0_spinone_heisenberg_L{L}_alpha{alpha}_nsweeps.log", 'a') as file:
        # If the file is empty, write the header first
        if file.tell() == 0:
            file.write("D,Gamma,nsweeps\n")


        nsweeps = len(sweep_info['sweep'])
        # Write the two values in CSV format
        file.write(f"{D},{Gamma},{Jz},{nsweeps}\n")

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


def write_quantity_to_file(quantity_string : str, quantity : float, L : int, alpha : float, D : float, Gamma : float, Jz : float, chi : int):

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
                writer.writerow(["D", "Gamma", "Jz", quantity_string])
                writer.writerow([D, Gamma, Jz, quantity])  # Append D and fidelity



def write_observables_to_file(str_base: str, str_observables: list, observables: list, L: int, alpha: float, D: float, Gamma: float, Jz: float, chi: int):
    if len(str_observables) != len(observables):
        print("Length of str_observables does not match length of observables.")
        print(str_observables)
        print(observables)
        return

    # Prepare filename
    filename = f'output/{str_base}_chi{chi}_alpha{alpha}_L{L}.csv'
    lockfile = filename + ".lock"

    # Prepend extra observables before locking
    header = ["Jz", "Gamma", "D"] + str_observables
    row = [Jz, Gamma, D] + observables

    # Lock files when writing (necessary when using a joblist)
    with FileLock(lockfile):
        file_exists = os.path.exists(filename)

        # Open file and write data
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists or os.stat(filename).st_size == 0:  # Check if the file is empty
                writer.writerow(header)
            writer.writerow(row)



def write_observables_to_file_fix_D(str_base: str, str_observables: list, observables: list, L: int, alpha: float, D: float, Gamma: float, Jz: float, chi: int):
    if len(str_observables) != len(observables):
        print("Length of str_observables does not match length of observables.")
        print(str_observables)
        print(observables)
        return

    # Prepare filename
    filename = f'output/{str_base}_chi{chi}_D{D}_L{L}.csv'
    lockfile = filename + ".lock"

    # Prepend extra observables before locking
    header = ["Jz", "Gamma", "alpha"] + str_observables
    row = [Jz, Gamma, alpha] + observables

    # Lock files when writing (necessary when using a joblist)
    with FileLock(lockfile):
        file_exists = os.path.exists(filename)

        # Open file and write data
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists or os.stat(filename).st_size == 0:  # Check if the file is empty
                writer.writerow(header)
            writer.writerow(row)

def write_correlations_to_file(base_name : str, correlator_strings : list, correlators : list, L : int, alpha : float, D : float, Gamma : float, Jz : float, chi : int):

    # Open a file in write mode
    filename = f'output/{base_name}_correlations_chi{chi}_D{D}_Gamma{Gamma}_Jz{Jz}_alpha{alpha}_L{L}.csv'

    correlator_array = np.vstack(correlators).T
    np.savetxt(filename, correlator_array, fmt='%f', delimiter=',', header=",".join(correlator_strings), comments='')


def write_entropies_to_file(base_name : str, entropy_strings : list, entropies : list, L : int, alpha : float, D : float, Gamma : float, Jz : float, chi : int):

    # Open a file in write mode
    filename = f'output/{base_name}_entropies_chi{chi}_D{D}_Gamma{Gamma}_Jz{Jz}_alpha{alpha}_L{L}.csv'

    entropy_array = np.vstack(entropies).T
    np.savetxt(filename, entropy_array, fmt='%f', delimiter=',', header=",".join(entropy_strings), comments='')

def write_mz_to_file(base_name : str, mz_strings : list, mzs : list, L : int, alpha : float, D : float, Gamma : float, Jz : float, chi : int):

    # Open a file in write mode
    filename = f'output/{base_name}_mzs_chi{chi}_D{D}_Gamma{Gamma}_Jz{Jz}_alpha{alpha}_L{L}.csv'

    mz_array = np.array(mzs).T
    np.savetxt(filename, mz_array, fmt='%f', delimiter=',', header=",".join(mz_strings), comments='')

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


