import csv
import os
from filelock import FileLock



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

def write_observables_to_file(str_observables : list, observables : list, L : int, alpha : float, D : list, chi : int):
    if len(str_observables) != len(observables):
        print("Length of str_observables does not match length of observables.")
        return

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_obs_chi{chi}_alpha{alpha}_L{L}.csv'  # FIXME

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

