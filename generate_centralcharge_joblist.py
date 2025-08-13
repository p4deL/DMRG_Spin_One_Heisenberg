import math
import numpy as np
import include.utilities as utilities

def write_joblist_files(basename, script, Ls, alpha, D, n_exps, sz1_flag):
    # Writing to file
    filename = f"{basename}_D{D}_alpha{alpha}.txt"
    with open(filename, "w") as file:
        for L, n_exp in zip(Ls,n_exps):
            line = f"python {script} -L {L} -D {D} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
            file.write(line)

    print(f"Output written to {filename}")

def write_one_joblist_file(filename, script, Ls, alpha, D, n_exps, sz1_flag, append=True):
    write_flag = 'w'
    if append:
        write_flag = 'a'

    # Writing to file
    with open(filename, write_flag) as file:
        for L, n_exp in zip(Ls,n_exps):
            line = f"python {script} -L {L} -D {D} -a {alpha} -e {n_exp} --sz{1 if sz1_flag else 0}\n"
            file.write(line)

    print(f"Output written to {filename}")



if __name__ == "__main__":

    #######################################################
    # Parameters
    script = "dmrg_spinone_heisenberg.py"
    basename = "joblist_dmrg_fss"
    sz1_flag = True
    plotflag = False
    alphas = [float('inf'), 25.0, 12.5, 8.333333333333334, 6.25, 5.0, 4.166666666666667, 3.571428571428571, 3.125, 2.7777777777777777]
    Ds = [0.968523515598219, 0.9685234920772998, 0.9685800021726326, 0.9695893107107798, 0.9732696633867682, 0.9802898648579852, 0.9905026303618746, 1.0035743011089546, 1.0195092619564319, 1.0384263592130984]
    #Ls = [60, 100, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    Ls = [60, 100, 140, 160, 180, 200]
    err_tol = 1e-9
    #######################################################

    n_exp_min = n_exp = 1
    for i, (D, alpha) in enumerate(zip(Ds, alphas)):

        if math.isinf(alpha):
            n_exps = np.ones(len(Ls), dtype=int)
            write_joblist_files(basename, script, Ls, alpha, D, n_exps, sz1_flag)
        else:
            n_exps = []
            for L in Ls:
                n_exp = utilities.determine_n_exp(n_exp_min=n_exp_min, err_tol=err_tol, L=L, alpha=alpha, plot=plotflag)
                n_exp_min = n_exp
                n_exps.append(n_exp)
            write_joblist_files(basename, script, Ls, alpha, D, n_exps, sz1_flag)

        write_one_joblist_file(f"{basename}_all.txt", script, Ls, alpha, D, n_exps, sz1_flag, append=i)


