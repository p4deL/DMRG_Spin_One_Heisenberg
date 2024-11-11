import numpy as np

# Parameters
#filename = "joblist_ed.txt"
filename = "joblist_dmrg.txt"
#script = "ed_fidelity.py"
script = "dmrg_spinone_heisenberg_fidelity.py"
L = 60
alpha = 'inf'
Ds = np.arange(-0.4,0.8,0.02)

#TODO fixme n e! -> common file with other joblist!

# Writing to file
with open(filename, "w") as file:
    for D in Ds:
        line = f"python {script} -L {L} -D {D} -a {alpha} -e 10 \n"
        file.write(line)

print(f"Output written to {filename}")

