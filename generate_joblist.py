import numpy as np

# Parameters
#filename = "joblist_ed.txt"
filename = "joblist_dmrg.txt"
#script = "ed_fidelity.py"
script = "dmrg_spinone_heisenberg_lr_couplings.py"
L = 10
alpha = 10.0
Ds = np.arange(-1.5,1.5,0.02)

# Writing to file
with open(filename, "w") as file:
    for D in Ds:
        line = f"python {script} -L {L} -D {D} -a {alpha}\n"
        file.write(line)

print(f"Output written to {filename}")

