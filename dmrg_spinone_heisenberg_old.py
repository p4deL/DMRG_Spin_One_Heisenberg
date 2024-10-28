import time
import csv
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinModel
from tenpy.models.spins import SpinChain
from tenpy.linalg import np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import InitialStateBuilder
from tenpy.networks.site import SpinSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain

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

def decay(x):
    #return np.exp(-0.1*x) / x**2
    return (-1)**(x+1) / (x ** alpha)


# TODO: I need to add one term between even sites and one term between odd sites
#from tenpy.tools.fit import fit_with_sum_of_exp, sum_of_exp
n_exp = 5
fit_range = 50
lam, pref = fit_with_sum_of_exp(decay, n_exp, fit_range)
x = np.arange(1, fit_range + 1)
print('error in fit: {0:.3e}'.format(np.sum(np.abs(decay(x) - sum_of_exp(lam, pref, x)))))

for pr, la in zip(pref, lam):
   self.add_exponentially_decaying_coupling(pr, la, 'Sz', 'N')


def calc_correlations(psi):
    length = psi.L
    corr_long = psi.correlation_function("Sz", "Sz", site1=range(length))
    corr_trans = psi.correlation_function("S+", "S-", site1=range(length))

    return corr_long, corr_trans


def calc_order_parameters(psi):

    Sz = psi.expectation_value("Sz")
    Spm = psi.expectation_value(["S+", "S-"])

    mag_z = np.mean(Sz)
    print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],Sz1=Sz[1],mag_z=mag_z))
    mag_pm = np.mean(Spm)
    print("<S_+S_-> = [{Spm0:.5f}, {Spm1:.5f}]; mean ={mag_pm:.5f}".format(Spm0=Spm[0],Spm1=Spm[1],mag_pm=mag_pm))
    Sz = psi.sites[0].Sz
    print(Sz)
    Bk = npc.expm(1.j * np.pi * Sz)
    str_order = psi.correlation_function("Sz", "Sz", opstr=Bk, str_on_first=False)
    print(f"<O_string>=${str_order}")
    #print("<O_string> = [{Spm0:.5f}, {Spm1:.5f}]; mean ={mag_pm:.5f}".format(Spm0=Spm[0],Spm1=Spm[1],mag_pm=mag_pm))


def calc_fidelity(psi, psi_eps, eps):
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions
    return -2 * np.log(overlap) / (eps ** 2)  # fidelity susceptiblity



def dmrg_lr_spinone_heisenberg_finite_fidelity(L=100, alpha=10.0, D=0.0, eps=1e-4, conserve='Sz'):
    model_params = dict(
        L=L,
        S=1.,  # spin 1
        D=D,  # anisotropy coupling
        bc_MPS='finite',
        conserve=conserve
    )

    M = SpinChain(model_params)

    #for dist in range(2, L):  # Only add for j > i to avoid double counting
        #print(dist)
        #strength = (-1)**(dist+1) / (dist ** alpha)  # Long-range decay
        #M.add_coupling(strength, 0, "Sz", 0, "Sz", dx=dist)
        #M.add_coupling(0.5*strength, 0, "Sp", 0, "Sm", dx=dist)
        #M.add_coupling(0.5*strength, 0, "Sm", 0, "Sp", dx=dist)
        #M.add_coupling(0.5*strength, 0, "Sp", 0, "Sm", dx=dist, plus_hc=True)
        #M.add_coupling(strength, 0, "Sx", 0, "Sx", dx=dist)
        #M.add_coupling(strength, 0, "Sy", 0, "Sy", dx=dist)

    # Assuming 'model' is your defined SpinModel
    #print("Coupling terms in the model:")
    #for term in M.coupling_terms:
    #    print(term)
    #    J, op1, i1, op2, i2, u1, u2 = term
    #    print(f"Interaction: {op1} at site {i1} (unit cell {u1}) with {op2} at site {i2} (unit cell {u2}), coupling strength: {J}")

    #M.init_H_from_terms()

    if D <= 0.0:
        product_state = [0, 2] * (M.lat.N_sites//2)  # initial state down = 0, 0 = 1, up = 2
    else:
        product_state = [1] * M.lat.N_sites

    #options = {'method': 'lat_product_state',
    #...            'product_state' : [[['up'], ['down']],
    #...                               [['down'], ['up']]],
    #                'chi' : 50
    #...            }
    #options = {
    #    'method': 'lat_product_state',
    #    'product_state': ['|+x>' if i % 2 == 0 else '|-x>' for i in range(M.lat.N_sites)],  # Alternating up and down in X direction
    #}
    #psi = InitialStateBuilder(M.lat, nit_state_params).run()


    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': False,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-8,
        },
        'chi_list': {
            0: 50,
            4: 100,
        #    8: 200,
        #    12: 400,
        #    16: 600,
        },
        'max_E_err': 1.e-8,  # TODO go back to 1.e-8
        #'max_S_err': 1.e-6,
        #'norm_tol': 1.e-6,
        'max_sweeps': 30,
    }


    # TODO: COMPARE WITH ED!


    #eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    #eng.chi_list(600,50,2)
    ##E,psi = eng.run()
    info = dmrg.run(psi, M, dmrg_params)
    log_sweep_statistics(L, alpha, D, info['sweep_statistics'])
    print("final bond dimensions: ", psi.chi)
    #print("E = {E:.13f}".format(E=E))



    M_eps = SpinChain(model_params)
    psi_eps = MPS.from_product_state(M_eps.lat.mps_sites(), product_state, bc=M_eps.lat.bc_MPS)

    #dmrg_params['chi_list'] = {0: 100}
    info = dmrg.run(psi_eps, M_eps, dmrg_params)
    log_sweep_statistics(L, alpha, D, info['sweep_statistics'])
    print("final bond dimensions: ", psi.chi)


    fidelity = calc_fidelity(psi, psi_eps, eps)

    return fidelity

def process_task(L, alpha, D, eps, lock):
    print(f"Thread processing task with parameter D={D}")
    print("-" * 100)

    fidelity = dmrg_lr_spinone_heisenberg_finite_fidelity(L=L, D=D, alpha=alpha, eps=eps)
    print(L, D, alpha, eps, fidelity)
    with lock:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([D, fidelity])  # Append D and fidelity
        print("-" * 100)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # TODO: create functions for different type of schedulings
    # TODO: One for dynamic and the other one for static scheduling
    L = 20
    alpha = 10.0
    Ds = np.arange(-1.0,-0.7,0.02)
    #Ds = np.arange(0.,0.02,0.02)
    eps = 1e-4
    n_threads = 4

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_fidelity_alpha{alpha}_L{L}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["D", "fidelity"])


    start_time = time.time()

    # List of tasks to be processed
    tasks = [f"task_{i}" for i in range(len(Ds))]  # Example task list

    # create lock object
    lock = threading.Lock()

    # Create a ThreadPoolExecutor with n threads
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit tasks dynamically to the executor with the additional parameter
        future_to_task = {executor.submit(process_task, L, alpha, D, eps, lock): D for D in Ds}

        # Collect results as they are completed
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()
                print(f"Completed: {task}")
            except Exception as exc:
                print(f"Task {task} generated an exception: {exc}")


    print("--- %s seconds ---" % (time.time() - start_time))


    #eps = 1e-4
    #for D in np.arange(0.1,1.52,0.02):  # np.arange(-1.5, 1.5, 0.02):
    #for D in np.arange(1.5, 0.08, -0.02):  # np.arange(-1.5, 1.5, 0.02):
    #    print("-" * 100)
    #    print(D)

    #    E, psi = dmrg_spinone_heisenberg_finite(D=D)
    #    E_eps, psi_eps = dmrg_spinone_heisenberg_finite(D=D+eps)
        #print("-" * 100)
        #print("-" * 100)
    #    fidelity = calc_fidelity(psi, psi_eps, eps)
    #    with open(filename, mode='a', newline='') as file:
    #        writer = csv.writer(file)
    #        writer.writerow([D, fidelity])  # Append D and fidelity
        #print("E = {E:.13f}".format(E=E))
    #    print("-" * 100)


# Number of threads
#n_threads = 4

# List of tasks to be processed
#Ds = np.arange(-1.5,1.6,0.02)
#tasks = [f"task_{i}" for i in range(len(Ds))]  # Example task list

# Create a ThreadPoolExecutor with n threads
#with ThreadPoolExecutor(max_workers=n_threads) as executor:
    # Submit tasks dynamically to the executor with the additional parameter
#    future_to_task = {executor.submit(process_task,D): D for D in Ds}

    # Collect results as they are completed
 #   for future in as_completed(future_to_task):
   #     task = future_to_task[future]
  #      try:
  #          future.result()
  #          print(f"Completed: {task}")
  #      except Exception as exc:
  #          print(f"Task {task} generated an exception: {exc}")


