# padelhardt


import time
import csv
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinModel
from tenpy.models.spins import SpinChain
from tenpy.algorithms import dmrg


def dmrg_spinone_heisenberg_finite(D, conserve='best'):
    #print("finite DMRG, S=1 Anisotorpic Heisenberg chain")
    #print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Jz, conserve=conserve))
    model_params = dict(
        L=100,
        S=1.,  # spin 1/2
        D=D,  # couplings
        bc_MPS='finite',
        conserve='Sz')
    M = SpinChain(model_params)
    # TODO: How to init zero states?
    product_state = [0, 2] * (M.lat.N_sites//2)  # initial state down = 0, 0 = 1, up = 2
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
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
        'max_E_err': 1.e-8,
        #'max_S_err': 1.e-6,
        #'norm_tol': 1.e-6,
        'max_sweeps': 25,
    }
    #eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    #eng.chi_list(600,50,2)
    ##E,psi = eng.run()
    info = dmrg.run(psi, M, dmrg_params)
    #print(info)
    E = info['E']
    #eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    #E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    #psi = info['psi']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    #print(info)
    #print(f"Number of sweeps completed: {info['sweep_number']}")
    #Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    #mag_z = np.mean(Sz)
    #print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],Sz1=Sz[1],mag_z=mag_z))
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    #corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
    #print("correlations <Sz_i Sz_j> =")
    #print(corrs)
    return E, psi

def calc_fidelity(psi, psi_eps, eps):
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions
    return -2 * np.log(overlap) / (eps ** 2)  # fidelity susceptiblity


def process_task(D, eps, lock):
    print(f"Thread processing task with parameter D={D}")
    print("-" * 100)

    E, psi = dmrg_spinone_heisenberg_finite(D=D)
    E_eps, psi_eps = dmrg_spinone_heisenberg_finite(D=D + eps)
    # print("-" * 100)
    # print("-" * 100)
    fidelity = calc_fidelity(psi, psi_eps, eps)
    with lock:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([D, fidelity])  # Append D and fidelity
        # print("E = {E:.13f}".format(E=E))
        print("-" * 100)
        #return f"Result of task {task_id} with param {additional_param}"


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Open a file in write mode
    filename = 'spinone_heisenberg_fidelity_alphaInf_L100.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["D", "fidelity"])


    start_time = time.time()

    # Number of threads
    n_threads = 1
    eps = 1e-4

    # List of tasks to be processed
    Ds = np.arange(0.34,0.36,0.02)
    tasks = [f"task_{i}" for i in range(len(Ds))]  # Example task list

    # create lock object
    lock = threading.Lock()

    # Create a ThreadPoolExecutor with n threads
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit tasks dynamically to the executor with the additional parameter
        future_to_task = {executor.submit(process_task,D,eps, lock): D for D in Ds}

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


