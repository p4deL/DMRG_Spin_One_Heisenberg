# padelhardt
import sys

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
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS

from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import dmrg



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


def calc_fidelity(psi, psi_eps, eps):
    overlap = np.abs(psi.overlap(psi_eps))  # contract the two mps wave functions
    return -2 * np.log(overlap) / (eps ** 2)  # fidelity susceptiblity


def calc_fidelity_ed(psi, psi_eps, eps):
    overlap = npc.inner(psi, psi_eps, axes='range', do_conj=True)
    print(f"ed_overlap: {overlap}")
    return -2 * np.log(overlap) / (eps ** 2)  # fidelity susceptiblity


class LongRangeSpin1Chain(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    #def init_sites(self, model_params):
    #    B = model_params.get('B', 0.)
    #    conserve = model_params.get('conserve', 'best')
    #    if conserve == 'best':
    #        conserve = 'Sz' if not model_params.any_nonzero(['B']) else None
    #        self.logger.info("%s: set conserve to %s", self.name, conserve)
    #    sort_charge = model_params.get('sort_charge', True)
    #    return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            return SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        else:
            return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, 'Sx')
            self.add_onsite(D, u, 'Sz Sz')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            print(f"dx={dx}")
            print(f"u1={u1}, u2={u2}")
            self.add_coupling(J / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)

        #for dist in range(2, L):  # Only add for j > i to avoid double counting
        #    # print(dist)
        #    strength = (-1) ** (dist + 1) / (dist ** alpha)  # Long-range decay
        #    self.add_coupling(strength, 0, "Sz", 0, "Sz", dx=dist)
        #    self.add_coupling(0.5 * strength, 0, "Sp", 0, "Sm", dx=dist, plus_hc=True)



def dmrg_lr_spinone_heisenberg_finite_fidelity(L=10, alpha=10.0, D=0.0, eps=1e-4, conserve='Sz'):
    model_params = dict(
        L=L,
        D=D,  # couplings
        bc_MPS='finite',
        conserve=conserve)
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
        'max_sweeps': 30,
    }


    # create spine one model
    M = LongRangeSpin1Chain(model_params)
    model_params['D'] = D+eps  # FIXME: Could something go wrong here?
    M_eps = LongRangeSpin1Chain(model_params)

    # create initial state
    if D <= 0.0:
        product_state = [0, 2] * (L//2)  # initial state down = 0, 0 = 1, up = 2
    else:
        product_state = [1] * L

    # initial guess mps
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    psi_eps = MPS.from_product_state(M_eps.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    # run dmrg
    info = dmrg.run(psi, M, dmrg_params)
    info_eps = dmrg.run(psi_eps, M_eps, dmrg_params)  # TODO: Alternatively, I could feed the previous psi

    # calculate fidelity
    fidelity = calc_fidelity(psi, psi_eps, eps)

    # output to check sanity
    print("E = {E:.13f}".format(E=info['E']))
    print("final bond dimensions psi: ", psi.chi)
    print("E_eps = {E:.13f}".format(E=info_eps['E']))
    print("final bond dimensions psi_eps: ", psi_eps.chi)

    return fidelity


def ed_lr_spinone_heisenberg_finite_fidelity(L=10, alpha=10.0, D=0.0, eps=1e-4, conserve='Sz'):
    model_params = dict(
        L=L,
        D=D,  # anisotropy coupling
        bc_MPS='finite',
        conserve=conserve
    )

    # create spine one model
    M = LongRangeSpin1Chain(model_params)
    model_params['D'] = D+eps  # FIXME: Could something go wrong here?
    M_eps = LongRangeSpin1Chain(model_params)

    #M = SpinChain(model_params)
    #model_params['D'] = D+eps  # FIXME: Could something go wrong here?
    #M_eps = SpinChain(model_params)

    # charge sector
    charge_sector = [0]

    # ED runs
    #ED = ExactDiag(M, charge_sector=charge_sector)
    ED = ExactDiag(M, charge_sector=charge_sector, max_size=4.e9)
    ED.build_full_H_from_mpo()
    #E, psi = ED.sparse_diag(k=1, which='SA')  # FIXME: not sure how to make this work
    ED.full_diagonalization()
    E, psi = ED.groundstate()

    ED_eps = ExactDiag(M_eps, charge_sector=charge_sector, max_size=4.e9)
    ED_eps.build_full_H_from_mpo()
    #E_eps, psi_eps = ED_eps.sparse_diag(k=1, which='SA')
    ED_eps.full_diagonalization()
    E_eps, psi_eps = ED_eps.groundstate()

    fidelity = calc_fidelity_ed(psi, psi_eps, eps)

    return fidelity


def process_task(L, alpha, D, eps, lock):
    print(f"Thread processing task with parameter D={D}")
    print("-" * 100)

    fidelity_dmrg = dmrg_lr_spinone_heisenberg_finite_fidelity(L=L, D=D, alpha=alpha, eps=eps)
    fidelity_ed = ed_lr_spinone_heisenberg_finite_fidelity(L=L, D=D, alpha=alpha, eps=eps)
    print(L, D, alpha, eps, fidelity_dmrg, fidelity_ed)
    with lock:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([D, fidelity_dmrg, fidelity_ed])  # Append D and fidelity
        print("-" * 100)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # TODO: create functions for different type of schedulings
    # TODO: One for dynamic and the other one for static scheduling
    L = 10
    alpha = 10.0
    Ds = np.arange(-1.5,1.5,0.02)
    #Ds = np.arange(0.,0.02,0.02)
    eps = 1e-4
    n_threads = 4

    # Open a file in write mode
    filename = f'output/spinone_heisenberg_fidelity_alpha{alpha}_L{L}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["D", "fidelity_dmrg", "fidelity_ed"])


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
