import math
import numpy as np

from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinSite
from tenpy.tools.fit import sum_of_exp

import include.utilities as utilities


class LongRangeSpinOneChain(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = \sum_i \vec{S}_i \cdot \vec{S}_{j} + \Gamma \sum_{j>i+1} (-1)^{i-j+1}/|i-j|^{\alpha}} \vec{S}_i \cdot \vec{S}_{j} + B S^z_0 + + D \sum_i (S^z_i)^2
           H = (1-\Gamma)\sum_i \vec{S}_i \cdot \vec{S}_{j} + \Gamma \sum_{j>i} (-1)^{i-j+1}/|i-j|^{\alpha}} \vec{S}_i \cdot \vec{S}_{j} + B S^z_0 + + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            spin_site = SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        elif conserve == 'parity':
            spin_site = SpinSite(S=1., conserve='parity', sort_charge=sort_charge)
        else:
            spin_site = SpinSite(S=1., conserve=None, sort_charge=sort_charge)

        return spin_site

    def init_terms(self, model_params):
        Gamma = model_params.get('Gamma', 0.)
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)
        Delta = model_params.get('Delta', 1.)
        alpha = model_params.get('alpha', float('inf'))
        n_exp = model_params.get('n_exp', 2)  # Number of exponentials in fit
        fit_range = model_params.get('fit_range', self.lat.N_sites)  # Range of fit for decay

        # add on-site terms
        # staggered auxillary field
        #Bstag = [B, -B] * (self.lat.N_sites // 2)
        #self.add_onsite(Bstag, 0, 'Sz')
        # single-site aux field to lift edge-state degeneracy
        self.add_onsite_term(B, 0, 'Sz')
        #self.add_onsite_term(-B, self.lat.N_sites-1, 'Sz')  # TODO: Try if things get better if I include two terms
        # Sz anisotropy
        self.add_onsite(D, 0, 'Sz Sz')

        if math.isinf(alpha):
            # add only NN term
            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                self.add_coupling(1. / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
                self.add_coupling(Delta, u1, 'Sz', u2, 'Sz', dx)
        else:
            # NN interactions
            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                self.add_coupling((1. - Gamma) / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
                self.add_coupling(Delta*(1. - Gamma), u1, 'Sz', u2, 'Sz', dx)

            # fit power-law decay with sum of exponentials
            lam, pref = utilities.fit_with_sum_of_exp(utilities.power_law_decay, alpha, n_exp, fit_range)
            x = np.arange(1, fit_range + 1)
            print("*" * 100)
            print(f'n_exp = {n_exp}')
            print('error in fit: {0:.3e}'.format(np.sum(np.abs(utilities.power_law_decay(x, alpha) - sum_of_exp(lam, pref, x)))))
            #print(lam, pref)
            #plot_fit(x, power_law_decay(x, alpha), sum_of_exp(lam, pref, x) )
            print("*" * 100)

            # rescale linear prefactors with coupling constant of long-range terms
            print(pref)
            pref = Gamma * pref
            print(pref)

            # add exponentially_decaying terms
            for pr, la in zip(pref, lam):
                self.add_exponentially_decaying_coupling(0.5*pr, la, 'Sp', 'Sm', plus_hc=True)
                self.add_exponentially_decaying_coupling(Delta*pr, la, 'Sz', 'Sz')
                # change sign of Ferro couplings
                prprime = -2*pr
                laprime = la**2
                # couplings on even sites
                even_sites = list(range(0,self.lat.N_sites,2))
                self.add_exponentially_decaying_coupling(0.5*prprime, laprime, 'Sp', 'Sm', subsites=even_sites, plus_hc=True)
                self.add_exponentially_decaying_coupling(Delta*prprime, laprime, 'Sz', 'Sz', subsites=even_sites)
                # couplings on odd sites
                odd_sites = list(range(1,self.lat.N_sites,2))
                self.add_exponentially_decaying_coupling(0.5*prprime, laprime, 'Sp', 'Sm', subsites=odd_sites, plus_hc=True)
                self.add_exponentially_decaying_coupling(Delta*prprime, laprime, 'Sz', 'Sz', subsites=odd_sites)