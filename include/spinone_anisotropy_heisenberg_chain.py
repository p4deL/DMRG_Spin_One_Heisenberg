from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinSite


class LongRangeSpinOneChain(CouplingMPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i (-1)^i S^z_i + D \sum_i (S^z_i)^2
       """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        sort_charge = model_params.get('sort_charge', True)
        if conserve == 'best' or conserve == 'Sz':
            return SpinSite(S=1., conserve='Sz', sort_charge=sort_charge)
        else:
            return SpinSite(S=1., conserve=None, sort_charge=sort_charge)

    def init_terms(self, model_params):
        B = model_params.get('B', 0.)
        D = model_params.get('D', 0.)

        for u in range(len(self.lat.unit_cell)):
            # staggered auxillary field
            Bstag = [B, -B] * (self.lat.N_sites // 2)
            self.add_onsite(Bstag, 0, 'Sz')
            # Sz anisotropy
            self.add_onsite(D, u, 'Sz Sz')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(1. / 2., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(1., u1, 'Sz', u2, 'Sz', dx)
