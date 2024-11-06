from tenpy.networks.site import SpinSite
from tenpy.models.model import CouplingModel, MPOModel, CouplingMPOModel
from tenpy.models.lattice import Lattice, Chain, Ladder


class LongRangeSpin1ChainTest(CouplingModel, MPOModel):
    r"""An example for a custom model, implementing the Hamiltonian of :arxiv:`1204.0704`.

       .. math ::
           H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + B \sum_i S^x_i + D \sum_i (S^z_i)^2
       """

    def __init__(self, model_params):
        # get model parameters
        L = model_params.get('L', 2)
        Nuc = L//2
        print(L, Nuc)
        J = model_params.get('J', 1.)
        D = model_params.get('D', 0.)
        alpha = model_params.get('alpha', 100.)  # FIXME no need for alpha anymore
        n_exp = model_params.get('n_exp', 1)  # Number of exponentials in fit
        fit_range = model_params.get('fit_range', 6)  # Range of fit for decay

        site1 = SpinSite(S=1., conserve='Sz')
        site2 = SpinSite(S=1., conserve='Sz')

        # initialize spin chain with tow site spin-one unit cell
        bc = 'periodic' if model_params['bc_MPS'] == 'infinite' else 'open'
        #lat = Ladder(L,[site1, site2], bc=bc, bc_MPS=model_params['bc_MPS'])
        lat = Lattice([Nuc],[site1, site2], bc=bc, bc_MPS=model_params['bc_MPS'])

        # initialize a coupling model
        CouplingModel.__init__(self, lat)

        # add on site terms
        self.add_onsite(D, 0, 'Sz Sz')
        self.add_onsite(D, 1, 'Sz Sz')

        # add nearest-neighbor coupling
        # intra-cell coupling
        self.add_coupling(J/2., 0, 'Sp', 1, 'Sm', 0, plus_hc=True)
        self.add_coupling(J, 0, 'Sz', 1, 'Sz', 0)
        # inter-cell coupling
        #self.add_coupling(J/2., 1, 'Sp', 0, 'Sm', 1, plus_hc=True)
        #self.add_coupling(J, 1, 'Sz', 0, 'Sz', 1)

        # add long-range interactions
        for shift in range(1, Nuc):
            # ferro couplings (even)
            dist = 2*shift
            strength = alternating_power_law_decay(dist, alpha)
            self.add_coupling(0.5 * J * strength, 0, "Sp", 0, "Sm", dx=shift, plus_hc=True)
            self.add_coupling(strength, 0, "Sz", 0, "Sz", dx=shift)
            self.add_coupling(0.5 * J * strength, 1, "Sp", 1, "Sm", dx=shift, plus_hc=True)
            self.add_coupling(strength, 1, "Sz", 1, "Sz", dx=shift)

            # af couplings (odd)
            dist = 2*shift-1
            strength = alternating_power_law_decay(dist, alpha)
            self.add_coupling(0.5 * J * strength, 1, "Sp", 0, "Sm", dx=shift, plus_hc=True)
            self.add_coupling(strength, 1, "Sz", 0, "Sz", dx=shift)
            dist = 2*shift+1
            strength = alternating_power_law_decay(dist, alpha)
            self.add_coupling(0.5 * J * strength, 0, "Sp", 1, "Sm", dx=shift, plus_hc=True)
            self.add_coupling(strength, 0, "Sz", 1, "Sz", dx=shift)

        # construct the Hamiltonian in the Matrix-Product-Operator (MPO) picture
        MPOModel.__init__(self, lat, self.calc_H_MPO())
