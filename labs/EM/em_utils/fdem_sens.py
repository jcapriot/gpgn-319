from simpeg.electromagnetics import frequency_domain as fdem
from simpeg import maps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict
nested_dict = lambda: defaultdict(nested_dict)


class HalfspaceSensitivity:

    def __init__(self):
        """Initialize widget for sensitivities"""
        # EM31 parameters
        freq_31 = 9_800 # Hz
        sep_31 = 3.66 # m
        height_31 = 0.5 #m
        rx_locs_31 = [[0, sep_31, 0], [0, sep_31, height_31]]
        rx_31_hcp_surf = [
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[0], orientation='z', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[0], orientation='z', component='imag')
        ]
        rx_31_hcp_waist = [
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[1], orientation='z', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[1], orientation='z', component='imag')
        ]
        rx_31_vcp_surf = [
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[0], orientation='x', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[0], orientation='x', component='imag')
        ]
        rx_31_vcp_waist = [
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[1], orientation='x', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary(rx_locs_31[1], orientation='x', component='imag')
        ]
        src_31 = [
            fdem.sources.MagDipole(receiver_list=rx_31_hcp_surf, location=[0, 0, 0], orientation='z',
                                   frequency=freq_31),
            fdem.sources.MagDipole(receiver_list=rx_31_hcp_waist, location=[0, 0, height_31], orientation='z',
                                   frequency=freq_31),
            fdem.sources.MagDipole(receiver_list=rx_31_vcp_surf, location=[0, 0, 0], orientation='x', frequency=freq_31),
            fdem.sources.MagDipole(receiver_list=rx_31_vcp_waist, location=[0, 0, height_31], orientation='x',
                                   frequency=freq_31),
        ]

        #EM34 parameters
        rx_34_hcp_10 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 10, 0], orientation='z', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 10, 0], orientation='z', component='imag'),
        ]
        rx_34_hcp_20 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 20, 0], orientation='z', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 20, 0], orientation='z', component='imag'),
        ]
        rx_34_hcp_40 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 40, 0], orientation='z', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 40, 0], orientation='z', component='imag'),
        ]
        src_34_hcp = [
            fdem.sources.MagDipole(receiver_list=rx_34_hcp_10, frequency=6400, location=[0, 0, 0], orientation='z'),
            fdem.sources.MagDipole(receiver_list=rx_34_hcp_20, frequency=1600, location=[0, 0, 0], orientation='z'),
            fdem.sources.MagDipole(receiver_list=rx_34_hcp_40, frequency=400, location=[0, 0, 0], orientation='z'),
        ]

        rx_34_vcp_10 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 10, 0], orientation='x', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 10, 0], orientation='x', component='imag'),
        ]
        rx_34_vcp_20 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 20, 0], orientation='x', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 20, 0], orientation='x', component='imag'),
        ]
        rx_34_vcp_40 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 40, 0], orientation='x', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 40, 0], orientation='x', component='imag'),
        ]
        src_34_vcp = [
            fdem.sources.MagDipole(receiver_list=rx_34_vcp_10, frequency=6400, location=[0, 0, 0], orientation='x'),
            fdem.sources.MagDipole(receiver_list=rx_34_vcp_20, frequency=1600, location=[0, 0, 0], orientation='x'),
            fdem.sources.MagDipole(receiver_list=rx_34_vcp_40, frequency=400, location=[0, 0, 0], orientation='x'),
        ]

        rx_34_vca_10 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 10, 0], orientation='y', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 10, 0], orientation='y', component='imag'),
        ]
        rx_34_vca_20 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 20, 0], orientation='y', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 20, 0], orientation='y', component='imag'),
        ]
        rx_34_vca_40 = [
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 40, 0], orientation='y', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary([0, 40, 0], orientation='y', component='imag'),
        ]
        src_34_vca = [
            fdem.sources.MagDipole(receiver_list=rx_34_vca_10, frequency=6400, location=[0, 0, 0], orientation='y'),
            fdem.sources.MagDipole(receiver_list=rx_34_vca_20, frequency=1600, location=[0, 0, 0], orientation='y'),
            fdem.sources.MagDipole(receiver_list=rx_34_vca_40, frequency=400, location=[0, 0, 0], orientation='y'),
        ]

        src_34 = src_34_hcp + src_34_vcp + src_34_vca

        # GEM2
        gem2_frequencies = [450, 1530, 5310, 18330, 63030]
        rx_gem2_hcp = [
            fdem.receivers.PointMagneticFluxDensitySecondary(locations=[0, 3*1.6, 0.5], orientation='z', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary(locations=[0, 3*1.6, 0.5], orientation='z', component='imag'),
        ]
        rx_gem2_vcp = [
            fdem.receivers.PointMagneticFluxDensitySecondary(locations=[0, 3*1.6, 0.5], orientation='x', component='real'),
            fdem.receivers.PointMagneticFluxDensitySecondary(locations=[0, 3*1.6, 0.5], orientation='x',
                                                             component='imag'),
        ]
        src_gem2 = [
            fdem.sources.MagDipole(rx_gem2_hcp, frequency=freq, location=[0, 0, 0.5], orientation='z') for freq in gem2_frequencies
        ] + [
            fdem.sources.MagDipole(rx_gem2_vcp, frequency=freq, location=[0, 0, 0.5], orientation='x') for freq in
            gem2_frequencies

        ]
        survey = fdem.Survey(src_31 + src_34 + src_gem2)

        n_layers = 501
        h = np.logspace(-2, 1, n_layers-1)
        sigma = np.full(n_layers, 0.01)
        sigma_mapping = maps.IdentityMap()
        self.sim = fdem.Simulation1DLayered(survey=survey, thicknesses=h, sigmaMap=sigma_mapping)
        self.J = self.sim.getJ(sigma)['ds']
        self.sens = self.J[:, :-1]/h
        self.depths = -(self.sim.depth[1:] + self.sim.depth[:-1])/2

    @property
    def data_map(self):
        x = nested_dict()
        i = 0
        for orient in ['hcp','vcp']:
            for height in ['surface', 'waist']:
                for comp in ['real', 'imag']:
                    x['em31'][orient][height][comp] = i
                    i += 1

        for orient in ['hcp', 'vcp', 'vca']:
            for offset in [10, 20, 40]:
                for comp in ['real', 'imag']:
                    x['em34'][orient][offset][comp] = i
                    i += 1

        for orient in ['hcp', 'vcp']:
            for frequency in [450, 1530, 5310, 18330, 63030]:
                for comp in ['real','imag']:
                    x['gem2'][orient][frequency][comp] = i
                    i += 1
        return x

        
                
            