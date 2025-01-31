from simpeg.electromagnetics import frequency_domain as fdem
from simpeg import maps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, set_loglevel
from collections import defaultdict

import matplotlib.pyplot as plt
from ipywidgets import HTML, VBox, Label, Widget, FloatSlider, HBox, IntSlider, Layout, ToggleButtons, Output, Button, \
    FloatLogSlider, Dropdown, RadioButtons

# Function to set and snap the value
def set_slider_value(slider, value):
    snapped_value = round(value / slider.step) * slider.step  # Round to the nearest step
    slider.value = snapped_value  # Set the slider value

def snap_to_log_slider(log_slider, value):
    log_base = log_slider.base
    min_exp = log_slider.min
    max_exp = log_slider.max
    step = log_slider.step

    # Convert value to log-space
    log_value = np.log(value)/np.log(log_base)

    # Clip log_value within slider range
    log_value = np.clip(log_value, min_exp, max_exp)

    # Snap to nearest step
    snapped_log_value = round((log_value - min_exp) / step) * step + min_exp

    # Convert back to linear scale and update slider
    log_slider.value = log_base ** snapped_log_value



class HalfspaceSensitivity:

    def __init__(self):
        """Initialize widget for sensitivities"""


        self.rx_offset = FloatLogSlider(
            min=-1, max=2,
            step=0.01,
            continuous_update=True,
            orientation='horizontal',
            description='Offset'
        )

        self.height = FloatSlider(
            min=0, max=5,
            step=0.01,
            continuous_update=True,
            orientation='horizontal',
            description='Height'
        )

        self.frequency = FloatLogSlider(
            min=1, max=5.5,
            step=0.01,
            continuous_update=True,
            orientation='horizontal',
            description='Tx frequency'
        )

        self.conductivity = FloatLogSlider(
            min=-5, max=1,
            value=0.01,
            step=0.01,
            continuous_update=True,
            orientation='horizontal',
            description='Conductivity'
        )

        self.orientation = RadioButtons(
            options=['HCP', 'VCP', "VCA"],
            description='Orientation',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['vertical dipole', 'horizontal dipole', 'horizontal dipole'],
        )

        self.rx_component = RadioButtons(
            options=['real', 'imag'],
            description='Component view',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Display the in-phase data', 'Display the quadrature data',],
        )


        self.n_layers = 201
        self.h = np.logspace(-2, 1, self.n_layers-1)
        self.set_preset("em31", update_plot=False)
        sens, depths = self.calc_sens()
        self.depths = depths

        # Create a figure and axis
        with plt.ioff():
            fig = plt.figure()
            ax = plt.gca()

        ax.set_xscale('log')
        ax.set_ylabel("|Sensitivity|")
        self.line, = ax.plot(depths, sens)
        self.ax = ax


        presets = Dropdown(
            options=[
                'em31', 'em34_10m', 'em34_20m', 'em34_40m',
                "gem2_450Hz", "gem2_1530Hz", "gem2_5310Hz", "gem2_18330Hz", "gem2_63030Hz"
            ],
            value='em31',
            description='Presets',
            disabled=False,
        )

        presets.observe(lambda change: self.set_preset(change.new), names='value')

        self.conductivity.observe(self.update_plot, names='value')
        self.rx_offset.observe(self.update_plot, names='value')
        self.height.observe(self.update_plot, names='value')
        self.frequency.observe(self.update_plot, names='value')
        self.orientation.observe(self.update_plot, names='value')
        self.rx_component.observe(self.update_plot, names='value')


        slider_box = VBox([self.conductivity, self.rx_offset, self.height, self.frequency])
        radio_box = HBox([self.orientation, self.rx_component])

        toggle_box = VBox([slider_box, radio_box, presets],
                          layout=Layout(width="30%"))

        self._box = HBox([toggle_box, fig.canvas])

        self._updating_presets = False

    def display(self):
        return self._box

    def update_plot(self, change):
        if not self._updating_presets:
            sens, _ = self.calc_sens()
            self.line.set_ydata(sens)
            vmin = sens.min()
            vmax = sens.max()
            span = vmax - vmin
            self.ax.set_ylim([vmin-0.1*span, vmax+0.1*span])

    def calc_sens(self):
        rx_offset = self.rx_offset.value
        orientation = self.orientation.value
        orientation = {"HCP":"z", "VCP":"x", "VCA":"y"}[orientation]
        rx_component = self.rx_component.value
        height = self.height.value

        location = np.r_[0, rx_offset, height]
        rx = fdem.receivers.PointMagneticFluxDensitySecondary(locations=location, orientation=orientation, component=rx_component)

        freq = self.frequency.value
        location = np.r_[0, 0, height]

        src = fdem.sources.MagDipole([rx], location=location, orientation=orientation, frequency=freq)

        srv = fdem.Survey([src])
        model = np.full(self.n_layers, self.conductivity.value)
        sigma_mapping = maps.IdentityMap()

        h = self.h
        sim = fdem.Simulation1DLayered(survey=srv, thicknesses=h, sigmaMap=sigma_mapping)
        J = sim.getJ(model)['ds']
        sens = J[:, :-1]/self.h

        depths = -(sim.depth[1:] + sim.depth[:-1]) / 2
        return sens[0], depths

    def set_preset(self, preset, update_plot=True):
        try:
            self._updating_presets = True
            options = PRESETS[preset]
            for option, value in options.items():
                option = getattr(self, option)
                if isinstance(option, FloatSlider):
                    set_slider_value(option, value)
                elif isinstance(option, FloatLogSlider):
                    snap_to_log_slider(option, value)
                else:
                    option.value = value
        except:
            pass
        finally:
            self._updating_presets = False

        if update_plot:
            self.update_plot(None)



PRESETS = {
    "em31":{"frequency":9800, "rx_offset":3.3, "orientation":"HCP", "rx_component":"imag", "height":0.5},
    "em34_10m":{"frequency":6400, "rx_offset":10, "orientation":"HCP", "rx_component":"imag", "height":0},
    "em34_20m": {"frequency": 1600, "rx_offset":20, "orientation":"HCP", "rx_component":"imag", "height":0},
    "em34_40m": {"frequency": 400, "rx_offset":40, "orientation":"HCP", "rx_component":"imag", "height":0},
    "gem2_450Hz": {"frequency": 450, "rx_offset":1.6, "orientation":"HCP", "rx_component":"imag", "height":0.5},
    "gem2_1530Hz": {"frequency": 1530, "rx_offset":1.6, "orientation":"HCP", "rx_component":"imag", "height":0.5},
    "gem2_5310Hz": {"frequency": 5310, "rx_offset":1.6, "orientation":"HCP", "rx_component":"imag", "height":0.5},
    "gem2_18330Hz": {"frequency": 18330, "rx_offset":1.6, "orientation":"HCP", "rx_component":"imag", "height":0.5},
    "gem2_63030Hz": {"frequency": 63030, "rx_offset":1.6, "orientation":"HCP", "rx_component":"imag", "height":0.5},
}
        
                
            