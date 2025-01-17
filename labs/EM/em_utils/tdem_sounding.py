import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import HTML, VBox, Label, Widget, FloatSlider, HBox, IntSlider, Layout, ToggleButtons, Output, Button
from scipy.constants import mu_0

from simpeg.electromagnetics import time_domain as tdem
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
import discretize

class TDEMSoundingInteract():

    def __init__(self, simpeg_survey, observed_voltage=None, standard_deviation=None, sigma_0=1):
        
        n_layer = 1
        def check_dims(locs):
            locs = np.atleast_1d(locs)

            if locs.ndim == 1:
                locs = locs[:, None]
            if locs.ndim == 2 and locs.shape[1] < 3:
                locs = np.pad(locs, [(0, 0), (0, 3-locs.shape[1])])
            return locs

        sigma = np.repeat(sigma_0, n_layer)
        thick = 10**(np.repeat(2, n_layer-1))

        self._model = self._sigma_thick_to_model(sigma, thick)

        # volt/apparent conductivity display
        self._dtype_toggle = ToggleButtons(
            options=['Volts', 'Apparent Conductivity'],
            description='Data View',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Display the data in voltage', 'Display the data as apparent conductivity',],
        )

        if observed_voltage is not None:
            if standard_deviation is None:
                standard_deviation = 0.05 * np.abs(observed_voltage)
            dobs = data.Data(simpeg_survey, dobs=-observed_voltage, standard_deviation=standard_deviation)
            self._dobs = dobs
        else:
            self._dobs = None
        
        self._sim = tdem.Simulation1DLayered(survey=simpeg_survey)
        
        # Create a figure and axis
        with plt.ioff():
            fig, [ax_model, ax_data] = plt.subplots(1, 2)
        self._fig = fig
        
        toolbar = plt.get_current_fig_manager().toolbar
        ax_model.set_xscale('log')
        ax_model.set_yscale('log')

        ax_model.set_ylim([1.05 * self._model[-1, 1], 0.5 * self._model[1, 1]])
        ax_model.grid(True)

        ax_model.set_xlabel(r'Conductivity (S/m)')
        ax_model.set_ylabel(r'Depth (m)')

        ax_data.set_yscale('log')
        ax_data.set_xscale('log')
        ax_data.yaxis.set_label_position("right")
        ax_data.yaxis.tick_right()
        ax_data.set_ylabel('Normalized Decay Voltage (Volts/(m^2 A))')
        ax_data.grid(True)
        self._ax_data = ax_data
        self._ax_model = ax_model
        
        # Initialize the line with some initial data
        self._line_model, = ax_model.plot(self._model[:, 0], self._model[:, 1], color='C0')
        self._markers_model, = ax_model.plot(self._model[:-1, 0], self._model[:-1, 1], color='C0', marker='o', linewidth=0)
        self._end_mark_model, = ax_model.plot(self._model[[-1], 0], self._model[[-1], 1], color='C0', marker='v', linewidth=0)

        dpred = self._dpred()
        line_datas = {}
        i_d = 0
        i_c = 0
        for src in simpeg_survey.source_list:
            for rx in src.receiver_list:
                d = dpred[i_d:i_d+rx.nD]

                line_datas[src, rx], = ax_data.plot(rx.times, d, marker='.', linewidth=0, color=f"C{i_c}")
                i_d += rx.nD
                i_c += 1
                
        self._line_data = line_datas
        ax_data.set_xlabel(r'Times (s)')

        if observed_voltage is not None:
            line_datas = {}
            i_d = 0
            i_c = 0
            for src in simpeg_survey.source_list:
                for rx in src.receiver_list:
                    d = observed_voltage[i_d:i_d+rx.nD]
    
                    line_datas[src, rx], = ax_data.plot(rx.times, d, marker='+', linewidth=0, color=f"C{i_c}")
                    i_d += rx.nD
                    i_c += 1
            self._data_line = line_datas

        def toggle_obs(change):
            if self._dtype_toggle.value == 'Apparent Conductivity':
                ax_data.set_ylabel(r'$\sigma_a$ ($S/m$)')
            else:
                ax_data.set_ylabel('Normalized Decay Voltage (Volts/(m^2 A))')
            self._update_dpred_plot()

        self._dtype_toggle.observe(toggle_obs)
        
        self.__dragging = False
        self.__selection_point = None
        self.__segment_ind = None
        self.__segment_start = None
        
        # setup function for clicking a line:
        # Define a function for handling button press events
        def on_press(event):
            if event.inaxes == ax_model and toolbar.mode != 'pan/zoom':
                contains, attrd = self._line_model.contains(event)
                if contains:
                    selection_point = np.r_[event.xdata, event.ydata]
                    close_inds = attrd['ind']
                    if len(close_inds) == 0:
                        close_inds = np.array([-2])
                    closest = np.argmin(np.linalg.norm(selection_point - self._model[close_inds]))
                    segment_ind = close_inds[closest]
                    if event.key == 'shift':
                        # need to split segment and trigger redraw of data
                        nodes_before = self._model[:segment_ind+1]
                        nodes_after = self._model[segment_ind+1:]

                        if segment_ind % 2 == 0:
                            # horizontal use event.xdata
                            new_point = np.array([[nodes_before[-1, 0], selection_point[1]]])
                        else:
                            new_point = np.array([[selection_point[0], nodes_before[-1, 1]]])

                        # find closest point on line to selection
                        new_nodes = np.r_[nodes_before, new_point, new_point, nodes_after]
                        self._model = new_nodes
                        self._update_model_plot()
                        self._update_dpred_plot()
                    else:
                        self.__selection_point = selection_point
                        self.__segment_ind = segment_ind
                        self.__segment_start = self._model[segment_ind:segment_ind+2].copy()
                        self.__dragging = True

        # Define a function for handling mouse motion events
        def on_motion(event):
            if self.__dragging and event.inaxes == ax_model and toolbar.mode != 'pan/zoom':
                segment_ind = self.__segment_ind
                selection_point = self.__selection_point
                segment_start = self.__segment_start
                x, y = event.xdata, event.ydata
                # vertical line
                if segment_ind % 2 == 0:
                    dx = x - selection_point[0]
                    dy = 0
                # horizontal line
                else:
                    dx = 0
                    dy = y - selection_point[1]
                new_points = segment_start + [dx, dy]
                # if horizontal line, check for bounds on heights
                if segment_ind % 2 == 1:
                    new_y = new_points[0, 1]
                    # need to check if it is valid
                    if segment_ind + 2 < self._model.shape[0]:
                        next_y = self._model[segment_ind + 2, 1]
                        new_y = min(new_y, next_y)
                    new_y = max(new_y, self._model[segment_ind - 1, 1])
                    new_points[:, 1] = new_y

                # put some reasonable guardrails on interacting.
                # new_points = np.maximum(1E-15, new_points)
                # new_points = np.minimum(1E15, new_points)

                self._model[segment_ind:segment_ind + 2] = new_points
                self._update_model_plot()
                self._update_dpred_plot()

        # Define a function for handling button release events
        def on_release(event):
            self.__dragging = False

        # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

        self._output = Output(layout={'border': '1px solid black'})

        self._invert_button = Button(
            description='Run an inversion',
            disabled=observed_voltage is None,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Invert',
            icon='square-left', # (FontAwesome names without the `fa-` prefix)
        )

        def button_call(change):
           self._run_inversion()

        self._invert_button.on_click(button_call)

        delete_button = Button(
           description='Remove Last Layer',
           disabled=False,
           button_style='',  # 'success', 'info', 'warning', 'danger' or ''
           tooltip='remove the last layer of the model',
           icon='x',  # (FontAwesome names without the `fa-` prefix)
        )
        def delete_call(change):
            if len(self._model) > 2:
                self._model = self._model[:-2]
                self._update_model_plot()
                self._update_dpred_plot()

        delete_button.on_click(delete_call)

        button_box = HBox([self._invert_button, delete_button, self._dtype_toggle])

        self._box = VBox([button_box, fig.canvas, self._output])

    def _dpred(self, volts=False):
        sigma = self._model[::2, 0]
        thick = np.diff(self._model[::2, 1])
        sim = self._sim
        sim.sigma = sigma
        sim.thicknesses = thick
        return -sim.dpred(None)
    
    def _update_model_plot(self):
        self._line_model.set_data(np.atleast_1d(self._model[:, 0]), np.atleast_1d(self._model[:, 1]))
        self._markers_model.set_data(np.atleast_1d(self._model[:-1, 0]), np.atleast_1d(self._model[:-1, 1]))
        self._end_mark_model.set_data(np.atleast_1d(self._model[-1, 0]), np.atleast_1d(self._model[-1, 1]))
        
    def _update_dpred_plot(self):
        sim = self._sim
        dpred = self._dpred()
        if self._plot_app_cond:
            dpred = self._app_cond_transform(dpred)

        d_axes = [dpred.min(), dpred.max()]

        i_d = 0
        for src in sim.survey.source_list:
            for rx in src.receiver_list:
                d = dpred[i_d:i_d+rx.nD]
                self._line_data[src, rx].set_ydata(np.atleast_1d(d))
                i_d += rx.nD

        if self._dobs is not None:
            d_v = -self._dobs.dobs
            if self._plot_app_cond:
                d_v = self._app_cond_transform(d_v)
            d_axes = [min(dpred.min(), d_v.min()), max(dpred.max(), d_v.max())]
            i_d = 0
            for src in sim.survey.source_list:
                for rx in src.receiver_list:
                    d = d_v[i_d:i_d+rx.nD]
                    self._data_line[src, rx].set_ydata(np.atleast_1d(d))
                    i_d += rx.nD

        self._ax_data.set_ylim([0.75*d_axes[0], 1.25 * d_axes[1]])

    def _app_cond_transform(self, data):
        sim = self._sim

        app_cond = []
        i_d = 0
        for src in sim.survey.source_list:
            for rx in src.receiver_list:
                v = data[i_d:i_d+rx.nD]
                t = rx.times
                temp = v**2 * t**5 / (src.moment**2 * mu_0**5) * 20**2
                app_cond.append(temp**(1/3) * np.pi)
                i_d += rx.nD
        return np.concatenate(app_cond)
        

    @property
    def _plot_app_cond(self):
        return self._dtype_toggle.value == "Apparent Conductivity"

    def display(self):
        return self._box

    def get_data(self):
        """Get the data associated with the current model.

        Returns
        -------
        numpy.ndarray
            The forward modeled data in volts.
        """
        return self._dpred(volts=True)

    def get_model(self):
        """

        Returns
        -------
        sigma : (n_layer,) numpy.ndarray
            conductivity of each layer, from the surface downward, in S/m.
        thicknesses : (n_layer-1,) numpy.ndarray
            thicknesses of each layer, from the surface downward, in meters.
        """
        sigma = self._model[::2, 0]
        thick = np.diff(self._model[::2, 1])

        return sigma, thick

    def _sigma_thick_to_model(self, conductivity, thicknesses):
        sigmas = np.c_[conductivity, conductivity].reshape(-1)
        if len(thicknesses) == 0:
            depths = np.r_[0.0, 100.0]
        else:
            thicks = np.r_[thicknesses, thicknesses[-1]*10]
            depths = np.cumsum(thicks)
            depths = np.r_[0, np.c_[depths, depths].reshape(-1)[:-1]]
        return np.c_[sigmas, depths]

    def set_model(self, conductivities, thicknesses=None):
        """Sets the current model to have the given conductivity and thicknesses.

        Parameters
        ----------
        conductivity : (n_layer, ) array_like
            conductivity of each layer, from the surface downward, in ohm*m.
        thicknesses : (n_layer-1, ) array_like
            thicknesses of each layer, from the surface downward, in meters.
            if n_layer == 1, this is optional.
        """
        conductivities = np.atleast_1d(conductivities)
        n_layer = conductivities.shape[0]

        if n_layer == 1 and thicknesses is None:
            thicknesses = np.array([])
        thicknesses = np.atleast_1d(thicknesses)

        if thicknesses.shape[0] != n_layer - 1:
            raise ValueError(
                'Incompatible number of conductivities and thicknesses. '
                f'There were {thicknesses.shape[0]}, but I expected {n_layer - 1}.'
            )

        self._model = self._sigma_thick_to_model(conductivities, thicknesses)
        self._update_model_plot()

        # update the axis limits
        xlim = [0.95 * conductivities.min(), 1.05 * conductivities.max()]
        ylim = [1.05 * self._model[-1, 1], 0.5 * self._model[1, 1]]

        self._ax_model.set_xlim(xlim)
        self._ax_model.set_ylim(ylim)
        self._update_dpred_plot()

    def _run_inversion(self):
        # get the current initial model
        if self._dobs is None:
            return
        dobs = self._dobs
        sigma = self._model[::2, 0]
        thick = np.diff(self._model[::2, 1])
        thick = np.maximum(thick, 0.1)

        # determine if the number of unknowns is less than the number of data points
        underdetermined = (len(sigma) + len(thick)) > dobs.nD

        # if underdetermined need to set up a regularization
        # set the mappings for the simulation
        n_layers = len(sigma)
        if n_layers > 1:
            mapping = maps.Wires(('sigma', n_layers), ('thick', n_layers-1))
            self._sim.sigma = None
            self._sim.thicknesses = None
            self._sim.sigmaMap = maps.ExpMap() * mapping.sigma
            self._sim.thicknessesMap = maps.ExpMap() * mapping.thick

            init_model = np.log(np.r_[sigma, thick])
        else:
            self._sim.sigma = None
            self._sim.thicknesses = []
            self._sim.sigmaMap = maps.ExpMap(nP=1)

            init_model = np.log(sigma)
        mesh = discretize.TensorMesh([len(init_model)])

        # Define the data misfit. Here the data misfit is the L2 norm of the weighted
        # residual between the observed data and the data predicted for a given model.
        # Within the data misfit, the residual between predicted and observed data are
        # normalized by the data's standard deviation.
        dmis = data_misfit.L2DataMisfit(simulation=self._sim, data=dobs)

        # Define the regularization (model objective function)
        reg = regularization.WeightedLeastSquares(
            mesh, alpha_s=1.0, alpha_x=0.0, reference_model=init_model
        )

        # Define how the optimization problem is solved. Here we will use an inexact
        # Gauss-Newton approach that employs the conjugate gradient solver.
        opt = optimization.InexactGaussNewton(maxIter=30, maxIterCG=40)

        # Define the inverse problem
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # Setting a stopping criteria for the inversion.
        target_misfit = directives.TargetMisfit(chifact=1)

        if underdetermined:
            # Apply and update sensitivity weighting as the model updates
            update_sensitivity_weights = directives.UpdateSensitivityWeights()

            # Defining a starting value for the trade-off parameter (beta) between the data
            # misfit and the regularization.
            starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

            # Set the rate of reduction in trade-off parameter (beta) each time the
            # the inverse problem is solved. And set the number of Gauss-Newton iterations
            # for each trade-off parameter value.
            beta_schedule = directives.BetaSchedule(coolingFactor=5.0, coolingRate=3)

            # The directives are defined as a list.
            directives_list = [
                update_sensitivity_weights,
                starting_beta,
                beta_schedule,
                target_misfit,
            ]
        else:
            inv_prob.beta = 0.0

            # The directives are defined as a list.
            directives_list = [
            ]

        self._output.clear_output()
        with self._output:
            # Here we combine the inverse problem and the set of directives
            inv = inversion.BaseInversion(inv_prob, directives_list)

            # Run the inversion
            recovered_model = inv.run(init_model)
        self._sim.model = recovered_model
        sigma = self._sim.sigma
        thick = self._sim.thicknesses

        self._sim.sigmaMap = None
        self._sim.thicknessesMap = None
        self._sim.model = None

        self.set_model(sigma, thick)
        