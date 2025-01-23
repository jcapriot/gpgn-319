import discretize
import numpy as np
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
import scipy.io

from ipywidgets import FloatLogSlider, HBox, VBox, Layout, AppLayout, interactive, IntSlider, widget, FloatText, FloatSlider, Checkbox


def get_loop_sim():
    """ Get an example tdem simulation with a single loop source """

    h_r = [(5., 2), (1, 16, -1.1), (1,16,1.1), (5., 50, 1.2)]
    h_z = [(5., 25, -1.2), (1, 16, -1.1),(1, 16, 1.1), (5., 25, 1.2)]
    mesh = discretize.CylindricalMesh([h_r, 1, h_z], origin=[0, 0, 'C'])
    
    sigma = np.ones(len(mesh))
    sigma[mesh.cell_centers[:,-1] > 0] = 1E-12
    
    source = tdem.sources.CircularLoop(location=[0, 0, 0], radius=51, waveform=tdem.sources.StepOffWaveform())
    srv = tdem.Survey(source)
    
    time_steps = [
        (1E-6, 40, 1.2)
    ]
    
    sim = tdem.Simulation3DMagneticFluxDensity(mesh, survey=srv, time_steps=time_steps, sigma=sigma, solver=get_default_solver())
    return sim


class JBPlotter():
    """For plotting J and B fields on a cylindrical mesh"""

    def __init__(self, sim, j, b, range_x, range_z, stream_nx=256, steam_nz=256, show=False):

        mesh = sim.mesh
        self.times = sim.times
        h_r, _, h_z = mesh.h
        h_x = np.r_[h_r[::-1], h_r]
    
        self.mesh_tens = discretize.TensorMesh([h_x, h_z], origin=['C', mesh.origin[-1]])

        j = j.reshape((*mesh.shape_edges_y, -1), order='F').squeeze()
        nt = j.shape[-1]
    
        # build up the whole thing...
        j_left = j[::-1]
        j_middle = np.zeros((1, *j.shape[1:]))
        j_right = j
    
        j = np.concatenate([-j_left, j_middle, j_right], axis=0).reshape((-1, nt), order='F')
        j = self.mesh_tens.average_node_to_cell @ j
        j[self.mesh_tens.cell_centers[:, -1]>=0, :] = np.nan

        self.j = j

        br, bz = b[:mesh.n_faces_x], b[mesh.n_faces_x:]
        br = br.reshape((*mesh.shape_faces_x, -1), order='F').squeeze()
        bz = bz.reshape((*mesh.shape_faces_z, -1), order='F').squeeze()
    
        bx = np.concatenate([-br[::-1], np.zeros((1, *br.shape[1:])), br], axis=0).reshape((-1, nt), order='F')
        bz = np.concatenate([bz[::-1], bz], axis=0).reshape((-1, nt), order='F')
        b = np.concatenate([bx, bz], axis=0)
        b_ccv = self.mesh_tens.average_face_to_cell_vector @ b
        bx, bz = b_ccv.reshape((2, -1, nt))

        self.range_x = range_x
        self.range_z = range_z
    
        nodes_x = np.linspace(*range_x, stream_nx+1)
        nodes_z = np.linspace(*range_z, steam_nz+1)
        hx = nodes_x[1:] - nodes_x[:-1]
        hz = nodes_z[1:] - nodes_z[:-1]
    
        self.stream_mesh = discretize.TensorMesh([hx, hz], [nodes_x[0], nodes_z[0]])
        stream_interp_mat = self.mesh_tens.get_interpolation_matrix(self.stream_mesh.cell_centers, 'cell_centers')

        self.bx = (stream_interp_mat @ bx).reshape((*self.stream_mesh.shape_cells, -1), order='F')
        self.bz = (stream_interp_mat @ bz).reshape((*self.stream_mesh.shape_cells, -1), order='F')

        if show:
            self.do_plot(0)

    def do_plot(self, time_index):

        j = self.j[:, time_index]
        bx = self.bx[..., time_index].T
        bz = self.bz[..., time_index].T

        fig = plt.figure(dpi=300)
        ax = plt.gca()

        #self._frame = display(fig, display_id=True)
    
        vmin = np.nanmin(j)
        vmax = np.nanmax(j)
        thresh = (vmax - vmin) * 0.5

        norm = SymLogNorm(thresh, vmin=vmin, vmax=vmax)

        j_im, = self.mesh_tens.plot_image(
            j, v_type='CC', ax=ax, range_x=self.range_x, range_y=self.range_z,
            pcolor_opts={'norm':norm}
        )
        plt.colorbar(j_im, format=lambda x, _: f"{x:.2E}")

        ax.set_title(rf'Time: {self.times[time_index]*1E6:.3E} $\mu$s')
        ax.set_ylabel('z')
        ax.axhline(0, color='k')

        s_mesh = self.stream_mesh
        s_plot = ax.streamplot(s_mesh.cell_centers_x, s_mesh.cell_centers_y, bx, bz, color='w')
        plt.show()


class AmpPhaseInteract:

    def __init__(self, amplitude=1, phase=45, frequency=0.1, figsize=None):

        with plt.ioff():
            #fig, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
            #fig, [ax_polar, ax_image] = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            axw, axz = axs

        toolbar = plt.get_current_fig_manager().toolbar
        fig.canvas.toolbar_visible

        z = amplitude * np.exp(1j * phase)
        line, = axz.plot([0, z.real], [0, z.imag], color='r', linestyle='--')
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        axz.spines['left'].set_position('center')
        axz.spines['bottom'].set_position('center')
        
        # Eliminate upper and right axes
        axz.spines['right'].set_color('none')
        axz.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        axz.xaxis.set_ticks_position('bottom')
        axz.yaxis.set_ticks_position('left')
    
        scatter = axz.scatter([z.real], [z.imag], color='C0')
        axz.set_xlim([-1.5, 1.5])
        axz.set_ylim([-1.5, 1.5])
        axz.set_xlabel('real')
        axz.set_ylabel('imag')
        axz.grid()
        axz.set_aspect(1)

        ts = np.linspace(-10, 10, 512)
        omega = 2 * np.pi * frequency
        wave, = axw.plot(ts, z.real * np.cos(ts * omega) + z.imag * np.sin(ts * omega), label='wave')
        cos, = axw.plot(ts, amplitude * np.cos(ts * omega), linestyle='--', alpha=0.5, label='cos')
        sin, = axw.plot(ts, amplitude * np.sin(ts * omega), linestyle='--', alpha=0.5, label='sin')
        axw.set_ylim([-1.5, 1.5])
        axw.set_xlim([-5, 5])
    
        axw.legend()
        

        freq_slider = FloatLogSlider(
            value=frequency,
            min=-2, max=2,
            continuous_update=True,
            orientation='horizontal',
        )

        def update_wave(x, y):
            omega = 2 * np.pi * freq_slider.value
            wave.set_ydata(x * np.cos(omega * ts) + y * np.sin(omega * ts))
            amp = np.sqrt(x*x + y * y)
            cos.set_ydata(amp * np.cos(omega * ts))
            sin.set_ydata(amp * np.sin(omega * ts))

        def slider_update(event):
            x, y = scatter.get_offsets()[0]
            update_wave(x, y)

        freq_slider.on_trait_change(slider_update)

        self.__dragging = False
        # # setup function for clicking a line:
        # # Define a function for handling button press events
        def on_press(event):
            if event.inaxes == axz and toolbar.mode != 'pan/zoom':
                contains, attrd = scatter.contains(event)
                if contains:
                    self.__dragging = True
        
        # Define a function for handling mouse motion events
        def on_motion(event):
            if self.__dragging and event.inaxes == axz and toolbar.mode != 'pan/zoom':
                x, y = event.xdata, event.ydata
                scatter.set_offsets([x, y])
                line.set_data([0, x], [0, y])
                update_wave(x, y)
        
        # Define a function for handling button release events
        def on_release(event):
            self.__dragging = False
        
        # # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

        box_layout = Layout(justify_content='center')

        #box = HBox([fig.canvas, freq_slider], layout=box_layout)
        box = AppLayout(
            center=fig.canvas,
            footer=freq_slider,
            pane_heights=[0, 6, 1]
        )
        self._box = box

    def display(self):
        return self._box

def mind(x, y, z, dincl, ddecl, x0, y0, z0, aincl, adecl):

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    x0 = np.array(x0, dtype=float)
    y0 = np.array(y0, dtype=float)
    z0 = np.array(z0, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    aincl = np.array(aincl, dtype=float)
    adecl = np.array(adecl, dtype=float)

    di = np.pi * dincl / 180.0
    dd = np.pi * ddecl / 180.0

    cx = np.cos(di) * np.cos(dd)
    cy = np.cos(di) * np.sin(dd)
    cz = np.sin(di)

    ai = np.pi * aincl / 180.0
    ad = np.pi * adecl / 180.0

    ax = np.cos(ai) * np.cos(ad)
    ay = np.cos(ai) * np.sin(ad)
    az = np.sin(ai)

    # begin the calculation
    a = x - x0
    b = y - y0
    h = z - z0

    rt = np.sqrt(a ** 2.0 + b ** 2.0 + h ** 2.0) ** 5.0

    txy = 3.0 * a * b / rt
    txz = 3.0 * a * h / rt
    tyz = 3.0 * b * h / rt

    txx = (2.0 * a ** 2.0 - b ** 2.0 - h ** 2.0) / rt
    tyy = (2.0 * b ** 2.0 - a ** 2.0 - h ** 2.0) / rt
    tzz = -(txx + tyy)

    bx = txx * cx + txy * cy + txz * cz
    by = txy * cx + tyy * cy + tyz * cz
    bz = txz * cx + tyz * cy + tzz * cz

    return bx * ax + by * ay + bz * az


def fem3loop(
    L, R, xc, yc, zc, dincl, ddecl, S, ht, f, xmin, xmax, dx, showDataPts=False
):

    L = np.array(L, dtype=float)
    R = np.array(R, dtype=float)
    xc = np.array(xc, dtype=float)
    yc = np.array(yc, dtype=float)
    zc = np.array(zc, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    S = np.array(S, dtype=float)
    ht = np.array(ht, dtype=float)
    f = np.array(f, dtype=float)
    dx = np.array(dx, dtype=float)

    ymin = xmin
    ymax = xmax
    dely = dx

    # generate the grid
    xp = np.arange(xmin, xmax, dx)
    yp = np.arange(ymin, ymax, dely)
    [y, x] = np.meshgrid(yp, xp)
    z = 0.0 * x - ht

    # set up the response arrays
    real_response = 0.0 * x
    imag_response = 0.0 * x

    # frequency characteristics
    alpha = 2.0 * np.pi * f * L / R

    f_factor = (alpha ** 2.0 + 1j * alpha) / (1 + alpha ** 2.0)

    # amin = 0.01
    # amax = 100.0
    da = 4.0 / 40.0
    alf = np.arange(-2.0, 2.0, da)
    alf = 10.0 ** alf

    fre = alf ** 2.0 / (1.0 + alf ** 2.0)
    fim = alf / (1.0 + alf ** 2.0)

    # simulate anomalies
    yt = y - S / 2.0
    yr = y + S / 2.0

    dm = -S / 2.0
    dp = S / 2.0

    M13 = mind(0.0, dm, 0.0, 90.0, 0.0, 0.0, dp, 0.0, 90.0, 0.0)
    M12 = L * mind(x, yt, z, 90.0, 0.0, xc, yc, zc, dincl, ddecl)
    M23 = L * mind(xc, yc, zc, dincl, ddecl, x, yr, z, 90.0, 0.0)

    c_response = -M12 * M23 * f_factor / (M13 * L)

    # scaled to simulate a net volumetric effect
    if np.logical_and(dincl == 0.0, ddecl == 0.0):
        real_response = np.real(c_response) * 0.0
        imag_response = np.imag(c_response) * 0.0
    else:
        real_response = np.real(c_response) * 1000.0
        imag_response = np.imag(c_response) * 1000.0

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    ax[0][0].semilogx(alf, fre, ".-b")
    ax[0][0].semilogx(alf, fim, ".--g")
    ax[0][0].plot([alpha, alpha], [0.0, 1.0], "-k")
    ax[0][0].legend(["Real", "Imag"], loc=2)
    ax[0][0].set_xlabel("$\\alpha = \\omega L /R$")
    ax[0][0].set_ylabel("Frequency Response")
    ax[0][0].set_title("Plot 1: EM responses of loop")
    ax[0][0].grid(which="major", color="0.6", linestyle="-", linewidth="0.5")
    ax[0][0].grid(which="minor", color="0.6", linestyle="-", linewidth="0.5")

    kx = int(np.ceil(xp.size / 2.0))
    ax[0][1].plot(y[kx, :], real_response[kx, :], ".-b")  # kx
    ax[0][1].plot(y[kx, :], imag_response[kx, :], ".--g")
    # ax[0][1].legend(['Real','Imag'],loc=2)
    ax[0][1].set_xlabel("Easting")
    ax[0][1].set_ylabel("H$_s$/H$_p$")
    ax[0][1].set_title("Plot 2: EW cross section along Northing = %1.1f" % (x[kx, 0]))
    ax[0][1].grid(which="major", color="0.6", linestyle="-", linewidth="0.5")
    ax[0][1].grid(which="minor", color="0.6", linestyle="-", linewidth="0.5")
    ax[0][1].set_xlim(np.r_[xmin, xmax])

    vminR = real_response.min()
    vmaxR = real_response.max()
    ax[1][0].plot(np.r_[xp.min(), xp.max()], np.zeros(2), "k--", lw=1)
    clb = plt.colorbar(
        ax[1][0].imshow(
            real_response,
            extent=[xp.min(), xp.max(), yp.min(), yp.max()],
            vmin=vminR,
            vmax=vmaxR,
        ),
        ax=ax[1][0],
    )
    ax[1][0].set_xlim(np.r_[xmin, xmax])
    ax[1][0].set_ylim(np.r_[xmin, xmax])
    ax[1][0].set_xlabel("Easting (m)")
    ax[1][0].set_ylabel("Northing (m)")
    ax[1][0].set_title("Plot 3: Real Component")
    # ax[1][0].colorbar()
    clb.set_label("H$_s$/H$_p$")

    if showDataPts:
        XP, YP = np.meshgrid(xp, yp)
        ax[1][0].plot(XP, YP, ".", color=[0.2, 0.2, 0.2])

    vminI = imag_response.min()
    vmaxI = imag_response.max()
    ax[1][1].plot(np.r_[xp.min(), xp.max()], np.zeros(2), "k--", lw=1)
    clb = plt.colorbar(
        ax[1][1].imshow(
            imag_response,
            extent=[xp.min(), xp.max(), yp.min(), yp.max()],
            vmin=vminI,
            vmax=vmaxI,
        ),
        ax=ax[1][1],
    )
    ax[1][1].set_xlim(np.r_[xmin, xmax])
    ax[1][1].set_ylim(np.r_[xmin, xmax])
    ax[1][1].set_xlabel("Easting (m)")
    ax[1][1].set_ylabel("Northing (m)")
    ax[1][1].set_title("Plot 4: Imag Component")
    clb.set_label("H$_s$/H$_p$")

    if showDataPts:
        ax[1][1].plot(XP, YP, ".", color=[0.2, 0.2, 0.2])

    plt.tight_layout()
    plt.show()


def interactfem3loop():

    S = 4.0
    ht = 1.0
    xmin = -10.0
    xmax = 10.0
    zmax = 10.0
    # xmin = lambda dx: -40.*dx
    # xmax = lambda dx: 40.*dx

    def fem3loopwrap(L, R, yc, xc, zc, dincl, ddecl, f, dx, showDataPts):
        return fem3loop(
            L, R, -yc, xc, zc, dincl, ddecl, S, ht, f, xmin, xmax, dx, showDataPts
        )

    Q = interactive(
        fem3loopwrap,
        L=FloatSlider(
            min=0.00, max=0.20, step=0.01, value=0.10, continuous_update=False
        ),
        R=FloatSlider(
            min=0.0, max=20000.0, step=1000.0, value=2000.0, continuous_update=False
        ),
        xc=FloatSlider(
            min=-10.0, max=10.0, step=1.0, value=0.0, continuous_update=False
        ),
        yc=FloatSlider(
            min=-10.0, max=10.0, step=1.0, value=0.0, continuous_update=False
        ),
        zc=FloatSlider(min=0.0, max=zmax, step=0.5, value=1.0, continuous_update=False),
        dincl=FloatSlider(
            min=-90.0,
            max=90.0,
            step=1.0,
            value=0.0,
            continuous_update=False,
            description="I",
        ),
        ddecl=FloatSlider(
            min=0.0,
            max=180.0,
            step=1.0,
            value=90.0,
            continuous_update=False,
            description="D",
        ),
        f=FloatSlider(
            min=10.0, max=19990.0, step=10.0, value=10000.0, continuous_update=False
        ),
        dx=FloatSlider(
            min=0.25, max=5.0, step=0.25, value=0.25, continuous_update=False
        ),
        showDataPts=Checkbox(value=False),
    )

    return Q
