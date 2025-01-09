import discretize
import numpy as np
from simpeg.electromagnetics import time_domain as tdem
from simpeg.utils.solver_utils import get_default_solver
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm


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

