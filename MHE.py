import os
import ctypes

os.environ["ACADOS_SOURCE_DIR"] = "/home/jacek/Documents/acados"

# Pre-load shared libraries to avoid libqpOASES_e.so loading issues
acados_lib_dir = os.path.join(os.environ["ACADOS_SOURCE_DIR"], "lib")
if os.path.exists(acados_lib_dir):
    try:
        ctypes.CDLL(os.path.join(acados_lib_dir, "libblasfeo.so"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libhpipm.so"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libqpOASES_e.so"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(os.path.join(acados_lib_dir, "libacados.so"), mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning: Could not pre-load acados shared libraries: {e}")

from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
from mhe_boat_model import export_mhe_boat_model
from export_mhe_solver import export_mhe_solver
import numpy as np
import casadi as ca

class MHE:
    def __init__(self, dt = 0.2, n = 60, max_alpha = np.pi/4, max_u = [-5, 24]):
        self.DT = dt
        self.N = n
        
        self.model_mhe = export_mhe_ode_model()

        nx = model_mhe.x.rows()
        nw = model_mhe.u.rows()
        ny = nx

        self.Q0_mhe = 100*np.eye((nx))
        self.Q_mhe  = 0.1*np.eye(nw)
        self.R_mhe  = 0.1*np.eye(ny)
        
        self.x_ref = np.zeros([N,nx])
        self.u_ref = np.zeros([N,nw])

        self.acados_solver_mhe = export_mhe_solver(self.model_mhe, self.N, self.DT, self.Q_mhe, self.Q0_mhe, self.R_mhe)

        self.x_hat = np.zeros([1, nx])
        
    def set_measurements(self, measure_x, cmd_u):
        self.x_ref[:-1] = self.x_ref[1:]
        self.x_ref[-1] = measure_x
        
        self.u_ref[:-1] = self.u_ref[1:]
        self.u_ref[-1] = cmd_u

    def predict_x(self):
        for idx in range(self.N + 1):
            self.acados_solver_mhe.set(idx, "yref", self.x_ref[idx])
            if idx < self.N:
                self.acados_solver_mhe.set(idx, "p", self.u_ref[idx])
        
        status = acados_solver_mhe.solve()

        if status != 0 and status != 2:
            raise Exception(f'acados returned status {status}.')

        self.x_hat = acados_solver_mhe.get(self.N, "x")
        return self.x_hat