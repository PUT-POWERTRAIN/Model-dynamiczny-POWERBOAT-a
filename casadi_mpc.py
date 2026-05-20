import casadi as ca
import numpy as np
import casadi_model as boat_model
from EKF import EKF

class PathFollowingMPC:
    def __init__(self, use_ekf=False):
        self.Np = 75  
        self.dt = boat_model.DT  
        self.use_ekf = use_ekf
        self.ekf = EKF()

        self.Q_lat     = 100.0  
        self.Q_lon     = 100.0    
        self.Q_vtheta  = 100.0    
        self.v_ref     = 2.00    
        self.R_dV      = 5.0     
        self.R_dAlpha  = 100.0   
        self.Q_speed   = 25.0     
        
        self.ghost_extension = 20.0
        self.path_s = None
        self.spline_x = None
        self.spline_y = None
        self.s_max = 0.0
        self.s_max_original = 0.0
        self.s_max_extended = 0.0

        self.last_sol_U = None
        self.last_sol_X = None
        self.last_theta_0 = 0.0

        self.c_lat_hist = []
        self.c_lon_hist = []
        self.c_vtheta_hist = []
        self.c_speed_hist = []
        self.c_dv_hist = []
        self.c_dalpha_hist = []

    def set_path(self, waypoints: np.ndarray):
        dir_vec = waypoints[-1] - waypoints[-2]
        dir_len = np.linalg.norm(dir_vec)
        dir_unit = dir_vec / dir_len if dir_len > 1e-6 else np.array([1.0, 0.0])
        
        ghost_point = waypoints[-1] + dir_unit * self.ghost_extension
        waypoints_ext = np.vstack([waypoints, ghost_point])

        diffs = np.diff(waypoints_ext, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        orig_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        self.s_max_original = float(orig_s[-2])
        self.s_max_extended = float(orig_s[-1])
        self.s_max = self.s_max_original

        s_dense = np.linspace(0, orig_s[-1], 120)
        x_dense = np.interp(s_dense, orig_s, waypoints_ext[:, 0])
        y_dense = np.interp(s_dense, orig_s, waypoints_ext[:, 1])

        self.path_s = s_dense
        self.spline_x = ca.interpolant('x_spline', 'bspline', [self.path_s], x_dense)
        self.spline_y = ca.interpolant('y_spline', 'bspline', [self.path_s], y_dense)

        self._build_opti()
        self.last_theta_0 = 0.0  
        self.last_sol_U = None
        self.last_sol_X = None

    def _build_opti(self):
        self.opti = ca.Opti()

        self.nx = 9
        self.X = self.opti.variable(self.nx, self.Np + 1)
        self.nu = 3
        self.U = self.opti.variable(self.nu, self.Np)

        self.x0_param = self.opti.parameter(self.nx)
        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        self.d_param = self.opti.parameter(3)

        self.opti.subject_to(self.opti.bounded(-5, self.U[0, :], 24))             
        self.opti.subject_to(self.opti.bounded(-np.pi/4, self.U[1, :], np.pi/4)) 
        self.opti.subject_to(self.opti.bounded(0.01, self.U[2, :], 6.0))          

        cost = 0
        cost_lat = cost_lon = cost_vtheta = cost_speed = cost_dV = cost_dAlpha = 0

        for k in range(self.Np):
            x_k     = self.X[0:6, k]
            u_k     = self.X[3, k]
            w_k     = self.X[6, k]
            alp_k   = self.X[7, k]
            theta_k = self.X[8, k]

            V_k       = self.U[0, k]
            acmd_k    = self.U[1, k]
            v_theta_k = self.U[2, k]

            alp_next = boat_model.servo_step(alp_k, acmd_k)
            w_next   = boat_model.motor_step(w_k, V_k)
            T_k      = boat_model.c_T * w_k * ca.fabs(w_k) 
            
            x_next = boat_model.boat_step(x_k, T_k, alp_k)
            
            # Dodanie biasu do prędkości w predykcji
            x_next = ca.vertcat(
                x_next[0:3], 
                x_next[3:6] + self.d_param
            )
            
            theta_next = theta_k + self.dt * v_theta_k

            self.opti.subject_to(self.X[0:6, k+1] == x_next)
            self.opti.subject_to(self.X[6, k+1]   == w_next)
            self.opti.subject_to(self.X[7, k+1]   == alp_next)
            self.opti.subject_to(self.X[8, k+1]   == theta_next)
            self.opti.subject_to(theta_next <= self.s_max_extended)

            dist_to_end = ca.fmax(self.s_max_original - theta_k, 0.0)
            v_target_virtual = ca.fmin(self.v_ref, ca.fmax(0.0, 0.1 * dist_to_end))
            v_target_physical = ca.fmin(self.v_ref, 0.2 * dist_to_end)

            safe_theta = ca.fmin(ca.fmax(theta_k, 0.0), self.s_max_extended)
            path_x = self.spline_x(safe_theta)
            path_y = self.spline_y(safe_theta)

            err_x = path_x - self.X[0, k]
            err_y = path_y - self.X[1, k]
            
            psi_k = self.X[2, k]
            ego_x =  ca.cos(psi_k) * err_x + ca.sin(psi_k) * err_y
            ego_y = -ca.sin(psi_k) * err_x + ca.cos(psi_k) * err_y

            c_lat  = self.Q_lat * ego_y**2
            c_lon  = self.Q_lon * ego_x**2
            c_vth  = self.Q_vtheta * (v_theta_k - v_target_virtual)**2
            c_spd  = self.Q_speed * (u_k - v_target_physical)**2

            cost += c_lat + c_lon + c_vth + c_spd
            cost_lat += c_lat; cost_lon += c_lon
            cost_vtheta += c_vth; cost_speed += c_spd

            if k > 0:
                dV     = self.U[0, k] - self.U[0, k-1]
                dAlpha = self.U[1, k] - self.U[1, k-1]
                c_dv  = self.R_dV * dV**2
                c_da  = self.R_dAlpha * dAlpha**2
                cost += c_dv + c_da
                cost_dV += c_dv; cost_dAlpha += c_da

        self.cost_lat_expr = cost_lat; self.cost_lon_expr = cost_lon
        self.cost_vtheta_expr = cost_vtheta; self.cost_speed_expr = cost_speed
        self.cost_dV_expr = cost_dV; self.cost_dAlpha_expr = cost_dAlpha

        self.opti.minimize(cost)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 200, "print_level": 0, "sb": "yes", 
                  "acceptable_tol": 1e-2, "warm_start_init_point": "yes"}
        self.opti.solver('ipopt', p_opts, s_opts)

    def _find_closest_theta(self, boat_x, boat_y, current_s):
        lookbehind, lookahead = 10.0, 20.0
        s_min = max(0.0, current_s - lookbehind)
        s_max = min(self.s_max_extended, current_s + lookahead)
        s_search = np.linspace(s_min, s_max, 100)
        path_x = self.spline_x(s_search).full().flatten()
        path_y = self.spline_y(s_search).full().flatten()
        dists = np.hypot(path_x - boat_x, path_y - boat_y)
        return float(s_search[np.argmin(dists)])

    def compute_control(self, x_current, current_alpha, current_w):
        if self.spline_x is None:
            raise RuntimeError("Najpierw wywołaj set_path(waypoints)!")

        if self.use_ekf:
            y_meas = np.array([x_current[0], x_current[1], x_current[2], x_current[5]]).reshape(4, 1)
            
            if not self.ekf._initialized: 
                self.ekf.x_hat = np.array(x_current).reshape(6, 1)
                self.ekf.w_hat = current_w
                self.ekf.alpha_hat = current_alpha
                self.ekf._initialized = True
                
            self.ekf.update(y_meas)
            
            # 1. Pobranie PRAWDZIWEGO stanu i biasu z EKF
            x_true = self.ekf.x_hat.flatten()
            d_est = self.ekf.d.flatten()
        else:
            x_true = np.array(x_current)
            d_est = np.zeros(3)

        # 2. KLUCZOWE: Obliczenie stanu NOMINALNEGO dla MPC
        # Odejmujemy bias od prędkości, żeby MPC nie naliczył go podwójnie!
        x_nom = x_true.copy()
        x_nom[3:6] -= d_est 

        # Szukamy thety na podstawie fizycznej pozycji łodzi
        boat_x, boat_y = x_true[0], x_true[1]
        theta_guess = 0.0 if self.last_sol_X is None else self.last_sol_X[8, 1]

        theta_0 = self._find_closest_theta(boat_x, boat_y, theta_guess)
        if not (self.last_theta_0 - theta_0 > 5.0):
            theta_0 = max(theta_0, self.last_theta_0)
        self.last_theta_0 = theta_0

        # Inicjalizacja MPC stanem nominalnym
        x0_full = np.concatenate([x_nom, [current_w, current_alpha, theta_0]])
        self.opti.set_value(self.x0_param, x0_full)
        self.opti.set_value(self.d_param, d_est)

        if self.last_sol_U is not None:
            U_init = np.roll(self.last_sol_U, -1, axis=1)
            U_init[:, -1] = U_init[:, -2]
            self.opti.set_initial(self.U, U_init)
            X_init = np.roll(self.last_sol_X, -1, axis=1)
            X_init[:, -1] = X_init[:, -2]
            self.opti.set_initial(self.X, X_init)

        try:
            sol = self.opti.solve()
            U_sol = sol.value(self.U)
            X_sol = sol.value(self.X)
            self.last_sol_U = U_sol
            self.last_sol_X = X_sol
            
            # Zapisz numeryczne wartości kosztów dla tego kroku
            self.c_lat_hist.append(float(sol.value(self.cost_lat_expr)))
            self.c_lon_hist.append(float(sol.value(self.cost_lon_expr)))
            self.c_vtheta_hist.append(float(sol.value(self.cost_vtheta_expr)))
            self.c_speed_hist.append(float(sol.value(self.cost_speed_expr)))
            self.c_dv_hist.append(float(sol.value(self.cost_dV_expr)))
            self.c_dalpha_hist.append(float(sol.value(self.cost_dAlpha_expr)))
            
        except Exception as e:
            if self.last_sol_U is not None:
                U_sol = np.roll(self.last_sol_U, -1, axis=1); U_sol[:, -1] = U_sol[:, -2]  
                X_sol = np.roll(self.last_sol_X, -1, axis=1); X_sol[:, -1] = X_sol[:, -2]
                self.last_sol_U = U_sol; self.last_sol_X = X_sol
            else:
                U_sol = np.zeros((self.nu, self.Np)); X_sol = np.zeros((self.nx, self.Np+1))

        self.predicted_path = X_sol[0:2, :].T
        V_cmd = float(U_sol[0, 0])
        alpha_cmd_deg = float(np.degrees(U_sol[1, 0]))

        if self.use_ekf:
            self.ekf.predict([V_cmd, float(U_sol[1, 0])])
        return V_cmd, alpha_cmd_deg