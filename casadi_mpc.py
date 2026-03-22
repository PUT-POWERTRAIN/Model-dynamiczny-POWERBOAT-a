import casadi as ca
import numpy as np
import casadi_model as boat_model

class PathFollowingMPC:
    def __init__(self):
        self.Np = 25  
        self.dt = boat_model.DT  

        # --- WAGI DO TUNINGU ---
        self.Q_contour = 250.0   
        self.Q_vtheta  = 10.0    
        self.v_ref     = 5.25    # PODNIESIONE: max prędkość łodzi (~5.25 m/s przy 24V)
        self.R_dV      = 5.0     
        self.R_dAlpha  = 50.0    
        
        # NOWA WAGA: zmusza łódź fizyczną do posłuszeństwa (przydaje się do twardego hamowania)
        self.Q_speed   = 20.0    

        self.path_s = None
        self.spline_x = None
        self.spline_y = None
        self.s_max = 0.0

        self.last_sol_U = None
        self.last_sol_X = None
        self.last_theta_0 = 0.0

    def set_path(self, waypoints: np.ndarray):
        diffs = np.diff(waypoints, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        orig_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        s_dense = np.linspace(0, orig_s[-1], 100)
        x_dense = np.interp(s_dense, orig_s, waypoints[:, 0])
        y_dense = np.interp(s_dense, orig_s, waypoints[:, 1])

        self.path_s = s_dense
        self.s_max = self.path_s[-1]

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

        self.opti.subject_to(self.opti.bounded(0, self.U[0, :], 24))             
        self.opti.subject_to(self.opti.bounded(-np.pi/4, self.U[1, :], np.pi/4)) 
        self.opti.subject_to(self.opti.bounded(0.0, self.U[2, :], 6.0))          

        cost = 0

        for k in range(self.Np):
            x_k     = self.X[0:6, k]
            u_k     = self.X[3, k]  # Fizyczna prędkość wzdłużna łodzi
            w_k     = self.X[6, k]
            alp_k   = self.X[7, k]
            theta_k = self.X[8, k]

            V_k       = self.U[0, k]
            acmd_k    = self.U[1, k]
            v_theta_k = self.U[2, k]

            alp_next = boat_model.servo_step(alp_k, acmd_k)
            w_next   = boat_model.motor_step(w_k, V_k)
            T_k      = boat_model.c_T * w_k * ca.fabs(w_k) 
            x_next   = boat_model.boat_step(x_k, T_k, alp_k)
            
            theta_next = theta_k + self.dt * v_theta_k

            self.opti.subject_to(self.X[0:6, k+1] == x_next)
            self.opti.subject_to(self.X[6, k+1]   == w_next)
            self.opti.subject_to(self.X[7, k+1]   == alp_next)
            self.opti.subject_to(self.X[8, k+1]   == theta_next)

            self.opti.subject_to(theta_next <= self.s_max + 5.0)

            # === INTELIGENTNE HAMOWANIE ===
            # 1. Odległość do końca trasy
            dist_to_end = ca.fmax(self.s_max - theta_k, 0.0)

            # 2. Prędkości docelowe: 
            # Wirtualny cel zwalnia, ale nie do zera (żeby minął metę) - np. min 0.2 m/s
            v_target_virtual = ca.fmin(self.v_ref, ca.fmax(0.2, 0.4 * dist_to_end))
            
            # FIZYCZNA łódź ma docelowo wyhamować do zera (w punkt!)
            v_target_physical = ca.fmin(self.v_ref, 0.4 * dist_to_end)

            # 3. Path following bazujące na naturalnej dynamice theta_k
            safe_theta = ca.fmin(ca.fmax(theta_k, 0.0), self.s_max)
            path_x = self.spline_x(safe_theta)
            path_y = self.spline_y(safe_theta)

            err_x = self.X[0, k] - path_x
            err_y = self.X[1, k] - path_y

            # 4. Kary
            cost += self.Q_contour * (err_x**2 + err_y**2)
            cost += self.Q_vtheta * (v_theta_k - v_target_virtual)**2 
            
            # Wymuszamy, by łódź hamowała tak jak wskazuje funkcja (0.4 * dist)
            cost += self.Q_speed * (u_k - v_target_physical)**2

            if k > 0:
                dV     = self.U[0, k] - self.U[0, k-1]
                dAlpha = self.U[1, k] - self.U[1, k-1]
                cost  += self.R_dV * dV**2 + self.R_dAlpha * dAlpha**2

        self.opti.minimize(cost)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 100, "print_level": 0, "sb": "yes", "acceptable_tol": 1e-2}
        self.opti.solver('ipopt', p_opts, s_opts)

    def _find_closest_theta(self, boat_x, boat_y, current_s):
        lookbehind = 10.0
        lookahead = 20.0
        s_min = max(0.0, current_s - lookbehind)
        s_max = min(self.s_max, current_s + lookahead)

        s_search = np.linspace(s_min, s_max, 100)
        path_x = self.spline_x(s_search).full().flatten()
        path_y = self.spline_y(s_search).full().flatten()
        dists = np.hypot(path_x - boat_x, path_y - boat_y)
        return float(s_search[np.argmin(dists)])

    def compute_control(self, x_current, current_alpha, current_w):
        if self.spline_x is None:
            raise RuntimeError("Najpierw wywołaj set_path(waypoints)!")

        boat_x, boat_y = x_current[0], x_current[1]
        theta_guess = 0.0 if self.last_sol_X is None else self.last_sol_X[8, 1]

        theta_0 = self._find_closest_theta(boat_x, boat_y, theta_guess)

        if self.last_theta_0 - theta_0 > 5.0:
            pass
        else:
            theta_0 = max(theta_0, self.last_theta_0)
        self.last_theta_0 = theta_0

        x0_full = np.concatenate([x_current, [current_w, current_alpha, theta_0]])
        self.opti.set_value(self.x0_param, x0_full)

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
            
        except Exception as e:
            print("[NMPC] Solver złapał zadyszkę na zakręcie! Przesuwam stare sterowanie.")
            if self.last_sol_U is not None:
                U_sol = np.roll(self.last_sol_U, -1, axis=1)
                U_sol[:, -1] = U_sol[:, -2]  
                X_sol = np.roll(self.last_sol_X, -1, axis=1)
                X_sol[:, -1] = X_sol[:, -2]
                
                self.last_sol_U = U_sol
                self.last_sol_X = X_sol
            else:
                U_sol = np.zeros((self.nu, self.Np))
                X_sol = np.zeros((self.nx, self.Np+1))

        self.predicted_path = X_sol[0:2, :].T

        V_cmd = float(U_sol[0, 0])
        alpha_cmd_deg = float(np.degrees(U_sol[1, 0]))

        return V_cmd, alpha_cmd_deg