import casadi as ca
import numpy as np
import casadi_model as boat_model

class PathFollowingMPC:
    def __init__(self):
        self.Np = 25  
        self.dt = boat_model.DT  

        # --- WAGI DO TUNINGU (Ego-Frame wg MDPI Sensors) ---
        self.Q_lat     = 1500.0  # Bardzo silny nacisk na trzymanie się ścieżki
        self.Q_lon     = 20.0    
        self.Q_vtheta  = 10.0    
        self.v_ref     = 5.25    
        self.R_dV      = 5.0     
        self.R_dAlpha  = 50.0    
        self.Q_speed   = 5.0     # Zmniejszone, by łódź wolała skręcić (użyć gazu) niż na siłę hamować
        self.Q_heading = 800.0   # Silny nacisk na kierunek dziobu
        self.Q_sway    = 50.0    

        # Ghost Point — przedłużenie ścieżki
        self.ghost_extension = 20.0  # [m]

        self.path_s = None
        self.spline_x = None
        self.spline_y = None
        self.s_max = 0.0
        self.s_max_original = 0.0
        self.s_max_extended = 0.0

        self.last_sol_U = None
        self.last_sol_X = None
        self.last_theta_0 = 0.0

    def set_path(self, waypoints: np.ndarray):
        # Ghost point — przedłuż ścieżkę wzdłuż ostatniego kierunku
        dir_vec = waypoints[-1] - waypoints[-2]
        dir_len = np.linalg.norm(dir_vec)
        if dir_len > 1e-6:
            dir_unit = dir_vec / dir_len
        else:
            dir_unit = np.array([1.0, 0.0])
        
        ghost_point = waypoints[-1] + dir_unit * self.ghost_extension
        waypoints_ext = np.vstack([waypoints, ghost_point])

        diffs = np.diff(waypoints_ext, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        orig_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        self.s_max_original = float(orig_s[-2])  # prawdziwa meta
        self.s_max_extended = float(orig_s[-1])   # z ghostem
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

        self.opti.subject_to(self.opti.bounded(0, self.U[0, :], 24))             
        self.opti.subject_to(self.opti.bounded(-np.pi/4, self.U[1, :], np.pi/4)) 
        self.opti.subject_to(self.opti.bounded(0.0, self.U[2, :], 6.0))          

        cost = 0
        cost_lat     = 0
        cost_lon     = 0
        cost_vtheta  = 0
        cost_speed   = 0
        cost_heading = 0
        cost_dV      = 0
        cost_dAlpha  = 0
        cost_sway    = 0

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
            x_next   = boat_model.boat_step(x_k, T_k, alp_k)
            theta_next = theta_k + self.dt * v_theta_k

            self.opti.subject_to(self.X[0:6, k+1] == x_next)
            self.opti.subject_to(self.X[6, k+1]   == w_next)
            self.opti.subject_to(self.X[7, k+1]   == alp_next)
            self.opti.subject_to(self.X[8, k+1]   == theta_next)

            # Theta może sięgać do przedłużonej ścieżki (ghost)
            self.opti.subject_to(theta_next <= self.s_max_extended)

            # === HAMOWANIE ===
            # Odg do PRAWDZIWEGO końca (nie ghost!)
            dist_to_end = ca.fmax(self.s_max_original - theta_k, 0.0)

            # Prędkości docelowe
            v_target_virtual = ca.fmin(self.v_ref, ca.fmax(0.0, 0.28 * dist_to_end))
            v_target_physical = ca.fmin(self.v_ref, 0.28 * dist_to_end)

            # Path following — theta sięga do ghost extension
            safe_theta = ca.fmin(ca.fmax(theta_k, 0.0), self.s_max_extended)
            path_x = self.spline_x(safe_theta)
            path_y = self.spline_y(safe_theta)

            err_x = path_x - self.X[0, k]
            err_y = path_y - self.X[1, k]
            psi_k = self.X[2, k]

            # Transformacja błędu to układu Ego (zgodnie z MDPI)
            ego_x =  ca.cos(psi_k) * err_x + ca.sin(psi_k) * err_y
            ego_y = -ca.sin(psi_k) * err_x + ca.cos(psi_k) * err_y

            c_lat  = self.Q_lat * ego_y**2
            c_lon  = self.Q_lon * ego_x**2
            c_vth  = self.Q_vtheta * (v_theta_k - v_target_virtual)**2
            c_spd  = self.Q_speed * (u_k - v_target_physical)**2

            cost += c_lat + c_lon + c_vth + c_spd
            cost_lat     += c_lat
            cost_lon     += c_lon
            cost_vtheta  += c_vth
            cost_speed   += c_spd

            # Anti-drift: kara za prędkość boczną (sway)
            v_sway = self.X[4, k]
            c_sway = self.Q_sway * v_sway**2
            cost += c_sway
            cost_sway += c_sway

            # Heading tracking — kąt łodzi vs styczna ścieżki (cross/dot, gładkie)
            ds = 0.5
            theta_fwd = ca.fmin(safe_theta + ds, self.s_max_extended)
            tx = self.spline_x(theta_fwd) - path_x  # wektor stycznej
            ty = self.spline_y(theta_fwd) - path_y
            t_norm = ca.sqrt(tx**2 + ty**2 + 1e-6)  # normalizacja
            tx_n = tx / t_norm
            ty_n = ty / t_norm

            psi_k = self.X[2, k]
            # cross = sin(psi - psi_path), dot = cos(psi - psi_path)
            cross = ca.cos(psi_k) * ty_n - ca.sin(psi_k) * tx_n
            dot   = ca.cos(psi_k) * tx_n + ca.sin(psi_k) * ty_n
            c_head = self.Q_heading * (cross**2 + (1 - dot)**2)
            cost += c_head
            cost_heading += c_head

            if k > 0:
                dV     = self.U[0, k] - self.U[0, k-1]
                dAlpha = self.U[1, k] - self.U[1, k-1]
                c_dv  = self.R_dV * dV**2
                c_da  = self.R_dAlpha * dAlpha**2
                cost      += c_dv + c_da
                cost_dV    += c_dv
                cost_dAlpha += c_da

        # Zapamiętaj wyrażenia symboliczne do ewaluacji po solve
        self.cost_lat_expr     = cost_lat
        self.cost_lon_expr     = cost_lon
        self.cost_vtheta_expr  = cost_vtheta
        self.cost_speed_expr   = cost_speed
        self.cost_heading_expr = cost_heading
        self.cost_dV_expr      = cost_dV
        self.cost_dAlpha_expr  = cost_dAlpha
        self.cost_sway_expr    = cost_sway

        self.opti.minimize(cost)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 200,
            "print_level": 0,
            "sb": "yes",
            "acceptable_tol": 1e-2,
            "warm_start_init_point": "yes",
        }
        self.opti.solver('ipopt', p_opts, s_opts)

    def _find_closest_theta(self, boat_x, boat_y, current_s):
        lookbehind = 10.0
        lookahead = 20.0
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

            # --- DEBUG: wyświetl składniki kosztów ---
            c1_lat = float(sol.value(self.cost_lat_expr))
            c1_lon = float(sol.value(self.cost_lon_expr))
            c2 = float(sol.value(self.cost_vtheta_expr))
            c3 = float(sol.value(self.cost_speed_expr))
            c4 = float(sol.value(self.cost_heading_expr))
            c5 = float(sol.value(self.cost_dV_expr))
            c6 = float(sol.value(self.cost_dAlpha_expr))
            c7 = float(sol.value(self.cost_sway_expr))
            total = c1_lat + c1_lon + c2 + c3 + c4 + c5 + c6 + c7
            theta_now = float(X_sol[8, 0])
            dist_end = max(self.s_max_original - theta_now, 0.0)
            print(f"[COST] dist={dist_end:5.1f}m | "
                  f"lat={c1_lat:7.1f} lon={c1_lon:7.1f} vth={c2:6.1f} spd={c3:5.1f} "
                  f"hdg={c4:6.1f} sway={c7:5.1f} dV={c5:4.1f} dA={c6:5.1f} | "
                  f"TOT={total:7.1f}")
            
        except Exception as e:
            full_msg = str(e)
            status = 'unknown'
            for line in full_msg.split('\n'):
                if any(kw in line for kw in ['Infeasible', 'Maximum', 'Restoration', 'Invalid']):
                    status = line.strip()
                    break
            print(f"[NMPC] Solver fail: {status}")
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