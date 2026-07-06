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
from boat_model import export_boat_model
import numpy as np
import casadi as ca

class NMPC:
    def __init__(self):
        # ZMIENNE DLA STEROWANIA MPC
        # horyzont
        self.N = 60
        self.DT = 0.2
        # wagi kosztów
        Q_mat = np.diag([11.1111, 11.1111, 14.5903, 5.0000, 0.0, 0.0, 0.0, 0.0])
        R_mat = np.diag([0.1017, 3.6211])
        # Q_ter = np.array([
        #     [   95.0829,    -0.3614,    -3.6983,    62.4492,     1.7827,    -1.8355,     0.8665,     1.3056],
        #     [   -0.3614,   159.7803,   846.0456,    -0.5224,   -95.3072,   239.0358,     0.0156,  -100.7342],
        #     [   -3.6983,   846.0456,  6514.9493,    -5.3837, -1359.5118,  2212.7167,     0.1444, -1078.3125],
        #     [   62.4492,    -0.5224,    -5.3837,    62.8581,     2.7450,    -2.7562,     1.0800,     2.0194],
        #     [    1.7827,   -95.3072, -1359.5118,     2.7450,   985.5476,  -863.0928,    -0.0530,   716.4743],
        #     [   -1.8355,   239.0358,  2212.7167,    -2.7562,  -863.0928,   981.9520,     0.0622,  -645.8128],
        #     [    0.8665,     0.0156,     0.1444,     1.0800,    -0.0530,     0.0622,     0.0251,    -0.0396],
        #     [    1.3056,  -100.7342, -1078.3125,     2.0194,   716.4743,  -645.8128,    -0.0396,   548.6710],
        # ])

        # zmienne dla wyznaczenia sciezki
        self.ghost_extension = 20.0 # <--- DODANE (zabezpieczenie przed błędem w splajnie)
        self.last_theta_0 = 0.0 
        self.s_max = 0.0
        self.s_max_extended = 0.0
        self.v_ref = 2.0 # należly sprawdzić lub ocenić jak szybko łódź ma płynąć
        self.a_ref = 0.5 # należy sprawdzić lub ocenić jak szybko łódź przyśpiesza
        # wartości a_ref i v_ref można przyjąć jako mniejsze niż są w rzeczywistości i łódź będzie płynąć
        # właśnie z taką zadaną prędkością i przyśpieszeniem

        # Puste kontenery (zabezpieczenie, by Python się nie sypnął przed wywołaniem set_path)
        self.spline_x = None
        self.spline_y = None

        # USTAWIENIA SOLVERA ACADOS
        # pusty kontener solvera
        self.ocp = AcadosOcp()
        # ustawienie modelu lodzi
        model = export_boat_model()
        self.ocp.model = model
        # horyzont predykcji 
        self.ocp.solver_options.N_horizon = self.N
        self.ocp.solver_options.tf = self.N * self.DT # <--- POPRAWKA (brakowało self.)
        
        nx = model.x.rows()
        nu = model.u.rows()

        # path cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
        self.ocp.cost.yref = np.zeros((nx+nu,))
        self.ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

        # terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.yref_e = np.zeros((nx,))
        self.ocp.model.cost_y_expr_e = model.x
        self.ocp.cost.W_e = 5*Q_mat

        # ograniczenia
        self.ocp.constraints.lbu = np.array([-5.0, -np.pi/4])
        self.ocp.constraints.ubu = np.array([+24.0, +np.pi/4])
        self.ocp.constraints.idxbu = np.array([0, 1])

        # stan początkowy
        #self.ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

        # set options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        self.ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 1
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        self.ocp.constraints.x0 = np.zeros(nx)
        
        self.ocp_solver = AcadosOcpSolver(self.ocp) # <--- POPRAWKA (brakowało self.)

        self.simX = np.zeros((self.N+1, nx))
        self.simU = np.zeros((self.N, nu))

        self.predicted_path = None
        self.last_sol_X = np.zeros((9, 2))
        
        self.c_lat_hist = []
        self.c_lon_hist = []
        self.c_vtheta_hist = []
        self.c_speed_hist = []
        self.c_dv_hist = []
        self.c_dalpha_hist = []

    def set_path(self, waypoints: np.ndarray) -> float:
        # 1. Przedłużenie ścieżki (ghost point) - niezbędne dla zachowania kształtu
        # splajnu na samym końcu i dla referencji podczas hamowania
        dir_vec = waypoints[-1] - waypoints[-2]
        dir_len = np.linalg.norm(dir_vec)
        dir_unit = dir_vec / dir_len if dir_len > 1e-6 else np.array([1.0, 0.0])
        
        ghost_point = waypoints[-1] + dir_unit * self.ghost_extension
        waypoints_ext = np.vstack([waypoints, ghost_point])

        # 2. Obliczanie fizycznych długości segmentów (Chord-length parameterization)
        diffs = np.diff(waypoints_ext, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        orig_s = np.concatenate([[0.0], np.cumsum(seg_lengths)]) # Skumulowana droga w metrach

        # 3. Zapisanie długości (oryginalnej i przedłużonej)
        self.s_max_original = float(orig_s[-2])
        self.s_max_extended = float(orig_s[-1])
        self.s_max = self.s_max_original

        # 4. Zagęszczenie punktów bazowych dla gładszego B-splajnu
        # Dynamiczna liczba punktów (np. 2-3 punkty na każdy metr ścieżki)
        num_dense_points = max(120, int(self.s_max_extended * 2.5))
        s_dense = np.linspace(0, self.s_max_extended, num_dense_points)
        
        x_dense = np.interp(s_dense, orig_s, waypoints_ext[:, 0])
        y_dense = np.interp(s_dense, orig_s, waypoints_ext[:, 1])

        # 5. Budowa silnika splajnów CasADi (oś X: dystans w metrach, oś Y: współrzędne)
        self.spline_x = ca.interpolant('x_spline', 'bspline', [s_dense], x_dense)
        self.spline_y = ca.interpolant('y_spline', 'bspline', [s_dense], y_dense)

        # 6. Wyzerowanie postępu (niezbędne, gdy podajemy nową ścieżkę z A* w locie!)
        self.last_theta_0 = 0.0  

        return self.s_max # <--- DODANE (brakowało returna na końcu)

    def get_next_step(self, current_s: float, distance: float):
        if self.spline_x is None:
            raise RuntimeError("Ścieżka nie została zainicjalizowana! Wywołaj set_path().")

        # Gdzie chcemy być na ścieżce?
        target_s = current_s + distance
        
        # ZABEZPIECZENIE: Jeśli każemy łodzi szukać punktu poza ścieżką (np. wyprzedzamy metę),
        # ucinamy postęp do przedłużonej mety (ghost point), żeby CasADi nie wyrzuciło błędu
        # out-of-bounds.
        safe_s = np.clip(target_s, 0.0, self.s_max_extended)
        
        # Pobieramy współrzędne (float konwertuje z formatu CasADi na zwykłą liczbę)
        x_target = float(self.spline_x(safe_s))
        y_target = float(self.spline_y(safe_s))
        
        return x_target, y_target, target_s

    def _get_heading_ref(self, s):
        """Oblicza kąt referencyjny psi ze stycznej splajnu w punkcie s."""
        eps = 0.5  # [m] krok do różnicy skończonej
        s_lo = np.clip(s, 0.0, self.s_max_extended)
        s_hi = np.clip(s + eps, 0.0, self.s_max_extended)
        if abs(s_hi - s_lo) < 1e-9:
            # Na samym końcu — cofnij się zamiast iść do przodu
            s_lo = np.clip(s - eps, 0.0, self.s_max_extended)
        dx = float(self.spline_x(s_hi)) - float(self.spline_x(s_lo))
        dy = float(self.spline_y(s_hi)) - float(self.spline_y(s_lo))
        return float(np.arctan2(dy, dx))
    
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
        #sprawdzenie czy sciezka istnieje
        if self.spline_x is None:
            raise RuntimeError("Najpierw wywołaj set_path(waypoints)!")
        
        x0_acados = np.concatenate([np.array(x_current).flatten(), [current_w, current_alpha]])
        self.ocp_solver.set(0, "lbx", x0_acados)
        self.ocp_solver.set(0, "ubx", x0_acados)

        ## pytanie czy nie warto dodać tutaj sprawdzenia gdzie jesteśmy na tej ścieżce faktycznie, żeby w czasie
        ## się to nie rozjechało bardzo
        self.last_theta_0 = self._find_closest_theta(x0_acados[0], x0_acados[1], self.last_theta_0)

        current_theta = self.last_theta_0
        current_v = x0_acados[3] 

        for idx in range(self.N):
            current_v += self.a_ref * self.DT
            
            dist_to_end = max(0.0, self.s_max_original - current_theta)
            # Zamiast liniowego 0.1, użyj pierwiastka (proporcjonalnego do energii kinetycznej)
            v_stop = min(self.v_ref, 0.5 * np.sqrt(max(0.0, dist_to_end)))
            
            next_v = min(current_v, v_stop)
            current_v = next_v 
            distance = next_v * self.DT
            x_target, y_target, current_theta = self.get_next_step(current_theta, distance)
            psi_ref = self._get_heading_ref(current_theta)

            # path cost: etapy 0..N-1, wymiar nx+nu
            yref_k = np.array([x_target, y_target, psi_ref, next_v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.ocp_solver.set(idx, "yref", yref_k)

        # terminal cost: etap N, wymiar nx
        yref_e = np.array([x_target, y_target, psi_ref, next_v, 0.0, 0.0, 0.0, 0.0])
        self.ocp_solver.set(self.N, "yref", yref_e)
        
        status = self.ocp_solver.solve()

        solve_time = self.ocp_solver.get_stats("time_tot") # Zwraca czas w sekundach
        print(f"Czas obliczeń OCP: {solve_time * 1000:.2f} ms")

        if status != 0:
            print(f"[Ostrzeżenie] Solver zwrócił status {status}!")

        u_opt = self.ocp_solver.get(0, "u")

        # Zapisz prognozę drogi dla symulatora
        predicted = np.zeros((self.N, 2))
        for k in range(self.N):
            xk = self.ocp_solver.get(k, "x")
            predicted[k, :] = xk[0:2]
        self.predicted_path = predicted

        # Zapisz wirtualny postęp dla kompatybilności
        self.last_sol_X = np.zeros((9, 2))
        self.last_sol_X[8, 1] = self.last_theta_0

        # Wylicz koszty pod wykresy
        path_x = float(self.spline_x(self.last_theta_0))
        path_y = float(self.spline_y(self.last_theta_0))
        err_x = path_x - x0_acados[0]
        err_y = path_y - x0_acados[1]
        psi = x0_acados[2]
        
        ego_x = np.cos(psi) * err_x + np.sin(psi) * err_y
        ego_y = -np.sin(psi) * err_x + np.cos(psi) * err_y

        self.c_lat_hist.append(float(100.0 * ego_y**2))
        self.c_lon_hist.append(float(100.0 * ego_x**2))
        self.c_vtheta_hist.append(0.0)
        self.c_speed_hist.append(float(25.0 * (x0_acados[3] - 5.0)**2))
        self.c_dv_hist.append(0.0)
        self.c_dalpha_hist.append(0.0)

        # alpha cmd musi być w stopniach dla casadi_path_following
        return u_opt[0], np.degrees(u_opt[1])