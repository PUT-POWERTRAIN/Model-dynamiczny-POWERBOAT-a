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
        Q_mat = np.diag([100.0, 100.0, 50.0, 25.0, 0.0, 0.0, 0.0, 0.0])
        R_mat = np.diag([5.0, 100.0])

        # zmienne dla wyznaczenia sciezki
        self.ghost_extension = 20.0 # <--- DODANE (zabezpieczenie przed błędem w splajnie)
        self.last_theta_0 = 0.0 
        self.s_max = 0.0
        self.s_max_extended = 0.0
        self.v_ref = 5.0 # należly sprawdzić lub ocenić jak szybko łódź ma płynąć
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
        self.ocp.cost.W_e = Q_mat

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
        self.ocp.solver_options.integrator_type = 'IRK'
        # ocp.solver_options.print_level = 1
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

        self.ocp_solver = AcadosOcpSolver(self.ocp) # <--- POPRAWKA (brakowało self.)

        self.simX = np.zeros((self.N+1, nx))
        self.simU = np.zeros((self.N, nu))

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

        x0_acados = np.concatenate([x_current[0:6], [current_w, current_alpha]])
        self.ocp_solver.set(0, "lbx", x0_acados)
        self.ocp_solver.set(0, "ubx", x0_acados)

        ## pytanie czy nie warto dodać tutaj sprawdzenia gdzie jesteśmy na tej ścieżce faktycznie, żeby w czasie
        ## się to nie rozjechało bardzo
        self.last_theta_0 = self._find_closest_theta(x_current[0], x_current[1], self.last_theta_0)

        current_theta = self.last_theta_0
        current_v = x_current[3] 

        for idx in range(self.N):
            current_v += self.a_ref * self.DT
            
            dist_to_end = max(0.0, self.s_max_extended - current_theta)
            v_stop = min(self.v_ref, 0.2 * dist_to_end)
            
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

        if status != 0:
            print(f"[Ostrzeżenie] Solver zwrócił status {status}!")

        u_opt = self.ocp_solver.get(0, "u")

        return u_opt[0], u_opt[1]