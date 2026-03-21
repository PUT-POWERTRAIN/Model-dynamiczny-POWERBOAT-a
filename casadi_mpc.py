import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import casadi_model as boat_model


class SimpleMPC:
    def __init__(self):
        self.Np = 14
        self.Nc = 3
        self.dt = 0.928
        self.t  = 0.0
        self.target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.Q_pos    = 10.0
        self.Q_psi    = 1.0
        self.R_dV     = 5.0
        self.R_dAlpha = 30.25 * self.Np

        self.bounds = [(0, 24), (-45, 45)] * self.Nc
        self.u_prev         = None
        self.predicted_path = None

        # ── PATH FOLLOWING ──────────────────────────────────────────────
        self.path_points   = None
        self.path_spline_x = None
        self.path_spline_y = None
        self.path_s        = None
        self.s_current     = 0.0
        self.goal_tolerance = 1.0

        # ── ADAPTACYJNY LOOKAHEAD ───────────────────────────────────────
        self.lookahead_min = 4.0    # [m] — ostry zakręt
        self.lookahead_max = 20.0   # [m] — prosta
        self.lookahead     = self.lookahead_max  # aktualny lookahead
        # Kąt przy wierzchołku środkowego punktu trójkąta:
        # ~180° = prosta → lookahead_max
        # ~90°  = zakręt → lookahead_max * 0.5
        # <60°  = ostry  → lookahead_min
        self._lookahead_angle = np.pi  # [rad] — zapamiętany dla UI

    # ── ŚCIEŻKA ─────────────────────────────────────────────────────────

    def set_path(self, waypoints: np.ndarray):
        self.path_points = waypoints
        diffs = np.diff(waypoints, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.path_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.path_spline_x = CubicSpline(self.path_s, waypoints[:, 0])
        self.path_spline_y = CubicSpline(self.path_s, waypoints[:, 1])
        self.s_current = 0.0
        self.lookahead = self.lookahead_max

    def is_path_complete(self):
        if self.path_s is None:
            return False
        return (self.path_s[-1] - self.s_current) < self.goal_tolerance

    # ── ADAPTACYJNY LOOKAHEAD ───────────────────────────────────────────

    def _compute_adaptive_lookahead(self):
        """
        Trzy punkty na ścieżce tworzą trójkąt:
          A = s_current + lookahead_min        (blisko)
          B = s_current + (min+max)/2          (środek)
          C = s_current + lookahead_max        (daleko)

        Kąt w wierzchołku B (środkowym):
          ~π (180°) = prosta    → lookahead_max
          ~π/2 (90°) = zakręt   → interpolacja
          <π/3 (60°) = ostry    → lookahead_min

        Lookahead interpolowany liniowo między min a max
        proporcjonalnie do kąta: lookahead = min + (max-min)*(angle/π)
        """
        s_max = self.path_s[-1]

        sA = min(self.s_current + self.lookahead_min,           s_max)
        sB = min(self.s_current + (self.lookahead_min +
                                   self.lookahead_max) / 2.0,  s_max)
        sC = min(self.s_current + self.lookahead_max,           s_max)

        A = np.array([self.path_spline_x(sA), self.path_spline_y(sA)])
        B = np.array([self.path_spline_x(sB), self.path_spline_y(sB)])
        C = np.array([self.path_spline_x(sC), self.path_spline_y(sC)])

        # Wektory BA i BC (od środkowego punktu do pozostałych)
        BA = A - B
        BC = C - B

        len_BA = np.linalg.norm(BA)
        len_BC = np.linalg.norm(BC)

        if len_BA < 1e-6 or len_BC < 1e-6:
            # Jesteśmy blisko końca — użyj min
            angle = self.lookahead_min / self.lookahead_max * np.pi
        else:
            cos_angle = np.dot(BA, BC) / (len_BA * len_BC)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)  # 0..π

        self._lookahead_angle = angle
        self._triangle_points = (A, B, C)  # do rysowania w UI

        # Interpolacja liniowa: angle=π → max, angle=0 → min
        t = angle / np.pi
        lookahead = self.lookahead_min + (self.lookahead_max -
                                          self.lookahead_min) * t
        return float(np.clip(lookahead, self.lookahead_min,
                              self.lookahead_max))

    # ── WIRTUALNY CEL ────────────────────────────────────────────────────

    def _update_virtual_target(self, x_current):
        if self.path_spline_x is None:
            raise RuntimeError("Wywołaj set_path() najpierw.")

        boat_x, boat_y = float(x_current[0]), float(x_current[1])
        s_max = self.path_s[-1]

        # Szukaj rzutu — tylko do przodu (zapobiega cofaniu)
        s_search = np.linspace(
            self.s_current,
            min(self.s_current + 30.0, s_max),
            300
        )
        path_x = self.path_spline_x(s_search)
        path_y = self.path_spline_y(s_search)
        dists  = np.hypot(path_x - boat_x, path_y - boat_y)
        idx_min = int(np.argmin(dists))
        self.s_current = max(self.s_current, s_search[idx_min])

        # Adaptacyjny lookahead na podstawie krzywizny
        self.lookahead = self._compute_adaptive_lookahead()

        # Punkt celu z wyprzedzeniem
        s_target  = min(self.s_current + self.lookahead, s_max)
        target_x  = float(self.path_spline_x(s_target))
        target_y  = float(self.path_spline_y(s_target))

        dx_ds = float(self.path_spline_x(s_target, 1))
        dy_ds = float(self.path_spline_y(s_target, 1))
        psi_path = np.arctan2(dy_ds, dx_ds)

        self.target = np.array([target_x, target_y, psi_path,
                                 0.0, 0.0, 0.0])
        return float(dists[idx_min])

    # ── MODEL (RK4 przez CasADi) ─────────────────────────────────────────

    def _rollout(self, u_opt, x0, alpha0, w0):
        Np, Nc = self.Np, self.Nc
        u_seq = np.zeros(2 * Np)
        u_seq[:2*Nc]     = u_opt
        u_seq[2*Nc::2]   = u_opt[-2]
        u_seq[2*Nc+1::2] = u_opt[-1]

        states = [np.asarray(x0).flatten()]
        alpha  = float(alpha0)
        w      = float(w0)

        for i in range(Np):
            V    = float(u_seq[2*i])
            acmd = float(np.radians(u_seq[2*i+1]))
            alpha = float(boat_model.servo_step(alpha, acmd))
            w     = float(boat_model.motor_step(w, V))
            T     = boat_model.c_T * w * abs(w)
            x_next = np.asarray(
                boat_model.boat_step(states[-1], T, alpha)).flatten()
            states.append(x_next)

        return np.array(states)

    # ── FUNKCJA KOSZTU ───────────────────────────────────────────────────

    def cost_function(self, u_opt, x_current, current_alpha, current_w):
        states = self._rollout(u_opt, x_current, current_alpha, current_w)

        cost = 0.0
        tx, ty    = self.target[0], self.target[1]
        psi_path  = self.target[2]

        for i in range(1, self.Np + 1):
            x, y, psi = states[i][0], states[i][1], states[i][2]
            dx = tx - x;  dy = ty - y
            cost += self.Q_pos * np.sqrt(dx**2 + dy**2)
            cost += self.Q_psi * (1.0 - np.cos(psi - psi_path))

        for i in range(1, self.Nc):
            dV   = u_opt[2*i]   - u_opt[2*(i-1)]
            dalp = u_opt[2*i+1] - u_opt[2*(i-1)+1]
            cost += self.R_dV * dV**2 + self.R_dAlpha * dalp**2

        return cost

    # ── STEROWANIE ───────────────────────────────────────────────────────

    def compute_control(self, x_current, current_alpha, current_w):
        if self.u_prev is None:
            u_init = np.zeros(2 * self.Nc)
            u_init[0::2] = 12.0
        else:
            u_init = np.roll(self.u_prev, -2)
            u_init[-2] = self.u_prev[-2]
            u_init[-1] = self.u_prev[-1]

        res = minimize(
            self.cost_function, u_init,
            args=(x_current, current_alpha, current_w),
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter': 50, 'ftol': 1e-4}
        )

        self.predicted_path = self._rollout(
            res.x, x_current, current_alpha, current_w)
        self.u_prev = res.x
        return float(res.x[0]), float(res.x[1])