"""
EKF — Offset-Free dla Path Following
Estymuje PRAWZIWY stan obiektu (x_true) oraz bias (d).
Równanie predykcji: x_true(k+1) = f(x_true) + D_ext * d
"""

import numpy as np
import casadi as ca
import casadi_model as boat_model

H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=float)

# D_ext: Jak bias [du, dv, dr] wpływa na STAN (dodawany do u, v, r)
D_ext = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=float)

class EKF:
    def __init__(self, Q_x=None, Q_d=None, R_y=None):
        self.x_hat = np.zeros((6, 1))  # Prawdziwy stan
        self.d = np.zeros((3, 1))      # Bias [du, dv, dr]
        self.w_hat = 0.0
        self.alpha_hat = 0.0
        self._initialized = False

        self.P = np.eye(9) * 0.1

        if Q_x is None:
            Q_x = np.diag([0.01, 0.01, 0.005, 0.1, 0.2, 0.05])
        if Q_d is None:
            Q_d = np.diag([0.1, 0.2, 0.05])
        if R_y is None:
            R_y = np.diag([0.05, 0.05, 0.01, 0.02])

        self.Q = np.block([
            [Q_x,               np.zeros((6, 3))],
            [np.zeros((3, 6)),  Q_d             ],
        ])
        self.R = R_y

        # C = [H | 0] bo d nie występuje bezpośrednio w równaniu pomiarowym
        self._C = np.hstack([H, np.zeros((4, 3))])

        x_sym = ca.MX.sym('x', 6)
        T_sym = ca.MX.sym('T')
        alp_sym = ca.MX.sym('alp')
        f_next = boat_model.boat_step(x_sym, T_sym, alp_sym)
        self._jac_fx = ca.Function('jac_fx', [x_sym, T_sym, alp_sym], [ca.jacobian(f_next, x_sym)])

    def update(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float).reshape(4, 1)

        y_pred = H @ self.x_hat
        e = y - y_pred
        e[2, 0] = _wrap_angle(e[2, 0])

        C = self._C
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.solve(S.T, np.eye(4)).T

        z = np.vstack([self.x_hat, self.d])
        z = z + K @ e
        
        self.x_hat = z[:6]
        self.d = z[6:]
        
        self.P = (np.eye(9) - K @ C) @ self.P
        self.P = (self.P + self.P.T) / 2  # Zapewnienie symetrii macierzy

    def predict(self, u):
        V_cmd = float(u[0])
        alpha_cmd = float(u[1])

        self.alpha_hat = float(boat_model.servo_step(self.alpha_hat, alpha_cmd))
        self.w_hat = float(boat_model.motor_step(self.w_hat, V_cmd))
        T_act = boat_model.c_T * self.w_hat * abs(self.w_hat)

        # PREDYKCJA PRAWDZIWEGO STANU: f(x_true) + D_ext * d
        x_pred = np.array(
            boat_model.boat_step(self.x_hat, T_act, self.alpha_hat)
        ).reshape(6, 1) + D_ext @ self.d

        A_x = np.array(self._jac_fx(self.x_hat, T_act, self.alpha_hat))
        A = np.block([
            [A_x,   D_ext],
            [np.zeros((3,6)), np.eye(3)],
        ])

        self.P = A @ self.P @ A.T + self.Q
        self.x_hat = x_pred

def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)