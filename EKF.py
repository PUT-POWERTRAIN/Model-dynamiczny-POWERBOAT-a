import casadi as ca
import numpy as np
import casadi_model as boat_model

class EKF:
    def __init__(self):
        # stan: [x,y,psi,u,v,r]
        self.x_hat = np.zeros((6,1))
        self.d = np.zeros((6,1))  # zakłócenie (bias dynamiki)
        self.w_hat = 0.0          # estymata prędkości obrotowej
        self.alpha_hat = 0.0      # estymata wychylenia steru
        self.P = np.eye(12) * 0.1 # kowariancja
        self.Q = np.eye(12) * 0.01 # szumy
        self.R = np.eye(4) * 0.05
        self.H = np.array([ # macierz pomiaru
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,0,0,1]
        ])
        # CasADi Jacobian
        x = ca.MX.sym('x',6)
        T = ca.MX.sym('T')
        alpha = ca.MX.sym('alpha')
        x_next = boat_model.boat_step(x, T, alpha)
        self.A_fun = ca.Function('A_fun', [x, T, alpha],
                                 [ca.jacobian(x_next, x)])

    def predict(self, u):
        V = u[0]
        alpha_cmd = u[1]
        
        self.alpha_hat = float(boat_model.servo_step(self.alpha_hat, alpha_cmd))
        self.w_hat = float(boat_model.motor_step(self.w_hat, V))
        
        T_actual = boat_model.c_T * self.w_hat * abs(self.w_hat)

        x_pred = np.array( # predykcja stanu
            boat_model.boat_step(self.x_hat, T_actual, self.alpha_hat)
        ).reshape(6,1)
        x_pred = x_pred + self.d # dodanie zakłócenia

        A_x = np.array(self.A_fun(self.x_hat, T_actual, self.alpha_hat)) # Jakobian
        A = np.block([ # rozszerzony Jacobian
            [A_x, np.eye(6)],
            [np.zeros((6,6)), np.eye(6)]
        ])

        # rozszerzony stan o zaklocenia
        z = np.vstack((self.x_hat, self.d))
        z_pred = np.vstack((x_pred, self.d))

        self.P = A @ self.P @ A.T + self.Q

        self.x_hat = z_pred[:6]
        self.d = z_pred[6:]

    def update(self, y):
        y_pred = self.H @ self.x_hat # predykcja pomiaru

        e = y.reshape(-1,1) - y_pred # blad miedzy obiektem rzezywistym a modelem

        C = np.block([
            [self.H, np.eye(4,6)]   # uproszczony model biasu
        ])

        S = C @ self.P @ C.T + self.R # macierz innowacji
        K = self.P @ C.T @ np.linalg.inv(S) # wzmocnienie Kalmana

        z = np.vstack((self.x_hat, self.d)) # aktualizacja
        z = z + K @ e

        self.x_hat = z[:6]
        self.d = z[6:]
        print(f"estymata bledu: {self.d}")
        I = np.eye(12)
        self.P = (I - K @ C) @ self.P