import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import time

# ==========================================
# 1. PARAMETRY ŁODZI I DYNAMIKI FOSSENA
# ==========================================
m = 150.0       
Iz = 140.0      
L_m = 1.5       

X_du, Y_dv, N_dr = -15.0, -150.0, -25.0
X_u, Y_v, N_r = -50.0, -100.0, -400.0  

M = np.array([
    [m - X_du, 0,          0],
    [0,        m - Y_dv,   0],
    [0,        0,          Iz - N_dr]
])
M_inv = np.linalg.inv(M)
k_walk = 0.01 # proppeler walk constant
# ==========================================
# 1A. PARAMETRY SILNIKA DC I ŚRUBY 
# ==========================================
J_m = 0.2    
R_m = 0.5    
k_t = 0.1     
k_e = 0.1     
b_m = 0.01    

c_Q = 0.0005  
c_T = 0.05   

# ==========================================
# 2. MODELE RÓWNAŃ RÓŻNICZKOWYCH (ODE)
# ==========================================

# MODEL SILNIKA: 
def drive_motor_ode(t, state, V):
    w = state[0] 
    
    
    
    Back_EMF = k_e * w
    i = (V - Back_EMF) / R_m 
    
    
    tau_motor = k_t * i
    
    
    tau_friction = b_m * w
    tau_propeller = c_Q * w * np.abs(w) 
    
     
    
    dw_dt = (tau_motor - tau_friction - tau_propeller) / J_m
    
    return [dw_dt]

def boat_ode(t, state, T, alpha, tau_env):
    x, y, psi, u, v, r = state
    nu = np.array([u, v, r])
    
    N_walk = k_walk * T 
    
    
    tau_silnika = np.array([
        T * np.cos(alpha),
        T * np.sin(alpha),
        -L_m * T * np.sin(alpha) + N_walk  
    ])
    
    C = np.array([
        [0, 0, -(m - Y_dv) * v],
        [0, 0,  (m - X_du) * u],
        [(m - Y_dv) * v, -(m - X_du) * u, 0]
    ])
    D = np.array([
        [-X_u, 0, 0],
        [0, -Y_v, 0],
        [0, 0, -N_r]
    ])
    J = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])

    tau_env_loc = np.dot(J.T,tau_env)
    nu_dot = np.dot(M_inv, tau_silnika + tau_env_loc - np.dot(C, nu) - np.dot(D, nu))
    eta_dot = np.dot(J, nu)
    
    return np.concatenate((eta_dot, nu_dot))

# --- PARAMETRY SERWA ---
T_servo = 3 # Stała czasowa (serwo reaguje w mgnieniu oka, ale nie natychmiast)
max_rate = np.radians(60.0) # Maksymalna prędkość obrotu (np. 60 stopni na sekundę)

def servo_ode(t, state, alpha_cmd):
    alpha = state[0]
    
    # 1. Wyliczamy, z jaką prędkością chce kręcić się inercja PT1
    rate = (alpha_cmd - alpha) / T_servo
    
    # 2. Narzucamy fizyczny limit prędkości obrotu silnika krokowego/serwa
    dalpha_dt = np.clip(rate, -max_rate, max_rate)
    
    return [dalpha_dt]
# ==========================================
# 3. PĘTLA SYMULATORA (KASKADA)
# ==========================================
dt = 0.1
T_max = 30.0  
time_steps = np.arange(0, T_max, dt)

states_sim = np.zeros((len(time_steps), 6))
current_state = np.zeros(6) 

current_alpha = np.array([0.0])
current_w = np.array([0.0]) 
T_hist = np.zeros(len(time_steps))
V_hist = np.zeros(len(time_steps)) 
alpha_hist = np.zeros(len(time_steps))
alpha_actual = 0
for i in range(len(time_steps)):
    t = time_steps[i]  
    states_sim[i, :] = current_state
    
    # --- ZADAWANIE WEJŚĆ ---
    
    V_cmd = 24.0
    alpha_cmd = 0.0
    
        
    V_in = np.clip(V_cmd, 0.0, 24.0)
    alpha = np.radians(np.clip(alpha_cmd, -45, 45))
    
    V_hist[i] = V_in
    alpha_hist[i] = alpha_actual
    tau_env = np.array([0, 0, 0])
    
    # 1. SOLVER SERWA
    sol_servo = solve_ivp(
        fun=servo_ode,
        t_span=(t, t + dt),
        y0=current_alpha,
        args=(alpha,),
        method='RK45'
    )
    current_alpha = sol_servo.y[:, -1]
    alpha_actual = current_alpha[0]
    
    sol_motor = solve_ivp(
        fun=drive_motor_ode,
        t_span=(t, t + dt),
        y0=current_w,
        args=(V_in,),
        method='RK45'
    )
    current_w = sol_motor.y[:, -1] 
    omega = current_w[0]
    
    
    T_actual = c_T * omega * np.abs(omega)
    T_hist[i] = T_actual
    
    
    sol_boat = solve_ivp(
        fun=boat_ode,
        t_span=(t, t + dt),
        y0=current_state,
        args=(T_actual, alpha_actual, tau_env), 
        method='RK45',
        max_step=dt
    )
    current_state = sol_boat.y[:, -1]
