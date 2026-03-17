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
    tau_env = np.array([-5, 0, 0])
    
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

# ==========================================
# 4. WIZUALIZACJA I ANIMACJA 
# ==========================================
print("Startuję animację... (Zostaw okno otwarte)")

x_hist = states_sim[:, 0]
y_hist = states_sim[:, 1]
psi_hist = states_sim[:, 2]
u_hist = states_sim[:, 3]
v_hist = states_sim[:, 4]  

fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(5, 2, width_ratios=[1, 1.5]) # <--- ZMIANA: 5 wierszy!

# 1. Wykres wzdłużnej (u)
ax_u = fig.add_subplot(gs[0, 0])
ax_u.plot(time_steps, u_hist, 'b-', lw=2)
ax_u.set_ylabel('u [m/s]\n(Wzdłużna)')
ax_u.grid(True)

# 2. Wykres poprzecznej (v) - POWRACA!
ax_v = fig.add_subplot(gs[1, 0], sharex=ax_u)
ax_v.plot(time_steps, v_hist, 'r-', lw=2) 
ax_v.set_ylabel('v [m/s]\n(Poprzeczna)')
ax_v.grid(True)

# 3. Wykres Napięcia (V)
ax_V = fig.add_subplot(gs[2, 0], sharex=ax_u)
ax_V.plot(time_steps, V_hist, 'r-', lw=2, color='orange') 
ax_V.set_ylabel('Napięcie\nWej. [V]')
ax_V.grid(True)

# 4. Wykres Ciągu (T)
ax_T = fig.add_subplot(gs[3, 0], sharex=ax_u)
ax_T.plot(time_steps, T_hist, 'g-', lw=2)
ax_T.set_ylabel('Ciąg T [N]')
ax_T.grid(True)

# 5. Wykres Kąta Steru (alpha)
ax_alpha = fig.add_subplot(gs[4, 0], sharex=ax_u)
ax_alpha.plot(time_steps, np.degrees(alpha_hist), 'm-', lw=2) 
ax_alpha.set_ylabel('Kąt α [°]')
ax_alpha.set_xlabel('Czas symulacji [s]')
ax_alpha.grid(True)

# Pionowe linie skanera
line_u = ax_u.axvline(x=0, color='k', linestyle='--', alpha=0.7)
line_v = ax_v.axvline(x=0, color='k', linestyle='--', alpha=0.7)
line_V = ax_V.axvline(x=0, color='k', linestyle='--', alpha=0.7)
line_T = ax_T.axvline(x=0, color='k', linestyle='--', alpha=0.7)
line_alpha = ax_alpha.axvline(x=0, color='k', linestyle='--', alpha=0.7)

# --- MAPA (Rozciągnięta na całą prawą kolumnę) ---
ax_map = fig.add_subplot(gs[:, 1]) 
ax_map.set_xlabel('Oś Y (Wschód) [m]')
ax_map.set_ylabel('Oś X (Północ) [m]')
ax_map.set_aspect('equal') 
ax_map.grid(True, linestyle='--', alpha=0.7)

margin = 5.0
ax_map.set_xlim(np.min(y_hist) - margin, np.max(y_hist) + margin)
ax_map.set_ylim(np.min(x_hist) - margin, np.max(x_hist) + margin)

line_traj, = ax_map.plot([], [], 'b-', linewidth=2, label='Ślad (Trajektoria)')
boat_shape, = ax_map.plot([], [], 'r-', linewidth=3, label='Dziób ')
ax_map.legend()

L = 4.0  
W = 2.0  
boat_local = np.array([[L/2, 0], [-L/2, -W/2], [-L/2, W/2], [L/2, 0]])

fig.tight_layout()

for i in range(0, len(time_steps), 2):
    t_current = time_steps[i]
    
    line_u.set_xdata([t_current, t_current])
    line_v.set_xdata([t_current, t_current])
    line_V.set_xdata([t_current, t_current])
    line_T.set_xdata([t_current, t_current])
    line_alpha.set_xdata([t_current, t_current])
    
    line_traj.set_data(y_hist[:i], x_hist[:i])
    
    x_i = x_hist[i]
    y_i = y_hist[i]
    psi_i = psi_hist[i]
    
    ax_map.set_title(f'Animacja Trajektorii USV 4m (Czas: {t_current:.1f} s)')
    
    rotated_x = x_i + boat_local[:, 0] * np.cos(psi_i) - boat_local[:, 1] * np.sin(psi_i)
    rotated_y = y_i + boat_local[:, 0] * np.sin(psi_i) + boat_local[:, 1] * np.cos(psi_i)
    boat_shape.set_data(rotated_y, rotated_x)
    plt.pause(0.01)

plt.show()
