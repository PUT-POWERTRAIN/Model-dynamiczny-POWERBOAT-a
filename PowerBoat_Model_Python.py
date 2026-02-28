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

def boat_ode(t, state, T, alpha, tau_env):
    x, y, psi, u, v, r = state
    nu = np.array([u, v, r])
    
    tau_silnika = np.array([
        T * np.cos(alpha),
        T * np.sin(alpha),
        -L_m * T * np.sin(alpha)
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
    
    nu_dot = np.dot(M_inv, tau_silnika + tau_env - np.dot(C, nu) - np.dot(D, nu))
    
    J = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])
    eta_dot = np.dot(J, nu)
    
    return np.concatenate((eta_dot, nu_dot))

# ==========================================
# 2. PĘTLA SYMULATORA
# ==========================================
dt = 0.1
T_max = 30.0  
time_steps = np.arange(0, T_max, dt)

states_sim = np.zeros((len(time_steps), 6))
current_state = np.zeros(6) 


T_hist = np.zeros(len(time_steps))
alpha_hist = np.zeros(len(time_steps))

for i in range(len(time_steps)):
    t = time_steps[i]  
    states_sim[i, :] = current_state
    
    # --- ZADAWANIE WEJŚĆ  ---
    if t < 5.0:
        T_cmd = 0.0
        alpha_cmd = 0.0
    elif 5.0 <= t < 10.0:
        
        T_cmd = 150.0
        alpha_cmd = np.radians(-15.0) 
    elif 10.0 <= t < 15.0:
        
        T_cmd = 0.0
        alpha_cmd = 0.0
    else:
        
        T_cmd = 150.0
        alpha_cmd = 0.0
        
    T = np.clip(T_cmd, 0.0, 200.0)
    alpha = np.clip(alpha_cmd, np.radians(-45), np.radians(45))
    
    
    T_hist[i] = T
    alpha_hist[i] = alpha
    
    tau_env = np.array([0, 0, 0])
    
    sol = solve_ivp(
        fun=boat_ode,
        t_span=(t, t + dt),
        y0=current_state,
        args=(T, alpha, tau_env),
        method='RK45',
        max_step=dt
    )
    current_state = sol.y[:, -1]

# ==========================================
# 3. WIZUALIZACJA I ANIMACJA (GridSpec)
# ==========================================
print("Startuję animację... (Zostaw okno otwarte)")

x_hist = states_sim[:, 0]
y_hist = states_sim[:, 1]
psi_hist = states_sim[:, 2]
u_hist = states_sim[:, 3]
v_hist = states_sim[:, 4]  

# Konfiguracja głównego okna
fig = plt.figure(figsize=(15, 8))

gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1.5]) 

# --- LEWA STRONA (4 Wykresy Telemetrii) ---

ax_u = fig.add_subplot(gs[0, 0])
ax_u.plot(time_steps, u_hist, 'b-', lw=2)
ax_u.set_ylabel('u [m/s]\n(Wzdłużna)')
ax_u.grid(True)

ax_v = fig.add_subplot(gs[1, 0], sharex=ax_u)
ax_v.plot(time_steps, v_hist, 'r-', lw=2)
ax_v.set_ylabel('v [m/s]\n(Poprzeczna)')
ax_v.grid(True)

ax_T = fig.add_subplot(gs[2, 0], sharex=ax_u)
ax_T.plot(time_steps, T_hist, 'g-', lw=2)
ax_T.set_ylabel('Ciąg T [N]')
ax_T.grid(True)

ax_alpha = fig.add_subplot(gs[3, 0], sharex=ax_u)

ax_alpha.plot(time_steps, np.degrees(alpha_hist), 'm-', lw=2) 
ax_alpha.set_ylabel('Kąt α [°]')
ax_alpha.set_xlabel('Czas symulacji [s]')
ax_alpha.grid(True)


line_u = ax_u.axvline(x=0, color='k', linestyle='--', alpha=0.7)
line_v = ax_v.axvline(x=0, color='k', linestyle='--', alpha=0.7)
line_T = ax_T.axvline(x=0, color='k', linestyle='--', alpha=0.7)
line_alpha = ax_alpha.axvline(x=0, color='k', linestyle='--', alpha=0.7)

# --- PRAWA STRONA (Mapa z Trajektorią) ---
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

L = 0.8  
W = 0.4  
boat_local = np.array([
    [L/2, 0],         
    [-L/2, -W/2],     
    [-L/2, W/2],      
    [L/2, 0]          
])

fig.tight_layout() # Układa wykresy

# Główna pętla animacji
for i in range(0, len(time_steps), 2):
    t_current = time_steps[i]
    
    # Aktualizacja skanera czasu na lewych wykresach
    line_u.set_xdata([t_current, t_current])
    line_v.set_xdata([t_current, t_current])
    line_T.set_xdata([t_current, t_current])
    line_alpha.set_xdata([t_current, t_current])
    
    # Aktualizacja mapy po prawej
    line_traj.set_data(y_hist[:i], x_hist[:i])
    
    x_i = x_hist[i]
    y_i = y_hist[i]
    psi_i = psi_hist[i]
    
    ax_map.set_title(f'Animacja Trajektorii Trimarana (Czas: {t_current:.1f} s)')
    
    rotated_x = x_i + boat_local[:, 0] * np.cos(psi_i) - boat_local[:, 1] * np.sin(psi_i)
    rotated_y = y_i + boat_local[:, 0] * np.sin(psi_i) + boat_local[:, 1] * np.cos(psi_i)
    
    boat_shape.set_data(rotated_y, rotated_x)
    
    plt.pause(0.01)

plt.show()