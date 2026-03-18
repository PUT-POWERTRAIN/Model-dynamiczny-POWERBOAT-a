import numpy as np
from scipy.optimize import minimize
import PowerBoat_Model_Python_2silnik as boat_model
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import math

class SimpleMPC:
    def __init__(self):
        self.Np = 25   # Horyzont przewidywania (dawne N)
        self.Nc = 3    # Horyzont sterowania (NOWOŚĆ)
        self.dt = 1
        self.target = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        self.t = 0
        
        # ZMIANA: Granice tylko dla ilości kroków sterowania (Nc)
        self.bounds = [(0, 24), (-45, 45)] * self.Nc
        self.u_prev = None
        self.predicted_path = None

    def model_predykcyjny(self, x_start, u_sequence, current_alpha, current_w):
        x_pred = [np.asarray(x_start).flatten()]
        current_alpha = float(np.asarray(current_alpha).flat[0])
        current_w = float(np.asarray(current_w).flat[0])

        # ZMIANA: Pętla leci po Np (chcemy widzieć daleko w przyszłość)
        for i in range(self.Np):
            V_cmd = u_sequence[2*i]
            alpha_cmd = u_sequence[2*i+1]

            sol_servo = solve_ivp(
                fun=boat_model.servo_ode, t_span=(0, self.dt),
                y0=[current_alpha], args=(np.radians(alpha_cmd),), method='RK23'
            )
            current_alpha = float(sol_servo.y[0, -1])

            sol_motor = solve_ivp(
                fun=boat_model.drive_motor_ode, t_span=(0, self.dt),
                y0=[current_w], args=(V_cmd,), method='RK23'
            )
            current_w = float(sol_motor.y[0, -1])
            T_actual = boat_model.c_T * current_w * abs(current_w)

            sol_boat = solve_ivp(
                fun=boat_model.boat_ode, t_span=(0, self.dt),
                y0=x_pred[-1], args=(T_actual, current_alpha, np.array([0, 0, 0])),
                method='RK23', max_step=self.dt
            )
            x_pred.append(sol_boat.y[:, -1])

        return np.array(x_pred)

    def cost_function(self, u_opt, x_current, current_alpha, current_w):
        # ==========================================
        # 1. ROZBUDOWA WEKTORA STEROWANIA (Nc -> Np)
        # ==========================================
        # Tworzymy pełny wektor dla horyzontu predykcji (Np)
        u_sequence = np.zeros(2 * self.Np)
        
        # Wpisujemy ruchy zoptymalizowane dla horyzontu sterowania (Nc)
        u_sequence[0:2*self.Nc] = u_opt  
        
        # Zamrażamy ostatnie wyliczone wartości na resztę horyzontu (od Nc do Np)
        u_sequence[2*self.Nc::2] = u_opt[-2]   # Napięcie V
        u_sequence[2*self.Nc+1::2] = u_opt[-1] # Kąt steru alpha

        # Symulacja modelu dla całego horyzontu
        states = self.model_predykcyjny(x_current, u_sequence, current_alpha, current_w)
        
        error_cost = 0.0
        effort_cost = 0.0

        # ==========================================
        # 2. SEKCJA STROJENIA WAG
        # ==========================================
        Q_pos  = 10.0      # Waga dążenia do współrzędnych celu
        Q_psi  = 30.0      # Waga błędu kursu (wysoka, by łódź "patrzyła" na cel)
        Q_vel  = 1.0       # Lekka kara za prędkość całkowitą (zapobiega "orbitowaniu")
        
        R_dV   = 0.5       # Kara za zmianę napięcia (zapobiega szarpaniu silnikiem)
        R_dAlpha = 15.0    # Kara za gwałtowne ruchy sterem (uspokaja wężykowanie)
        # ==========================================

        target_x, target_y = self.target[0], self.target[1]

        # ==========================================
        # 3. KOSZT BŁĘDU STANU (Pętla po Np)
        # ==========================================
        for i in range(1, self.Np + 1):
            x, y, psi, u, v, r = states[i]
            
            # Wektory odległości
            dx = target_x - x
            dy = target_y - y
            dist_sq = dx**2 + dy**2

            # A) Błąd odległości (im bliżej celu, tym mniejszy koszt)
            error_cost += Q_pos * np.sqrt(dist_sq)

            # B) Płynny błąd kursu (Line-of-Sight)
            # Funkcja (1 - cos) tworzy gładką "miskę" dla optymalizatora, bez ostrych skoków
            psi_target = np.arctan2(dy, dx)
            error_cost += Q_psi * (1.0 - np.cos(psi - psi_target))

            # C) Koszt prędkości
            # Utrzymuje prędkość w ryzach, by łódź mogła wyrobić się w zakręcie
            error_cost += Q_vel * (u**2 + v**2)

        # ==========================================
        # 4. KOSZT WYSIŁKU STERUJĄCEGO (Pętla po Nc)
        # ==========================================
        # Używamy kwadratów różnic, aby algorytm SLSQP łatwiej liczył gradienty
        for i in range(1, self.Nc):
            delta_V = u_opt[2*i] - u_opt[2*(i-1)]
            delta_alpha = u_opt[2*i+1] - u_opt[2*(i-1)+1]
            
            effort_cost += R_dV * (delta_V**2)
            effort_cost += R_dAlpha * (delta_alpha**2)

        return error_cost + effort_cost

    def compute_control(self, x_current, current_alpha, current_w):
        # Inicjalizacja pierwszej próby - rozmiar 2*Nc
        if self.u_prev is None:
            u_init = np.zeros(2 * self.Nc)
            u_init[0::2] = 12.0
            u_init[1::2] = 0.0
        else:
            u_init = np.roll(self.u_prev, -2)
            u_init[-2] = self.u_prev[-2]
            u_init[-1] = self.u_prev[-1]
            
        res = minimize(self.cost_function, u_init,
                       args=(x_current, current_alpha, current_w),
                       bounds=self.bounds, method='SLSQP',
                       options={'maxiter': 50, 'ftol': 1e-4})

        # Do rysowania predykcji musimy ponownie rozszerzyć wyliczone u do Np
        u_full = np.zeros(2 * self.Np)
        u_full[0:2*self.Nc] = res.x
        u_full[2*self.Nc::2] = res.x[-2]
        u_full[2*self.Nc+1::2] = res.x[-1]

        self.predicted_path = self.model_predykcyjny(
            x_current, u_full, current_alpha, current_w
        )
        self.u_prev = res.x
        
        # Zwracamy tylko akcję dla aktualnej chwili t=0
        return res.x[0], res.x[1]








# ==========================================
# FUNKCJE RYSOWANIA
# ==========================================
def draw_boat(ax, x, y, psi, size=0.2):
    """Rysuje łódź jako trójkąt w pozycji (x,y) z kursem psi"""
    L = size * 1.5
    W = size * 0.8
    local = np.array([
        [ L,      0],
        [-L * 0.5,  W],
        [-L * 0.5, -W],
        [ L,      0],
    ])
    R = np.array([[np.cos(psi), -np.sin(psi)],
                  [np.sin(psi),  np.cos(psi)]])
    world = (R @ local.T).T
    # oś X = Północ, oś Y = Wschód
    return ax.fill(world[:, 1] + y, world[:, 0] + x,
                   color='royalblue', zorder=5, alpha=0.95)[0]


def init_plot(target):
    fig = plt.figure(figsize=(13, 8), facecolor='#1a1a2e')
    fig.suptitle('MPC — Autonomiczna łódź', color='white', fontsize=14, y=0.98)

    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.35,
                          left=0.07, right=0.97, top=0.93, bottom=0.08)

    # --- MAPA ---
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('#0d1b2a')
    ax.set_xlabel('Y — Wschód [m]', color='white', fontsize=10)
    ax.set_ylabel('X — Północ [m]', color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.25, color='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    # Cel
    tx, ty = target[0], target[1]
    ax.plot(ty, tx, '*', color='yellow', markersize=16, zorder=6)
    ax.add_patch(plt.Circle((ty, tx), 0.4, color='yellow', alpha=0.15, zorder=3))
    ax.annotate('CEL', (ty, tx), textcoords='offset points',
                xytext=(8, 8), color='yellow', fontsize=9)

    # Linie dynamiczne
    traj_line,  = ax.plot([], [], '-',    color='#00d4ff', lw=2,   alpha=0.8,
                           zorder=3, label='Trasa')
    pred_line,  = ax.plot([], [], 'o--',  color='#ff6b35', lw=1.5, alpha=0.75,
                           markersize=4, zorder=4, label='Predykcja MPC')
    start_dot,  = ax.plot([0], [0], 'o',  color='lime',    markersize=8,
                           zorder=4, label='Start')

    ax.legend(loc='upper left', facecolor='#1a1a2e',
              labelcolor='white', framealpha=0.7, fontsize=9)
    ax.set_xlim(-2, 14)
    ax.set_ylim(-6, 6)

    # --- PANEL TELEMETRII ---
    ax_info = fig.add_subplot(gs[1])
    ax_info.set_facecolor('#0d1b2a')
    ax_info.axis('off')
    ax_info.set_title('Telemetria', color='white', fontsize=11, pad=8)

    fields = [
        ('Czas sym.',  't'),
        ('X (Północ)', 'x'),
        ('Y (Wschód)', 'y'),
        ('Kurs ψ',     'psi'),
        ('Prędkość u', 'u_vel'),
        ('Napięcie V', 'V'),
        ('Kąt steru α','alpha'),
        ('Dist. do celu','dist'),
    ]
    texts = {}
    for idx, (label, key) in enumerate(fields):
        ypos = 0.93 - idx * 0.11
        ax_info.text(0.04, ypos, label + ':',
                     color='#aaaaaa', fontsize=10, transform=ax_info.transAxes,
                     va='top')
        texts[key] = ax_info.text(0.60, ypos, '—',
                                   color='white', fontsize=10, fontweight='bold',
                                   transform=ax_info.transAxes, va='top')

    return fig, ax, ax_info, traj_line, pred_line, texts


# ==========================================
# SYMULACJA
# ==========================================
mpc = SimpleMPC()
current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
current_alpha = 0.0
current_w = 0.0

traj_x, traj_y = [0.0], [0.0]
boat_patch = [None]

fig, ax_map, ax_info, traj_line, pred_line, texts = init_plot(mpc.target)
plt.ion()
plt.show()

print(f"Start: {current_state.tolist()}, Cel: {mpc.target}")

for step in range(200):
    t0 = time.time()
    V_cmd, alpha_cmd = mpc.compute_control(current_state, current_alpha, current_w)
    t_opt = time.time() - t0

    V_in    = np.clip(V_cmd, 0.0, 24.0)
    alpha   = np.radians(np.clip(alpha_cmd, -45, 45))

    sol_servo = solve_ivp(boat_model.servo_ode,
                          (0, mpc.dt), [current_alpha],
                          args=(alpha,), method='RK45')
    current_alpha = float(sol_servo.y[0, -1])

    sol_motor = solve_ivp(boat_model.drive_motor_ode,
                          (0, mpc.dt), [current_w],
                          args=(V_in,), method='RK45')
    current_w = float(sol_motor.y[0, -1])
    T_actual  = boat_model.c_T * current_w * abs(current_w)

    sol_boat = solve_ivp(boat_model.boat_ode,
                         (0, mpc.dt), current_state,
                         args=(T_actual, current_alpha, np.array([0, 0, 0])),
                         method='RK45', max_step=mpc.dt)
    current_state = sol_boat.y[:, -1]

    traj_x.append(current_state[0])
    traj_y.append(current_state[1])

    # --- Aktualizacja mapy ---
    traj_line.set_data(traj_y, traj_x)

    if mpc.predicted_path is not None:
        p = mpc.predicted_path
        pred_line.set_data(p[:, 1], p[:, 0])

    if boat_patch[0] is not None:
        boat_patch[0].remove()
    boat_patch[0] = draw_boat(ax_map,
                               current_state[0], current_state[1],
                               current_state[2])

    # Dynamiczny zakres mapy
    margin = 2.5
    all_x = traj_x + [mpc.target[0]]
    all_y = traj_y + [mpc.target[1]]
    if mpc.predicted_path is not None:
        all_x += mpc.predicted_path[:, 0].tolist()
        all_y += mpc.predicted_path[:, 1].tolist()
    ax_map.set_xlim(min(all_y) - margin, max(all_y) + margin)
    ax_map.set_ylim(min(all_x) - margin, max(all_x) + margin)

    # --- Aktualizacja telemetrii ---
    dist = np.sqrt((current_state[0] - mpc.target[0])**2 +
                   (current_state[1] - mpc.target[1])**2)
    texts['t'].set_text(f'{mpc.t:.1f} s')
    texts['x'].set_text(f'{current_state[0]:.3f} m')
    texts['y'].set_text(f'{current_state[1]:.3f} m')
    texts['psi'].set_text(f'{np.degrees(current_state[2]) % 360:.1f}°')
    texts['u_vel'].set_text(f'{current_state[3]:.3f} m/s')
    texts['V'].set_text(f'{V_cmd:.2f} V')
    texts['alpha'].set_text(f'{alpha_cmd:.2f}°')
    texts['dist'].set_text(f'{dist:.3f} m')

    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f"Krok {step:3d} | "
          f"Poz=[{current_state[0]:6.3f}, {current_state[1]:6.3f}] | "
          f"V={V_cmd:5.2f}V | α={alpha_cmd:5.2f}° | "
          f"opt={t_opt:.2f}s | dist={dist:.3f}m")

    mpc.t += mpc.dt

plt.ioff()
plt.show()