import casadi as ca
import numpy as np

# ==========================================
# PARAMETRY
# ==========================================
# Zmienione, gorsze parametry masowe (łódź dostała 200kg ładunku)
m=350.0; Iz=300.0; L_m=1.5
X_du,Y_dv,N_dr = -15.0,-150.0,-25.0

# Dużo silniejsze liniowe opory wody
X_u, Y_v, N_r  = -80.0,-180.0,-500.0

M_inv = np.linalg.inv(np.array([
    [m-X_du, 0,      0      ],
    [0,      m-Y_dv, 0      ],
    [0,      0,      Iz-N_dr],
]))
M_inv_ca = ca.DM(M_inv)

k_walk=0.01; J_m=0.2; R_m=0.5; k_t=0.1; k_e=0.1; b_m=0.01

# Słabszy uciąg silnika (MPC myśli, że ma współczynnik 0.05, a ma tylko 0.035)
c_Q=0.0005; c_T=0.035
T_servo=3.0; max_rate=float(np.radians(60.0))

DT = 0.2  # krok czasowy [s] — wbudowany w steppery (zmniejszone dla stabilności)

# ==========================================
# ODE
# ==========================================
def make_boat_ode():
    x   = ca.MX.sym('x', 6)
    T   = ca.MX.sym('T')
    alp = ca.MX.sym('alpha')
    psi=x[2]; u=x[3]; v=x[4]; r=x[5]
    nu=ca.vertcat(u,v,r)
    
    # 1. Standardowe siły pędnika i steru
    tau_prop=ca.vertcat(T*ca.cos(alp), T*ca.sin(alp),
                   -L_m*T*ca.sin(alp)+k_walk*T)
    
    # 2. Stałe zakłócenie środowiskowe, tak samo jak w PowerBoat_Model_Python_2silnik
    # [Surge, Sway, Yaw] - odpychanie łodzi z mocą -5 do tyłu
    tau_env = ca.vertcat(-5.0, 0.0, 0.0)
    
    # Całkowity nacisk/siły na wektor kadłuba (współrzędne lokalne)
    tau_total = tau_prop + tau_env
    
    C=ca.vertcat(ca.horzcat(0,0,-(m-Y_dv)*v),
                 ca.horzcat(0,0, (m-X_du)*u),
                 ca.horzcat((m-Y_dv)*v,-(m-X_du)*u,0))
    
    # Nieliniowe opory wody (kwadratowe), których nominalny model MPC w ogóle niema!
    D_lin = ca.diag([-X_u, -Y_v, -N_r]) @ nu
    D_quad = ca.vertcat(12.0 * u * ca.fabs(u),
                        40.0 * v * ca.fabs(v),
                        150.0 * r * ca.fabs(r))
    D_total = D_lin + D_quad

    J=ca.vertcat(ca.horzcat(ca.cos(psi),-ca.sin(psi),0),
                 ca.horzcat(ca.sin(psi), ca.cos(psi),0),
                 ca.horzcat(0,0,1))
    
    # Używamy zsumowanego tau i nieliniowych oporów
    nu_dot=M_inv_ca@(tau_total-C@nu-D_total)
    eta_dot=J@nu
    return ca.Function('boat_ode',[x,T,alp],[ca.vertcat(eta_dot,nu_dot)])

def make_motor_ode():
    w=ca.MX.sym('w'); V=ca.MX.sym('V')
    dw=(k_t*(V-k_e*w)/R_m - b_m*w - c_Q*w*ca.fabs(w))/J_m
    return ca.Function('motor_ode',[w,V],[dw])

def make_servo_ode():
    alp=ca.MX.sym('alpha'); cmd=ca.MX.sym('alpha_cmd')
    dalpha=ca.fmin(ca.fmax((cmd-alp)/T_servo, -max_rate), max_rate)
    return ca.Function('servo_ode',[alp,cmd],[dalpha])

# ==========================================
# RK4 — dt WBUDOWANE jako stała Python
# ==========================================
def _make_rk4(ode_fn, state_dim, *param_dims, dt=DT):
    """state_dim=1 → skalar MX, >1 → wektor MX"""
    s = ca.MX.sym('s') if state_dim==1 else ca.MX.sym('s', state_dim)
    params = [ca.MX.sym(f'p{i}') if d==1 else ca.MX.sym(f'p{i}',d)
              for i,d in enumerate(param_dims)]
    def f(st): return ode_fn(st, *params)
    k1=f(s); k2=f(s+dt/2*k1); k3=f(s+dt/2*k2); k4=f(s+dt*k3)
    s_next = s + dt/6*(k1+2*k2+2*k3+k4)
    return ca.Function('rk4', [s]+params, [s_next])

boat_ode_fn  = make_boat_ode()
motor_ode_fn = make_motor_ode()
servo_ode_fn = make_servo_ode()

boat_step  = _make_rk4(boat_ode_fn,  6, 1, 1)  # (x, T, alpha)
motor_step = _make_rk4(motor_ode_fn, 1, 1)     # (w, V)
servo_step = _make_rk4(servo_ode_fn, 1, 1)     # (alpha, alpha_cmd)
