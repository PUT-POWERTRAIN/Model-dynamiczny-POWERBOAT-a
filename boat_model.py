from casadi import SX, vertcat, DM, cos, sin, horzcat, diag
from acados_template import AcadosModel
import numpy as np

def export_boat_model() -> AcadosModel:

    model_name = 'autonomous_boat'

    # ==========================================
    # 1. ZMIENNE STANU (x)
    # ==========================================
    x_pos = SX.sym('x_pos')
    y_pos = SX.sym('y_pos')
    psi   = SX.sym('psi')
    u     = SX.sym('u')
    v     = SX.sym('v')
    r     = SX.sym('r')
    w     = SX.sym('w')       # prędkość obrotowa silnika
    alp   = SX.sym('alpha')   # kąt wychylenia pędnika/steru

    x = vertcat(x_pos, y_pos, psi, u, v, r, w, alp)

    # ==========================================
    # 2. ZMIENNE STEROWANIA (u)
    # ==========================================
    V       = SX.sym('V')         # Napięcie silnika
    alp_cmd = SX.sym('alpha_cmd') # Żądany kąt steru

    u_ctrl = vertcat(V, alp_cmd)

    # ==========================================
    # ZMIENNE POMOCNICZE / POCHODNE (xdot)
    # ==========================================
    xdot = SX.sym('xdot', 8)

    # Parametry fizyczne (te z Twojego kodu)
    m=150.0; Iz=140.0; L_m=1.5
    X_du, Y_dv, N_dr = -15.0, -150.0, -25.0
    X_u, Y_v, N_r = -50.0, -100.0, -400.0
    
    k_walk=0.01; J_m=0.2; R_m=0.5; k_t=0.1; k_e=0.1; b_m=0.01
    c_Q=0.0005; c_T=0.05; T_servo=3.0

    M_inv = np.linalg.inv(np.array([
        [m-X_du, 0,      0      ],
        [0,      m-Y_dv, 0      ],
        [0,      0,      Iz-N_dr],
    ]))
    M_inv_ca = DM(M_inv)

    # ==========================================
    # 3. RÓWNANIA RÓŻNICZKOWE (Ciągłe!)
    # ==========================================
    
    # A) Dynamika Silnika
    # Zastąpiłem ca.fabs(w) dla płynności. Jeśli silnik kręci się tylko 
    # w jedną stronę, w > 0 wystarczy uciąć fabs.
    dw = (k_t*(V - k_e*w)/R_m - b_m*w - c_Q*w*w) / J_m

    # B) Dynamika Serwa (BEZ fmin/fmax - limity nakładamy w OCP solverze!)
    dalp = (alp_cmd - alp) / T_servo

    # C) Siła Ciągu (powiązanie silnika z fizyką łodzi)
    T_thrust = c_T * w * w  # Zakładając ruch w przód. Dla tyłu: c_T * w * ca.fabs(w)

    # D) Dynamika Łodzi
    nu = vertcat(u, v, r)
    tau = vertcat(T_thrust*cos(alp), 
                     T_thrust*sin(alp),
                     -L_m*T_thrust*sin(alp) + k_walk*T_thrust)
                     
    C = vertcat(horzcat(0, 0, -(m-Y_dv)*v),
                   horzcat(0, 0, (m-X_du)*u),
                   horzcat((m-Y_dv)*v, -(m-X_du)*u, 0))
    D = diag([-X_u, -Y_v, -N_r])
    
    J = vertcat(horzcat(cos(psi), -sin(psi), 0),
                   horzcat(sin(psi),  cos(psi), 0),
                   horzcat(0, 0, 1))
                   
    nu_dot = M_inv_ca @ (tau - C@nu - D@nu)
    eta_dot = J @ nu

    # ==========================================
    # 4. ZŁOŻENIE MODELU ACADOS
    # ==========================================
    f_expl = vertcat(eta_dot, nu_dot, dw, dalp)

    model = AcadosModel()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u_ctrl
    model.name = model_name

    return model