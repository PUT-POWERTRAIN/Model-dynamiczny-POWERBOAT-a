"""
Symulator 2D w pygame dla klasy MPC (True Path Following).

Sterowanie:
  LPM        — dodaj waypoint do ścieżki
  PPM        — usuń ostatni waypoint
  ENTER      — zatwierdź ścieżkę i uruchom symulację
  SPACE      — pauza / wznów
  P          — pokaż / ukryj predykcję MPC
  T          — pokaż / ukryj ślad łodzi
  R          — reset
  ESC        — wyjście
  Scroll     — zoom
  MMB        — przesuwanie widoku
"""

import sys
import math
import numpy as np
import pygame

sys.path.insert(0, ".")
# Zmiana: importujemy nową klasę PathFollowingMPC
from casadi_mpc import PathFollowingMPC as MPC
import casadi_model as boat_model

# ==================== KONFIGURACJA ====================
SCREEN_W, SCREEN_H = 1280, 800
WORLD_W, WORLD_H = 200.0, 150.0
FPS = 30
SIM_DT = boat_model.DT  # musi = mpc.dt

# Kolory
C_BG = (15, 20, 35)
C_GRID = (30, 40, 60)
C_PATH = (40, 200, 120)
C_WAYPOINT = (60, 220, 140)
C_WAYPOINT_TMP = (180, 220, 80)
C_BOAT = (80, 160, 255)
C_PRED = (255, 160, 40)
C_TRAIL = (60, 100, 180)
C_TARGET = (200, 80, 255)
C_TEXT = (200, 210, 230)
C_TEXT_DIM = (100, 120, 150)
C_UI_BG = (20, 28, 48)
C_UI_BORDER = (50, 65, 100)

# ==================== KAMERA ====================
class Camera:
    def __init__(self):
        self.scale = min(SCREEN_W / WORLD_W, SCREEN_H / WORLD_H) * 0.85
        self.ox = SCREEN_W / 2 - (WORLD_W / 2) * self.scale
        self.oy = SCREEN_H / 2 - (WORLD_H / 2) * self.scale
        self._drag_start = None
        self._drag_origin = None

    def w2s(self, wx, wy):
        sx = self.ox + wx * self.scale
        sy = self.oy + (WORLD_H - wy) * self.scale
        return int(sx), int(sy)

    def s2w(self, sx, sy):
        wx = (sx - self.ox) / self.scale
        wy = WORLD_H - (sy - self.oy) / self.scale
        return wx, wy

    def zoom(self, factor, cx, cy):
        wx, wy = self.s2w(cx, cy)
        self.scale *= factor
        self.ox = cx - wx * self.scale
        self.oy = cy - (WORLD_H - wy) * self.scale

    def start_drag(self, sx, sy):
        self._drag_start = (sx, sy)
        self._drag_origin = (self.ox, self.oy)

    def update_drag(self, sx, sy):
        if self._drag_start:
            dx = sx - self._drag_start[0]
            dy = sy - self._drag_start[1]
            self.ox = self._drag_origin[0] + dx
            self.oy = self._drag_origin[1] + dy

    def end_drag(self):
        self._drag_start = None


# ==================== STAN SYMULACJI ====================
class SimState:
    def __init__(self):
        self.reset()

    def reset(self):
        # Stan łodzi
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.alpha = 0.0
        self.w = 0.0
        self.V_cmd = 0.0
        self.alpha_cmd_deg = 0.0
        self.step = 0

        # Ścieżka
        self.waypoints = []
        self.path_ready = False
        self.path_done = False

        # Wizualizacja
        self.trail = []
        self.pred_path = None
        self.target_point = None

        # UI
        self.show_pred = True
        self.show_trail = True
        self.paused = False

        # MPC
        self.mpc = MPC()


# ==================== RYSOWANIE ====================
def draw_grid(surf, cam):
    step = 10.0
    for x in np.arange(0, WORLD_W + step, step):
        p1 = cam.w2s(x, 0)
        p2 = cam.w2s(x, WORLD_H)
        pygame.draw.line(surf, C_GRID, p1, p2, 1)
    for y in np.arange(0, WORLD_H + step, step):
        p1 = cam.w2s(0, y)
        p2 = cam.w2s(WORLD_W, y)
        pygame.draw.line(surf, C_GRID, p1, p2, 1)


def draw_path(surf, cam, mpc):
    if mpc.spline_x is None:
        return
    # Ewaluacja CasADi interpolant na potrzeby rysowania
    s_arr = np.linspace(0, mpc.s_max, 500)
    px = mpc.spline_x(s_arr).full().flatten()
    py = mpc.spline_y(s_arr).full().flatten()
    
    pts = [cam.w2s(x, y) for x, y in zip(px, py)]
    if len(pts) >= 2:
        pygame.draw.lines(surf, C_PATH, False, pts, 2)


def draw_waypoints(surf, cam, waypoints, confirmed):
    color = C_WAYPOINT if confirmed else C_WAYPOINT_TMP
    for i, (wx, wy) in enumerate(waypoints):
        sx, sy = cam.w2s(wx, wy)
        pygame.draw.circle(surf, color, (sx, sy), 7)
        pygame.draw.circle(surf, (255, 255, 255), (sx, sy), 7, 1)
        if not confirmed and i > 0:
            prev = cam.w2s(*waypoints[i - 1])
            pygame.draw.line(surf, C_WAYPOINT_TMP, prev, (sx, sy), 1)


def draw_prediction(surf, cam, pred_path):
    if pred_path is None or len(pred_path) < 2:
        return
    pts = []
    for wx, wy in pred_path:
        if math.isfinite(wx) and math.isfinite(wy):
            sx, sy = cam.w2s(wx, wy)
            pts.append((sx, sy))
    if len(pts) >= 2:
        pygame.draw.lines(surf, C_PRED, False, pts, 2)
    for pt in pts[::3]:
        pygame.draw.circle(surf, C_PRED, pt, 3)


def draw_trail(surf, cam, trail):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        alpha = int(60 + 140 * i / len(trail))
        color = (*C_TRAIL[:3], alpha)
        p1 = cam.w2s(*trail[i - 1])
        p2 = cam.w2s(*trail[i])
        pygame.draw.line(surf, C_TRAIL, p1, p2, 2)


def draw_boat(surf, cam, state):
    x, y, psi, _, _, _ = state.x
    sx, sy = cam.w2s(x, y)

    # Trójkąt łodzi
    boat_len = 3.0
    boat_w = 1.5

    cos_p = math.cos(-psi)
    sin_p = math.sin(-psi)

    def rot(lx, ly):
        rx = lx * cos_p - ly * sin_p
        ry = lx * sin_p + ly * cos_p
        return rx, ry

    bow = rot(boat_len * 0.6, 0)
    stern_l = rot(-boat_len * 0.4, boat_w * 0.5)
    stern_r = rot(-boat_len * 0.4, -boat_w * 0.5)

    scale = cam.scale
    poly = [
        (sx + bow[0] * scale, sy + bow[1] * scale),
        (sx + stern_l[0] * scale, sy + stern_l[1] * scale),
        (sx + stern_r[0] * scale, sy + stern_r[1] * scale),
    ]
    pygame.draw.polygon(surf, C_BOAT, poly)
    pygame.draw.polygon(surf, (255, 255, 255), poly, 1)

    # Kierunek
    dir_len = boat_len * 1.2
    dsx = sx + math.cos(-psi) * dir_len * scale * 0.5
    dsy = sy + math.sin(-psi) * dir_len * scale * 0.5
    pygame.draw.line(surf, (255, 255, 255), (sx, sy), (int(dsx), int(dsy)), 2)

    # Wirtualny punkt celu (theta wyliczona z MPC)
    if state.target_point:
        tx, ty = cam.w2s(*state.target_point)
        pygame.draw.circle(surf, C_TARGET, (tx, ty), 6)
        pygame.draw.circle(surf, (255, 255, 255), (tx, ty), 6, 1)


def draw_ui(surf, state, font, font_small):
    panel_x = SCREEN_W - 280
    panel_w = 270

    # Tło
    panel_surf = pygame.Surface((panel_w, SCREEN_H), pygame.SRCALPHA)
    panel_surf.fill((20, 28, 48, 210))
    surf.blit(panel_surf, (panel_x, 0))
    pygame.draw.line(surf, C_UI_BORDER, (panel_x, 0), (panel_x, SCREEN_H), 1)

    y = 20
    def text(txt, color=C_TEXT, big=False):
        nonlocal y
        f = font if big else font_small
        surf.blit(f.render(txt, True, color), (panel_x + 16, y))
        y += f.get_linesize() + 4

    def separator():
        nonlocal y
        pygame.draw.line(surf, C_UI_BORDER, (panel_x + 8, y + 4),
                         (panel_x + panel_w - 8, y + 4), 1)
        y += 14

    text("NMPC Path Following", C_TEXT, big=True)
    separator()

    # Stan
    if not state.path_ready:
        text("Tryb: rysowanie ścieżki", (100, 220, 140))
        text(f"Waypointy: {len(state.waypoints)}", C_TEXT_DIM)
    elif state.path_done:
        text("Tryb: ukończono!", (80, 200, 255))
    elif state.paused:
        text("Tryb: PAUZA", (255, 180, 60))
    else:
        text("Tryb: symulacja", (80, 200, 255))

    separator()

    # Dane łodzi
    x, y, psi, u, v, r = state.x
    speed = math.hypot(u, v)
    text(f"x = {x:7.2f} m   y = {y:7.2f} m")
    text(f"ψ = {math.degrees(psi):7.1f}°")
    text(f"V = {speed:.2f} m/s")

    separator()

    # Sterowanie
    text(f"V_cmd  = {state.V_cmd:6.2f} V")
    text(f"α_cmd  = {state.alpha_cmd_deg:6.2f}°")
    text(f"krok   = {state.step}")

    if state.path_ready:
        mpc = state.mpc
        if mpc.s_max > 0.0:
            current_theta = mpc.last_sol_X[8, 1] if mpc.last_sol_X is not None else 0.0
            text(f"s (θ)  = {current_theta:6.1f} / {mpc.s_max:.1f} m")
            prog = (current_theta / mpc.s_max) * 100
            text(f"postęp = {prog:5.1f}%")

    separator()

    # Flagi
    col_pred = C_TEXT if state.show_pred else C_TEXT_DIM
    col_trail = C_TEXT if state.show_trail else C_TEXT_DIM
    text(f"[P] Predykcja MPC", col_pred)
    text(f"[T] Ślad łodzi", col_trail)

    separator()

    # Legenda
    text("Legenda", C_TEXT_DIM)
    items = [
        (C_PATH, "Ścieżka"),
        (C_PRED, "Predykcja MPC"),
        (C_BOAT, "Łódź"),
        (C_TARGET, "Wirtualny stan θ"),
    ]
    for color, label in items:
        pygame.draw.rect(surf, color, (panel_x + 16, y, 12, 12))
        surf.blit(font_small.render(label, True, C_TEXT_DIM), (panel_x + 36, y))
        y += 20

    separator()

    # Klawisze
    text("Klawiatura", C_TEXT_DIM)
    keys = [
        ("LPM", "dodaj waypoint"),
        ("PPM", "usuń ostatni"),
        ("ENTER", "start"),
        ("SPACE", "pauza"),
        ("R", "reset"),
        ("ESC", "wyjście"),
    ]
    for k, desc in keys:
        surf.blit(font_small.render(f"  {k:<7} {desc}", True, C_TEXT_DIM),
                 (panel_x + 16, y))
        y += 18


# ==================== SYMULACJA ====================
def simulation_step(state):
    """Jeden krok symulacji z nowym MPC bazującym na stanie wirtualnym."""
    mpc = state.mpc

    # 1. Sprawdź, czy dotarliśmy do końca ścieżki I czy łódź wyhamowała
    current_theta = mpc.last_sol_X[8, 1] if mpc.last_sol_X is not None else 0.0
    speed = float(np.hypot(state.x[3], state.x[4]))
    if mpc.s_max > 0 and (mpc.s_max - current_theta) < 1.0 and speed < 0.3:
        state.path_done = True
        return

    # Timeout safety — max 500 kroków
    if state.step > 500:
        print(f"[SIM] Timeout! step={state.step}, speed={speed:.2f}")
        state.path_done = True
        return

    # 2. Oblicz optymalne sterowanie NMPC
    try:
        V_cmd, alpha_cmd_deg = mpc.compute_control(state.x, state.alpha, state.w)
    except Exception as e:
        print(f"[BŁĄD] compute_control NMPC: {e}")
        return

    # 3. Zastosuj sterowanie do modelu obiektu
    V_in = float(np.clip(V_cmd, 0.0, 24.0))
    alpha_rad = float(np.radians(np.clip(alpha_cmd_deg, -45.0, 45.0)))

    state.alpha = float(boat_model.servo_step(state.alpha, alpha_rad))
    state.w = float(boat_model.motor_step(state.w, V_in))
    T_actual = boat_model.c_T * state.w * abs(state.w)

    state.x = np.asarray(
        boat_model.boat_step(state.x, T_actual, state.alpha)
    ).flatten()

    state.V_cmd = V_cmd
    state.alpha_cmd_deg = alpha_cmd_deg
    state.step += 1

    # 4. Pobierz przewidywaną trajektorię (X_sol[0:2, :])
    if mpc.predicted_path is not None and len(mpc.predicted_path) >= 2:
        state.pred_path = [(float(s[0]), float(s[1])) for s in mpc.predicted_path]
    else:
        state.pred_path = None

    # 5. Aktualizuj pozycję wirtualnego punktu celu na podstawie wyliczonej zmiennej theta
    if mpc.last_sol_X is not None and mpc.spline_x is not None:
        theta_0 = float(mpc.last_sol_X[8, 1])
        tx = float(mpc.spline_x(theta_0))
        ty = float(mpc.spline_y(theta_0))
        state.target_point = (tx, ty)

    # 6. Ślad łodzi
    state.trail.append((state.x[0], state.x[1]))
    if len(state.trail) > 2000:
        state.trail.pop(0)


# ==================== GŁÓWNA PĘTLA ====================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("NMPC True Path Following")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 15, bold=True)
    font_small = pygame.font.SysFont("monospace", 13)

    cam = Camera()
    state = SimState()

    sim_timer = 0.0
    running = True
    
    while running:
        dt_real = clock.tick(FPS) / 1000.0

        # === EVENTS ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    state = SimState()

                elif event.key == pygame.K_SPACE:
                    if state.path_ready:
                        state.paused = not state.paused

                elif event.key == pygame.K_p:
                    state.show_pred = not state.show_pred

                elif event.key == pygame.K_t:
                    state.show_trail = not state.show_trail

                elif event.key == pygame.K_RETURN:
                    if not state.path_ready and len(state.waypoints) >= 2:
                        wp = np.array(state.waypoints, dtype=float)
                        
                        # Set path w NMPC interpoluje nasze punkty po B-Spline
                        state.mpc.set_path(wp)

                        # Ustaw pozycję startową i kurs na podstawie dwóch pierwszych punktów
                        sx, sy = wp[0]
                        dx0 = wp[1, 0] - wp[0, 0]
                        dy0 = wp[1, 1] - wp[0, 1]
                        psi0 = float(np.arctan2(dy0, dx0))
                        state.x = np.array([sx, sy, psi0, 0.0, 0.0, 0.0])
                        state.path_ready = True

                        print(f"[SIM] Ścieżka zainicjalizowana. "
                              f"Długość = {state.mpc.s_max:.1f} m")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # LPM
                    if not state.path_ready and event.pos[0] < SCREEN_W - 280:
                        wx, wy = cam.s2w(*event.pos)
                        state.waypoints.append((wx, wy))

                elif event.button == 3:  # PPM
                    if not state.path_ready and state.waypoints:
                        state.waypoints.pop()

                elif event.button == 2:  # MMB drag
                    cam.start_drag(*event.pos)

                elif event.button == 4:  # scroll up
                    cam.zoom(1.1, *event.pos)

                elif event.button == 5:  # scroll down
                    cam.zoom(0.9, *event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    cam.end_drag()

            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[1]:
                    cam.update_drag(*event.pos)

        # === LOGIKA SYMULACJI ===
        if state.path_ready and not state.paused and not state.path_done:
            sim_timer += dt_real

            if sim_timer >= SIM_DT:
                sim_timer = 0.0
                simulation_step(state)

        # === RYSOWANIE ===
        screen.fill(C_BG)
        draw_grid(screen, cam)

        if state.path_ready:
            draw_path(screen, cam, state.mpc)

        draw_waypoints(screen, cam, state.waypoints, state.path_ready)

        if state.show_trail:
            draw_trail(screen, cam, state.trail)

        if state.show_pred and state.path_ready:
            draw_prediction(screen, cam, state.pred_path)

        if state.path_ready:
            draw_boat(screen, cam, state)

        draw_ui(screen, state, font, font_small)

        # Współrzędne myszy (podpowiedź)
        mx, my = pygame.mouse.get_pos()
        if mx < SCREEN_W - 280:
            wx, wy = cam.s2w(mx, my)
            coord_txt = font_small.render(f"({wx:.1f}, {wy:.1f}) m",
                                          True, C_TEXT_DIM)
            screen.blit(coord_txt, (mx + 14, my - 18))

        # Komunikat ukończenia
        if state.path_done:
            msg = font.render("Ścieżka ukończona! Naciśnij R aby reset.",
                              True, (80, 220, 100))
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2,
                              SCREEN_H // 2 - 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()