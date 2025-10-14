import os
import sys
import time
from math import sqrt

import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QWidget, QLineEdit, QListWidget,
    QMessageBox, QLabel, QPushButton
)

# ---------------------- Utilidades ----------------------
def calculate_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_shaka_gesture_np(lm):  # lm: np.array 21x3 normalizado
    y = lm[:,1]
    thumb_extended = y[4] < y[2]
    pinky_extended = y[20] < y[18]
    index_bent  = y[8]  > y[5]
    middle_bent = y[12] > y[9]
    ring_bent   = y[16] > y[13]
    return thumb_extended and pinky_extended and index_bent and middle_bent and ring_bent

def is_fist_closed_np(lm):
    """
    Puño cerrado (robusto a giros):
      - Índice/medio/anular/meñique flexionados: tip.y > pip.y
      - Pulgar cerca del centro de la palma.
    """
    y = lm[:,1]
    fingers_folded = (y[8]  > y[6]) and (y[12] > y[10]) and (y[16] > y[14]) and (y[20] > y[18])

    palm_idxs = [0,1,5,9,13,17]
    palm_center = lm[palm_idxs, :2].mean(axis=0)
    thumb_tip = lm[4, :2]
    dist_thumb_palm = float(np.linalg.norm(thumb_tip - palm_center))

    hand_size = float(
        np.linalg.norm(lm[0,:2] - lm[9,:2]) +
        np.linalg.norm(lm[5,:2] - lm[17,:2])
    )
    thresh = max(0.04, 0.25 * hand_size)  # umbral adaptativo con mínimo de seguridad
    thumb_folded = dist_thumb_palm < thresh

    return fingers_folded and thumb_folded

# Zoom virtual centrado (crop y resize al tamaño original)
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 3.0, 0.1
def virtual_zoom(frame, zoom: float):
    h, w = frame.shape[:2]
    zoom = float(np.clip(zoom, ZOOM_MIN, ZOOM_MAX))
    if zoom <= 1.0 + 1e-6:
        return frame
    cw = int(w / zoom); ch = int(h / zoom)
    cw -= (cw % 2); ch -= (ch % 2)
    x1 = (w - cw) // 2; y1 = (h - ch) // 2
    crop = frame[y1:y1+ch, x1:x1+cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

# ---------------------- Constantes ----------------------
CANVAS_DIR = "lienzos"
os.makedirs(CANVAS_DIR, exist_ok=True)

CANVAS_H, CANVAS_W = 1200, 1800
PROXIMITY_THRESHOLD = 35
HOVER_SECONDS = 1.5

COLOR_PRESETS = [
    (0, 0, 0), (0, 0, 255), (0, 255, 0),
    (255, 0, 0), (255, 0, 255), (255, 255, 0),
]
THICKNESS_PRESETS = [3, 5, 8, 12, 16]
SMOOTHING_PRESETS = [0.2, 0.5, 0.8]

LM_SMOOTH_ALPHA = 0.70  # suavizado landmarks
FIST_DEBOUNCE_S = 0.050  # 50 ms de puño continuo para activar borrado

# ---------------------- Diálogos PyQt ----------------------
class SaveCanvasDialog(QDialog):
    def __init__(self, canvas_ref):
        super().__init__()
        self.setWindowTitle("Guardar Lienzo")
        self.canvas_ref = canvas_ref
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Nombre del lienzo (sin extensión):"))
        self.name_input = QLineEdit()
        lay.addWidget(self.name_input)
        btn_row = QWidget()
        btn_lay = QVBoxLayout(btn_row)
        self.btn_png = QPushButton("Guardar como PNG")
        self.btn_npy = QPushButton("Guardar como NPY")
        btn_lay.addWidget(self.btn_png)
        btn_lay.addWidget(self.btn_npy)
        lay.addWidget(btn_row)
        self.btn_png.clicked.connect(lambda: self.do_save("png"))
        self.btn_npy.clicked.connect(lambda: self.do_save("npy"))

    def do_save(self, ext):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "El nombre no puede estar vacío.")
            return
        path = os.path.join(CANVAS_DIR, f"{name}.{ext}")
        if ext == "png":
            cv2.imwrite(path, self.canvas_ref)
        else:
            np.save(path, self.canvas_ref)
        QMessageBox.information(self, "Éxito", f"Lienzo guardado en:\n{path}")
        self.accept()

class LoadCanvasDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cargar Lienzo")
        self.loaded = None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Selecciona un lienzo (PNG/NPY):"))
        self.listw = QListWidget()
        lay.addWidget(self.listw)
        files = [f for f in os.listdir(CANVAS_DIR) if f.lower().endswith((".png", ".npy"))]
        files.sort()
        self.listw.addItems(files)
        btn = QPushButton("Cargar")
        btn.clicked.connect(self.load_selected)
        lay.addWidget(btn)

    def load_selected(self):
        item = self.listw.currentItem()
        if not item:
            QMessageBox.warning(self, "Error", "Debes seleccionar un archivo.")
            return
        path = os.path.join(CANVAS_DIR, item.text())
        if path.lower().endswith(".npy"):
            self.loaded = np.load(path)
        else:
            self.loaded = cv2.imread(path)
        if self.loaded is None:
            QMessageBox.critical(self, "Error", "No se pudo cargar el archivo.")
            return
        QMessageBox.information(self, "Éxito", f"Lienzo cargado de:\n{path}")
        self.accept()

# ---------------------- Utilidades de archivos ----------------------
def list_canvas_stems():
    files = [f for f in os.listdir(CANVAS_DIR) if f.lower().endswith((".png", ".npy"))]
    stems = set(os.path.splitext(f)[0] for f in files)
    stems = sorted(stems, key=str.lower)
    return stems

def save_canvas_both(canvas, stem):
    png_path = os.path.join(CANVAS_DIR, f"{stem}.png")
    npy_path = os.path.join(CANVAS_DIR, f"{stem}.npy")
    cv2.imwrite(png_path, canvas)
    np.save(npy_path, canvas)

def load_canvas_best(stem):
    png_path = os.path.join(CANVAS_DIR, f"{stem}.png")
    npy_path = os.path.join(CANVAS_DIR, f"{stem}.npy")
    if os.path.exists(png_path):
        arr = cv2.imread(png_path)
    elif os.path.exists(npy_path):
        arr = np.load(npy_path)
    else:
        return None
    return arr

# ---------------------- App principal ----------------------
def main():
    app = QApplication(sys.argv)

    # Tamaño real de la pantalla (para escalar salida)
    screen = app.primaryScreen()
    SCR_SIZE = screen.size()
    SCR_W, SCR_H = SCR_SIZE.width(), SCR_SIZE.height()

    cv2.namedWindow("Hoja de Trabajo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hoja de Trabajo", SCR_W, SCR_H)

    # Estado del lienzo y dibujo
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
    viewport_top_left = [0, 0]
    draw_mode_enabled = True
    draw_color_idx = 0
    thickness_idx = 1
    smoothing_idx = 1

    is_drawing = False
    start_point = None
    last_point = None
    last_hand_center = None

    # Mostrar esqueleto (mano) sobre fondo blanco
    show_skeleton = True  # toggle con 'm'

    # Zoom virtual (detección a distancia)
    zoom_factor = 2.0

    # Menús
    menu_level = 0  # 0: principal, 1: submenú dibujo, 2: panel archivos, 3: calibración
    menu_active = False

    main_menu_labels = [
        ("Guardar", "save_panel"),
        ("Cargar",  "load_panel"),
        ("Dibujo",  "draw_menu"),
        ("Calibrar","calib_menu"),
        ("Limpiar", "clear"),
    ]
    draw_menu_labels = [
        ("Color", "color"),
        ("Grosor", "thick"),
        ("Estabilizador", "smooth"),
        ("Regresar", "back"),
    ]
    calib_menu_labels = [
        ("+ Zoom",  "zoom_plus"),
        ("- Zoom",  "zoom_minus"),
        ("Reset",   "zoom_reset"),
        ("OK",      "zoom_ok"),
    ]

    panel_mode = None   # "save" o "load"
    panel_page = 0
    PANEL_ITEMS_PER_PAGE = 6  # 2x3 grandes
    PANEL_X1, PANEL_Y1, PANEL_X2, PANEL_Y2 = 0.52, 0.10, 0.98, 0.90

    toast_text = ""
    toast_expire = 0.0
    def show_toast(msg, duration=1.2):
        nonlocal toast_text, toast_expire
        toast_text = msg
        toast_expire = time.time() + duration

    # hover states
    def make_hover_state(items):
        return {key: {"inside": False, "t0": 0.0} for _, _, key in items}

    def build_menus_with_scale(w, h):
        UI_SCALE = max(0.8, min(1.8, min(w/640, h/480)))
        btn_width = 0.15 + 0.05*(UI_SCALE-1.0)
        btn_width = min(0.25, max(btn_width, 0.17))
        x2 = 0.99
        x1 = x2 - btn_width

        top = 0.14
        btn_h = 0.12 + 0.04*(UI_SCALE-1.0)
        btn_h = min(0.22, max(btn_h, 0.12))
        gap  = 0.035 + 0.01*(UI_SCALE-1.0)
        gap  = min(0.06, max(gap, 0.03))

        # principal
        main_items = []
        y = top
        for label, key in main_menu_labels:
            main_items.append((label, (x1, y, x2, y+btn_h), key))
            y += btn_h + gap

        # dibujo
        y = 0.18
        draw_items = []
        for label, key in draw_menu_labels:
            draw_items.append((label, (x1-0.08, y, x2, y+btn_h), key))
            y += btn_h + gap

        # calibración
        y = 0.20
        calib_items = []
        for label, key in calib_menu_labels:
            calib_items.append((label, (x1-0.10, y, x2, y+btn_h), key))
            y += btn_h + gap

        return main_items, draw_items, calib_items, UI_SCALE

    def current_menu_items(w, h):
        main_items, draw_items, calib_items, _ = build_menus_with_scale(w, h)
        if menu_level == 0: return main_items
        if menu_level == 1: return draw_items
        if menu_level == 3: return calib_items
        return []

    hover_state = {}
    panel_hover_state = {}

    def build_panel_items(stems, w, h, UI_SCALE):
        items = []
        px1, py1 = int(PANEL_X1*w), int(PANEL_Y1*h)
        px2, py2 = int(PANEL_X2*w), int(PANEL_Y2*h)
        panel_w = px2 - px1
        panel_h = py2 - py1

        cols, rows = 2, 3
        cell_w = panel_w // cols
        cell_h = panel_h // (rows + 1)

        pad = int(14 * UI_SCALE)

        start = panel_page * PANEL_ITEMS_PER_PAGE
        page_stems = stems[start:start + PANEL_ITEMS_PER_PAGE]

        for i, stem in enumerate(page_stems):
            r = i // cols
            c = i % cols
            x1 = px1 + c * cell_w + pad
            y1 = py1 + r * cell_h + pad
            x2 = x1 + cell_w - 2*pad
            y2 = y1 + cell_h - 2*pad
            items.append((stem, (x1/w, y1/h, x2/w, y2/h), f"file_{i}"))

        if panel_mode == "save" and len(page_stems) < PANEL_ITEMS_PER_PAGE:
            i = len(page_stems)
            r = i // cols
            c = i % cols
            x1 = px1 + c * cell_w + pad
            y1 = py1 + r * cell_h + pad
            x2 = x1 + cell_w - 2*pad
            y2 = y1 + cell_h - 2*pad
            items.append(("Nuevo (fecha-hora)", (x1/w, y1/h, x2/w, y2/h), "new_file"))

        ctrl_y1 = py1 + rows * cell_h + pad
        ctrl_y2 = ctrl_y1 + cell_h - 2*pad
        ctrl_w = panel_w // 3
        pad2 = int(10 * UI_SCALE)

        bx1 = px1 + 0 * ctrl_w + pad2
        bx2 = bx1 + ctrl_w - 2*pad2
        items.append(("Menu", (bx1/w, ctrl_y1/h, bx2/w, ctrl_y2/h), "panel_back"))

        pxl1 = px1 + 1 * ctrl_w + pad2
        pxl2 = pxl1 + ctrl_w - 2*pad2
        items.append(("Atras", (pxl1/w, ctrl_y1/h, pxl2/w, ctrl_y2/h), "panel_prev"))

        nxl1 = px1 + 2 * ctrl_w + pad2
        nxl2 = nxl1 + ctrl_w - 2*pad2
        items.append(("Siguiente", (nxl1/w, ctrl_y1/h, nxl2/w, ctrl_y2/h), "panel_next"))

        return items

    def draw_panel_background(frame, w, h, page_cur, page_tot, UI_SCALE):
        p1 = (int(PANEL_X1*w), int(PANEL_Y1*h))
        p2 = (int(PANEL_X2*w), int(PANEL_Y2*h))
        cv2.rectangle(frame, p1, p2, (32, 32, 32), -1)
        cv2.rectangle(frame, (p1[0]-2, p1[1]-2), (p2[0]+2, p2[1]+2), (0, 255, 255), 2)
        fs = 0.7 * UI_SCALE
        th = max(1, int(2 * UI_SCALE))
        title = "GUARDAR (elige un archivo o 'Nuevo')" if panel_mode == "save" else "CARGAR (elige un archivo)"
        cv2.putText(frame, title, (p1[0]+10, max(20, p1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,255), th)
        cv2.putText(frame, f"Pag {page_cur}/{page_tot}", (p2[0]-160, max(20, p1[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,255), th)

    # MediaPipe + Cámara
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    mp_model_complexity = 1
    mp_min_det = 0.75
    mp_min_trk = 0.75

    with mp_hands.Hands(
        model_complexity=mp_model_complexity,
        max_num_hands=1,
        min_detection_confidence=mp_min_det,
        min_tracking_confidence=mp_min_trk
    ) as hands:

        ema_landmarks = None  # 21x3 np.array
        fist_t0 = None        # tiempo de inicio del puño

        while True:
            ok, raw = cap.read()
            if not ok:
                break

            # Sin flip; aplicar zoom virtual
            frame = virtual_zoom(raw, zoom_factor)
            h, w = frame.shape[:2]
            UI_SCALE = max(0.8, min(1.8, min(w/640, h/480)))

            # Fondo blanco
            surface = np.full((h, w, 3), 255, dtype=np.uint8)

            # Menús para este tamaño
            main_menu_items, draw_menu_items, calib_menu_items, _ = build_menus_with_scale(w, h)
            def _current_items(w_, h_):
                if menu_level == 0: return main_menu_items
                if menu_level == 1: return draw_menu_items
                if menu_level == 3: return calib_menu_items
                return []
            desired_keys = {k for _, _, k in _current_items(w, h)}
            if (not hover_state) or (set(hover_state.keys()) != desired_keys):
                hover_state = make_hover_state(_current_items(w, h))

            # === MediaPipe ===
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index_xy = None

            # indicadores del borrador
            show_eraser = False
            eraser_center = (0, 0)
            eraser_radius = max(10, int(30 * UI_SCALE))

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                # pasar a np.array normalizado y suavizar
                cur = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
                if ema_landmarks is None:
                    ema_landmarks = cur.copy()
                else:
                    ema_landmarks = LM_SMOOTH_ALPHA * ema_landmarks + (1.0 - LM_SMOOTH_ALPHA) * cur
                lm = ema_landmarks  # suavizado

                # Shaka con lm suavizado
                if is_shaka_gesture_np(lm):
                    cx = int(lm[9,0] * w); cy = int(lm[9,1] * h)
                    if last_hand_center is not None:
                        dx = cx - last_hand_center[0]
                        dy = cy - last_hand_center[1]
                        viewport_top_left[0] = int(np.clip(viewport_top_left[0] - dx, 0, CANVAS_W - w))
                        viewport_top_left[1] = int(np.clip(viewport_top_left[1] - dy, 0, CANVAS_H - h))
                    last_hand_center = (cx, cy)
                else:
                    last_hand_center = None

                # Estados de dedos con lm suavizado (para dibujo/menú)
                y = lm[:,1]
                idx_up  = y[5]  > y[8]
                mid_up  = y[9]  > y[12]
                ring_up = y[13] > y[16]
                pnk_up  = y[17] > y[20]
                th_up2  = y[13] > y[4]
                mano_completa = idx_up and mid_up and ring_up and pnk_up and th_up2

                # Coordenadas de índice y medio
                index_xy  = (int(lm[8,0]  * w), int(lm[8,1]  * h))
                middle_xy = (int(lm[12,0] * w), int(lm[12,1] * h))
                dist_im   = calculate_distance(index_xy, middle_xy)

                # --------- BORRADOR: PUÑO CERRADO con debounce 50 ms ---------
                now_ts = time.time()
                if is_fist_closed_np(lm) and (menu_level != 2):
                    if fist_t0 is None:
                        fist_t0 = now_ts
                    if (now_ts - fist_t0) >= FIST_DEBOUNCE_S:
                        adj = (index_xy[0] + viewport_top_left[0],
                               index_xy[1] + viewport_top_left[1])
                        cv2.circle(canvas, adj, eraser_radius, (255, 255, 255), -1)
                        show_eraser = True
                        eraser_center = index_xy
                else:
                    fist_t0 = None  # resetea si deja de estar en puño

                # Dibujo (no en panel)
                if (menu_level != 2) and draw_mode_enabled and mid_up and idx_up and (not ring_up) and (not pnk_up) and dist_im < PROXIMITY_THRESHOLD*UI_SCALE:
                    is_drawing = True
                else:
                    is_drawing = False
                    start_point = None
                    last_point = None

                adj_index = (index_xy[0] + viewport_top_left[0],
                             index_xy[1] + viewport_top_left[1])
                if is_drawing:
                    if start_point is None:
                        start_point = adj_index
                    else:
                        smooth = SMOOTHING_PRESETS[smoothing_idx]
                        sx = int(start_point[0] + (adj_index[0] - start_point[0]) * smooth)
                        sy = int(start_point[1] + (adj_index[1] - start_point[1]) * smooth)
                        smoothed = (sx, sy)
                        cv2.line(canvas, last_point if last_point else start_point, smoothed,
                                 COLOR_PRESETS[draw_color_idx],
                                 max(1, int(THICKNESS_PRESETS[thickness_idx]*UI_SCALE)))
                        start_point = smoothed
                    last_point = start_point

                # Menú ON/OFF (lateral derecho) con lm suavizado
                if menu_level != 2:
                    if (lm[12,0] > 0.88) and mano_completa:
                        if not menu_active:
                            menu_active = True
                            hover_state = make_hover_state(_current_items(w, h))
                    elif lm[12,0] < 0.82:
                        if menu_active:
                            menu_active = False
                            hover_state = make_hover_state(_current_items(w, h))

                # ---------- Menús ----------
                pending_menu_level = None
                action_triggered = False

                def draw_buttons(items, level):
                    nonlocal hover_state, panel_mode, draw_color_idx, thickness_idx, smoothing_idx, zoom_factor, pending_menu_level, action_triggered
                    # ^^^ Mover nonlocal ARRIBA evita el SyntaxError

                    if set(hover_state.keys()) != {k for _, _, k in items}:
                        for k in list(hover_state.keys()):
                            if k not in {kk for _,_,kk in items}:
                                hover_state.pop(k, None)
                        for _,_,k in items:
                            hover_state.setdefault(k, {"inside": False, "t0": 0.0})

                    fs = 0.6 * UI_SCALE
                    th_text = max(1, int(2 * UI_SCALE))
                    th_border = max(2, int(3 * UI_SCALE))
                    hover_bar_h = max(4, int(6 * UI_SCALE))
                    inflate = int(10 * UI_SCALE)

                    for label, (x1, y1, x2, y2), key in items:
                        p1 = (int(x1*w), int(y1*h)); p2 = (int(x2*w), int(y2*h))
                        dp1 = (p1[0]-inflate, p1[1]-inflate)
                        dp2 = (p2[0]+inflate, p2[1]+inflate)
                        cv2.rectangle(surface, dp1, dp2, (64, 41, 4) if level!=3 else (40,40,40), -1)
                        cv2.putText(surface, label, (dp1[0]+int(12*UI_SCALE), dp1[1]+int(0.65*(dp2[1]-dp1[1]))),
                                    cv2.FONT_HERSHEY_SIMPLEX, fs, (225,250,250), th_text)

                        if level == 1:
                            if key == "color":
                                cv2.circle(surface, (dp2[0]-int(26*UI_SCALE), dp1[1]+int(22*UI_SCALE)),
                                           max(10, int(12*UI_SCALE)), COLOR_PRESETS[draw_color_idx], -1)
                            elif key == "thick":
                                cv2.putText(surface, f"{THICKNESS_PRESETS[thickness_idx]}px",
                                            (dp2[0]-int(120*UI_SCALE), dp1[1]+int(26*UI_SCALE)),
                                            cv2.FONT_HERSHEY_SIMPLEX, fs*0.9, (255,255,255), th_text)
                            elif key == "smooth":
                                cv2.putText(surface, f"{SMOOTHING_PRESETS[smoothing_idx]:.1f}",
                                            (dp2[0]-int(80*UI_SCALE), dp1[1]+int(26*UI_SCALE)),
                                            cv2.FONT_HERSHEY_SIMPLEX, fs*0.9, (255,255,255), th_text)
                        elif level == 3 and key in ("zoom_plus","zoom_minus","zoom_reset","zoom_ok"):
                            cv2.putText(surface, f"Zoom: {zoom_factor:.1f}x",
                                        (int(0.06*w), int(0.10*h)),
                                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,255), th_text)

                        inside = (lm[12,0]*w >= dp1[0] and lm[12,0]*w <= dp2[0] and
                                  lm[12,1]*h >= dp1[1] and lm[12,1]*h <= dp2[1])
                        hs = hover_state.setdefault(key, {"inside": False, "t0": 0.0})
                        now = time.time()
                        if inside:
                            if not hs["inside"]:
                                hs["inside"] = True
                                hs["t0"] = now
                                cv2.rectangle(surface, (dp1[0]-2, dp1[1]-2), (dp2[0]+2, dp2[1]+2), (0, 255, 255), th_border)
                            else:
                                prog = min(1.0, (now - hs["t0"]) / HOVER_SECONDS)
                                bar_w = int((dp2[0]-dp1[0]) * prog)
                                cv2.rectangle(surface, (dp1[0], dp2[1]-hover_bar_h),
                                              (dp1[0]+bar_w, dp2[1]-2), (0, 255, 255), -1)
                                if (now - hs["t0"]) >= HOVER_SECONDS:
                                    if level == 0:
                                        if key == "save_panel":
                                            panel_mode = "save"; pending_menu_level = 2
                                        elif key == "load_panel":
                                            panel_mode = "load"; pending_menu_level = 2
                                        elif key == "draw_menu":
                                            pending_menu_level = 1
                                        elif key == "calib_menu":
                                            pending_menu_level = 3
                                        elif key == "clear":
                                            canvas.fill(255)
                                    elif level == 1:
                                        if key == "color":
                                            draw_color_idx = (draw_color_idx + 1) % len(COLOR_PRESETS)
                                        elif key == "thick":
                                            thickness_idx = (thickness_idx + 1) % len(THICKNESS_PRESETS)
                                        elif key == "smooth":
                                            smoothing_idx = (smoothing_idx + 1) % len(SMOOTHING_PRESETS)
                                        elif key == "back":
                                            pending_menu_level = 0
                                    elif level == 3:
                                        if key == "zoom_plus":
                                            zoom_factor = float(np.clip(zoom_factor + ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
                                        elif key == "zoom_minus":
                                            zoom_factor = float(np.clip(zoom_factor - ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
                                        elif key == "zoom_reset":
                                            zoom_factor = 1.0
                                        elif key == "zoom_ok":
                                            pending_menu_level = 0
                                    hs["t0"] = now + 10
                                    action_triggered = True
                                    break
                        else:
                            hs["inside"] = False

                if menu_active and menu_level in (0,1,3):
                    draw_buttons(_current_items(w, h), menu_level)

                if action_triggered:
                    if pending_menu_level is not None:
                        if pending_menu_level == 2:
                            panel_page = 0
                            panel_hover_state.clear()
                        menu_level = pending_menu_level
                    hover_state = make_hover_state(_current_items(w, h))

                # ----- Panel de archivos -----
                panel_pending_menu_level = None
                panel_action_triggered = False

                if menu_level == 2:
                    stems = list_canvas_stems()
                    total_pages = max(1, int(np.ceil(max(0, len(stems)) / PANEL_ITEMS_PER_PAGE)))
                    panel_page = int(np.clip(panel_page, 0, total_pages - 1))

                    draw_panel_background(surface, w, h, page_cur=panel_page+1, page_tot=total_pages, UI_SCALE=UI_SCALE)

                    panel_items = build_panel_items(stems, w, h, UI_SCALE)
                    keys = {k for *_, k in panel_items}
                    if set(panel_hover_state.keys()) != keys:
                        panel_hover_state = make_hover_state(panel_items)

                    fs = 0.5 * UI_SCALE
                    th_text = max(1, int(2 * UI_SCALE))
                    th_border = max(2, int(3 * UI_SCALE))
                    hover_bar_h = max(4, int(6 * UI_SCALE))
                    inflate = int(8 * UI_SCALE)

                    for label, (x1n, y1n, x2n, y2n), key in panel_items:
                        x1, y1, x2, y2 = int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)
                        dp1 = (x1 - inflate, y1 - inflate)
                        dp2 = (x2 + inflate, y2 + inflate)
                        cv2.rectangle(surface, dp1, dp2, (83, 92, 0), -1)
                        shown = label[:28] + ("…" if len(label) > 28 else "")
                        cv2.putText(surface, shown, (dp1[0]+int(12*UI_SCALE), dp1[1]+int(0.6*(dp2[1]-dp1[1]))),
                                    cv2.FONT_HERSHEY_SIMPLEX, fs, (225,250,250), th_text)

                        inside = (lm[12,0]*w >= dp1[0] and lm[12,0]*w <= dp2[0] and
                                  lm[12,1]*h >= dp1[1] and lm[12,1]*h <= dp2[1])
                        hs = panel_hover_state[key]
                        now = time.time()
                        if inside:
                            if not hs["inside"]:
                                hs["inside"] = True
                                hs["t0"] = now
                                cv2.rectangle(surface, (dp1[0]-2, dp1[1]-2), (dp2[0]+2, dp2[1]+2), (0,255,255), th_border)
                            else:
                                prog = min(1.0, (now - hs["t0"]) / HOVER_SECONDS)
                                bar_w = int((dp2[0]-dp1[0]) * prog)
                                cv2.rectangle(surface, (dp1[0], dp2[1]-hover_bar_h),
                                              (dp1[0]+bar_w, dp2[1]-2), (0,255,255), -1)
                                if (now - hs["t0"]) >= HOVER_SECONDS:
                                    if key == "panel_back":
                                        panel_pending_menu_level = 0
                                    elif key == "panel_prev":
                                        if panel_page > 0:
                                            panel_page = max(0, panel_page - 1)
                                            panel_hover_state.clear()
                                        else:
                                            show_toast("No hay página anterior", 1.2)
                                    elif key == "panel_next":
                                        if panel_page < (total_pages - 1):
                                            panel_page = min(total_pages - 1, panel_page + 1)
                                            panel_hover_state.clear()
                                        else:
                                            show_toast("No hay página siguiente", 1.2)
                                    elif key == "new_file" and panel_mode == "save":
                                        stem = time.strftime("LZ_%m%d_%H%M")
                                        save_canvas_both(canvas, stem)
                                        cv2.putText(surface, f"Guardado: {stem}", (int(0.56*w), int(0.95*h)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th_text)
                                    elif key.startswith("file_"):
                                        idx_local = int(key.split("_")[1])
                                        idx_global = panel_page * PANEL_ITEMS_PER_PAGE + idx_local
                                        if idx_global < len(stems):
                                            stem = stems[idx_global]
                                            if panel_mode == "load":
                                                loaded = load_canvas_best(stem)
                                                if loaded is not None:
                                                    if loaded.shape[:2] != (CANVAS_H, CANVAS_W):
                                                        loaded = cv2.resize(loaded, (CANVAS_W, CANVAS_H))
                                                    canvas[:] = loaded
                                                    cv2.putText(surface, f"Cargado: {stem}", (int(0.56*w), int(0.95*h)),
                                                                cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th_text)
                                            else:
                                                save_canvas_both(canvas, stem)
                                                cv2.putText(surface, f"Sobrescrito: {stem}", (int(0.56*w), int(0.95*h)),
                                                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th_text)
                                    hs["t0"] = now + 10
                                    panel_action_triggered = True
                                    break
                        else:
                            hs["inside"] = False

                if 'panel_action_triggered' in locals() and panel_action_triggered:
                    if 'panel_pending_menu_level' in locals() and panel_pending_menu_level is not None:
                        menu_level = panel_pending_menu_level
                        menu_active = True
                        hover_state = make_hover_state(_current_items(w, h))
                        panel_hover_state.clear()

            # Recorte del lienzo al tamaño del frame
            x, y = viewport_top_left
            view = canvas[y:y+h, x:x+w]
            if view.shape[0] != h or view.shape[1] != w:
                pad = np.ones((h, w, 3), dtype=np.uint8) * 255
                pad[:view.shape[0], :view.shape[1]] = view
                view = pad

            # Mezcla superficie (blanca) + lienzo
            alpha = 0.5
            combined = surface.copy()
            roi = combined[0:h, 0:w]
            blended = cv2.addWeighted(roi, 1 - alpha, view, alpha, 0)
            combined[0:h, 0:w] = blended

            # Esqueleto (visual)
            if show_skeleton and results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        combined,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )

            # Indicador de borrador
            if show_eraser:
                cv2.circle(combined, eraser_center, eraser_radius, (0, 0, 0), 2)

            # Puntero rojo
            if index_xy is not None:
                cv2.line(combined, (index_xy[0]-10, index_xy[1]), (index_xy[0]+10, index_xy[1]), (0,0,255), 2)
                cv2.line(combined, (index_xy[0], index_xy[1]-10), (index_xy[0], index_xy[1]+10), (0,0,255), 2)

            # HUD
            fs_hud = 0.7 * UI_SCALE
            th_hud = max(1, int(2 * UI_SCALE))
            hud = f"Dibujo:{'ON' if draw_mode_enabled else 'OFF'} | Color:{COLOR_PRESETS[draw_color_idx]} | Grosor:{THICKNESS_PRESETS[thickness_idx]}px | Suav:{SMOOTHING_PRESETS[smoothing_idx]:.1f} | Mano:{'ON' if show_skeleton else 'OFF'} | Zoom:{zoom_factor:.1f}x"
            cv2.putText(combined, hud, (10, int(28*UI_SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_hud, (0,0,0), th_hud)

            # Escalar a pantalla (letterbox si hace falta)
            src_h, src_w = combined.shape[:2]
            scale = min(SCR_W / src_w, SCR_H / src_h)
            disp_w = int(src_w * scale)
            disp_h = int(src_h * scale)
            resized = cv2.resize(combined, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            screen_img = np.zeros((SCR_H, SCR_W, 3), dtype=np.uint8)
            off_x = (SCR_W - disp_w) // 2
            off_y = (SCR_H - disp_h) // 2
            screen_img[off_y:off_y+disp_h, off_x:off_x+disp_w] = resized

            cv2.imshow("Hoja de Trabajo", screen_img)

            # Teclas rápidas
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord('m'):
                show_skeleton = not show_skeleton
            elif k == ord('c'):
                canvas.fill(255)
            elif k == ord('s'):
                SaveCanvasDialog(canvas).exec()
            elif k == ord('l'):
                dlg = LoadCanvasDialog()
                if dlg.exec() and dlg.loaded is not None:
                    loaded = dlg.loaded
                    if loaded.shape[:2] != (CANVAS_H, CANVAS_W):
                        loaded = cv2.resize(loaded, (CANVAS_W, CANVAS_H))
                    canvas[:] = loaded
            elif k == ord('d'):
                draw_mode_enabled = not draw_mode_enabled
            elif k == ord(']'):
                zoom_factor = float(np.clip(zoom_factor + ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
            elif k == ord('['):
                zoom_factor = float(np.clip(zoom_factor - ZOOM_STEP, ZOOM_MIN, ZOOM_MAX))
            elif k == ord('='):
                zoom_factor = 1.0

            app.processEvents()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
