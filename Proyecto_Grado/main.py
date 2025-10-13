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

def is_thumb_inside_palm(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    p5, p18, p0, p2 = (hand_landmarks.landmark[i] for i in (5, 18, 0, 2))
    tx, ty = thumb_tip.x, thumb_tip.y
    min_x = min(p5.x, p18.x, p0.x, p2.x)
    max_x = max(p5.x, p18.x, p0.x, p2.x)
    min_y = min(p5.y, p18.y, p0.y, p2.y)
    max_y = max(p5.y, p18.y, p0.y, p2.y)
    return min_x <= tx <= max_x and min_y <= ty <= max_y

def is_shaka_gesture(hand_landmarks, width, height):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    pinky_tip = hand_landmarks.landmark[20]
    pinky_base = hand_landmarks.landmark[18]

    thumb_tip = (int(thumb_tip.x * width), int(thumb_tip.y * height))
    pinky_tip = (int(pinky_tip.x * width), int(pinky_tip.y * height))
    thumb_base = (int(thumb_base.x * width), int(thumb_base.y * height))
    pinky_base = (int(pinky_base.x * width), int(pinky_base.y * height))

    thumb_extended = thumb_tip[1] < thumb_base[1]
    pinky_extended = pinky_tip[1] < pinky_base[1]

    index_bent  = hand_landmarks.landmark[8].y  > hand_landmarks.landmark[5].y
    middle_bent = hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y
    ring_bent   = hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y
    return thumb_extended and pinky_extended and index_bent and middle_bent and ring_bent

# ---------------------- Constantes ----------------------
CANVAS_DIR = "lienzos"
os.makedirs(CANVAS_DIR, exist_ok=True)

CANVAS_H, CANVAS_W = 1200, 1800
VIEW_H, VIEW_W = 720, 1280
PROXIMITY_THRESHOLD = 35
HOVER_SECONDS = 1.0

COLOR_PRESETS = [
    (0, 0, 0),       # Negro
    (0, 0, 255),     # Rojo (BGR)
    (0, 255, 0),     # Verde
    (255, 0, 0),     # Azul
    (255, 0, 255),   # Morado
    (255, 255, 0),   # Cian
]
THICKNESS_PRESETS = [3, 5, 8, 12, 16]
SMOOTHING_PRESETS = [0.2, 0.5, 0.8]

# ---------------------- Diálogos PyQt (opcionales) ----------------------
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

    # Estado del lienzo y dibujo
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
    viewport_top_left = [0, 0]
    draw_mode_enabled = True
    draw_color_idx = 0
    draw_color = COLOR_PRESETS[draw_color_idx]
    thickness_idx = 1
    thickness = THICKNESS_PRESETS[thickness_idx]
    smoothing_idx = 1
    SMOOTHING_FACTOR = SMOOTHING_PRESETS[smoothing_idx]

    is_drawing = False
    start_point = None
    last_point = None
    last_hand_center = None

    # Menús
    menu_level = 0  # 0: principal, 1: submenú dibujo, 2: panel archivos
    menu_active = False

    main_menu_items = [
        ("Guardar", (0.88, 0.20, 0.98, 0.30), "save_panel"),
        ("Cargar",  (0.88, 0.35, 0.98, 0.45), "load_panel"),
        ("Dibujo",  (0.88, 0.50, 0.98, 0.60), "draw_menu"),
        ("Limpiar", (0.88, 0.70, 0.98, 0.80), "clear"),
    ]
    draw_menu_items = [
        ("Color",        (0.78, 0.18, 0.98, 0.30), "color"),
        ("Grosor",       (0.78, 0.33, 0.98, 0.45), "thick"),
        ("Estabilizador",(0.78, 0.48, 0.98, 0.60), "smooth"),
        ("Regresar",     (0.78, 0.72, 0.98, 0.84), "back"),
    ]

    # Panel de archivos (gestual)
    panel_mode = None   # "save" o "load"
    panel_page = 0
    PANEL_ITEMS_PER_PAGE = 8
    PANEL_X1, PANEL_Y1, PANEL_X2, PANEL_Y2 = 0.55, 0.10, 0.98, 0.90

    # Toasts
    toast_text = ""
    toast_expire = 0.0
    def show_toast(msg, duration=1.2):
        nonlocal toast_text, toast_expire
        toast_text = msg
        toast_expire = time.time() + duration

    def build_panel_items(stems, w, h):
        """Devuelve lista de (label, (x1n,y1n,x2n,y2n), key) para la página actual."""
        items = []

        # área panel
        px1, py1 = int(PANEL_X1*w), int(PANEL_Y1*h)
        px2, py2 = int(PANEL_X2*w), int(PANEL_Y2*h)
        panel_w = px2 - px1
        panel_h = py2 - py1

        # grid 2x4 para archivos (8 items)
        cols, rows = 2, 4
        cell_w = panel_w // cols
        cell_h = panel_h // (rows + 1)  # +1 fila para controles

        start = panel_page * PANEL_ITEMS_PER_PAGE
        page_stems = stems[start:start + PANEL_ITEMS_PER_PAGE]

        # Tiles de archivos
        for i, stem in enumerate(page_stems):
            r = i // cols
            c = i % cols
            x1 = px1 + c * cell_w + 10
            y1 = py1 + r * cell_h + 10
            x2 = x1 + cell_w - 20
            y2 = y1 + cell_h - 20
            items.append((stem, (x1/w, y1/h, x2/w, y2/h), f"file_{i}"))

        # En modo SAVE: tile "Nuevo (fecha-hora)" si hay hueco
        if panel_mode == "save" and len(page_stems) < PANEL_ITEMS_PER_PAGE:
            i = len(page_stems)
            r = i // cols
            c = i % cols
            x1 = px1 + c * cell_w + 10
            y1 = py1 + r * cell_h + 10
            x2 = x1 + cell_w - 20
            y2 = y1 + cell_h - 20
            items.append(("Nuevo (fecha-hora)", (x1/w, y1/h, x2/w, y2/h), "new_file"))

        # Controles en la fila inferior (3 columnas)
        ctrl_y1 = py1 + rows * cell_h + 10
        ctrl_y2 = ctrl_y1 + cell_h - 20
        ctrl_w = panel_w // 3

        # Regresar
        bx1 = px1 + 0 * ctrl_w + 10
        bx2 = bx1 + ctrl_w - 20
        items.append(("Regresar", (bx1/w, ctrl_y1/h, bx2/w, ctrl_y2/h), "panel_back"))

        # « Atrás
        pxl1 = px1 + 1 * ctrl_w + 10
        pxl2 = pxl1 + ctrl_w - 20
        items.append(("« Atrás", (pxl1/w, ctrl_y1/h, pxl2/w, ctrl_y2/h), "panel_prev"))

        # Siguiente »
        nxl1 = px1 + 2 * ctrl_w + 10
        nxl2 = nxl1 + ctrl_w - 20
        items.append(("Siguiente »", (nxl1/w, ctrl_y1/h, nxl2/w, ctrl_y2/h), "panel_next"))

        return items

    def draw_panel_background(frame, w, h, page_cur, page_tot):
        p1 = (int(PANEL_X1*w), int(PANEL_Y1*h))
        p2 = (int(PANEL_X2*w), int(PANEL_Y2*h))
        cv2.rectangle(frame, p1, p2, (32, 32, 32), -1)
        cv2.rectangle(frame, (p1[0]-2, p1[1]-2), (p2[0]+2, p2[1]+2), (0, 255, 255), 2)
        title = "GUARDAR (elige un archivo o 'Nuevo')" if panel_mode == "save" else "CARGAR (elige un archivo)"
        cv2.putText(frame, title, (p1[0]+10, p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # Indicador de página
        cv2.putText(frame, f"Pag {page_cur}/{page_tot}", (p2[0]-150, p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # hover states
    def make_hover_state(items):
        return {key: {"inside": False, "t0": 0.0} for _, _, key in items}

    def current_menu_items():
        return main_menu_items if menu_level == 0 else draw_menu_items if menu_level == 1 else []

    hover_state = make_hover_state(current_menu_items())
    panel_hover_state = {}

    # MediaPipe
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIEW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIEW_H)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            index_xy = None  # guardaremos el índice para dibujar el puntero al final (encima de los menús)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                # Shaka -> desplazar viewport
                if is_shaka_gesture(hand, w, h):
                    cx = int(hand.landmark[9].x * w)
                    cy = int(hand.landmark[9].y * h)
                    if last_hand_center is not None:
                        dx = cx - last_hand_center[0]
                        dy = cy - last_hand_center[1]
                        viewport_top_left[0] = int(np.clip(viewport_top_left[0] - dx, 0, CANVAS_W - VIEW_W))
                        viewport_top_left[1] = int(np.clip(viewport_top_left[1] - dy, 0, CANVAS_H - VIEW_H))
                    last_hand_center = (cx, cy)
                else:
                    last_hand_center = None

                # Estados de dedos
                idx_up  = hand.landmark[5].y  > hand.landmark[8].y
                mid_up  = hand.landmark[9].y  > hand.landmark[12].y
                ring_up = hand.landmark[13].y > hand.landmark[16].y
                pnk_up  = hand.landmark[17].y > hand.landmark[20].y
                th_up2  = hand.landmark[13].y > hand.landmark[4].y
                mano_completa = idx_up and mid_up and ring_up and pnk_up and th_up2

                # Índice/medio y distancia
                index_xy = (int(hand.landmark[8].x * w), int(hand.landmark[8].y * h))
                middle_xy= (int(hand.landmark[12].x* w), int(hand.landmark[12].y* h))
                dist_im  = calculate_distance(index_xy, middle_xy)

                # Borrador (no dentro del panel)
                if is_thumb_inside_palm(hand) and mano_completa and menu_level != 2:
                    adj = (index_xy[0] + viewport_top_left[0],
                           index_xy[1] + viewport_top_left[1])
                    cv2.circle(canvas, adj, 30, (255, 255, 255), -1)

                # Dibujo (deshabilitado dentro del panel)
                if (menu_level != 2) and draw_mode_enabled and mid_up and idx_up and (not ring_up) and (not pnk_up) and dist_im < PROXIMITY_THRESHOLD:
                    is_drawing = True
                else:
                    is_drawing = False
                    start_point = None
                    last_point = None

                # Trazo suavizado
                adj_index = (index_xy[0] + viewport_top_left[0],
                             index_xy[1] + viewport_top_left[1])
                if is_drawing:
                    if start_point is None:
                        start_point = adj_index
                    else:
                        sx = int(start_point[0] + (adj_index[0] - start_point[0]) * SMOOTHING_FACTOR)
                        sy = int(start_point[1] + (adj_index[1] - start_point[1]) * SMOOTHING_FACTOR)
                        smoothed = (sx, sy)
                        cv2.line(canvas, last_point if last_point else start_point, smoothed, draw_color, thickness)
                        start_point = smoothed
                    last_point = start_point

                # ---------- Activación/desactivación del menú (no se aplica dentro del panel) ----------
                if menu_level != 2:
                    if (hand.landmark[12].x > 0.9) and mano_completa:
                        if not menu_active:
                            menu_active = True
                            hover_state = make_hover_state(current_menu_items())
                    elif hand.landmark[12].x < 0.85:
                        if menu_active:
                            menu_active = False
                            hover_state = make_hover_state(current_menu_items())

                # ---------- Menús / Panel ----------
                pending_menu_level = None
                action_triggered = False

                # Menú lateral (niveles 0 y 1)
                if menu_active and menu_level in (0, 1):
                    items = current_menu_items()
                    if set(hover_state.keys()) != {k for _, _, k in items}:
                        hover_state = make_hover_state(items)

                    for label, (x1, y1, x2, y2), key in items:
                        p1 = (int(x1*w), int(y1*h))
                        p2 = (int(x2*w), int(y2*h))
                        cv2.rectangle(frame, p1, p2, (64, 41, 4), -1)
                        cv2.putText(frame, label, (p1[0]+6, p1[1]+int(0.6*(p2[1]-p1[1]))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225,250,250), 2)

                        # Indicadores del submenú Dibujo
                        if menu_level == 1:
                            if key == "color":
                                cv2.circle(frame, (p2[0]-22, p1[1]+18), 10, draw_color, -1)
                            elif key == "thick":
                                cv2.putText(frame, f"{thickness}px", (p2[0]-90, p1[1]+22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                            elif key == "smooth":
                                cv2.putText(frame, f"{SMOOTHING_FACTOR:.1f}", (p2[0]-70, p1[1]+22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                        inside = (x1 <= hand.landmark[12].x <= x2) and (y1 <= hand.landmark[12].y <= y2)
                        hs = hover_state[key]
                        now = time.time()
                        if inside:
                            if not hs["inside"]:
                                hs["inside"] = True
                                hs["t0"] = now
                                cv2.rectangle(frame, (p1[0]-2, p1[1]-2), (p2[0]+2, p2[1]+2), (0, 255, 255), 2)
                            else:
                                prog = min(1.0, (now - hs["t0"]) / HOVER_SECONDS)
                                bar_w = int((p2[0]-p1[0]) * prog)
                                cv2.rectangle(frame, (p1[0], p2[1]-6), (p1[0]+bar_w, p2[1]-2), (0, 255, 255), -1)

                                if (now - hs["t0"]) >= HOVER_SECONDS:
                                    if menu_level == 0:
                                        if key == "save_panel":
                                            panel_mode = "save"; pending_menu_level = 2
                                        elif key == "load_panel":
                                            panel_mode = "load"; pending_menu_level = 2
                                        elif key == "draw_menu":
                                            pending_menu_level = 1
                                        elif key == "clear":
                                            canvas.fill(255)
                                    else:  # nivel 1
                                        if key == "color":
                                            draw_color_idx = (draw_color_idx + 1) % len(COLOR_PRESETS)
                                            draw_color = COLOR_PRESETS[draw_color_idx]
                                        elif key == "thick":
                                            thickness_idx = (thickness_idx + 1) % len(THICKNESS_PRESETS)
                                            thickness = THICKNESS_PRESETS[thickness_idx]
                                        elif key == "smooth":
                                            smoothing_idx = (smoothing_idx + 1) % len(SMOOTHING_PRESETS)
                                            SMOOTHING_FACTOR = SMOOTHING_PRESETS[smoothing_idx]
                                        elif key == "back":
                                            pending_menu_level = 0

                                    hs["t0"] = now + 10
                                    action_triggered = True
                                    break
                        else:
                            hs["inside"] = False

                if action_triggered:
                    if pending_menu_level is not None:
                        if pending_menu_level == 2:
                            panel_page = 0
                            panel_hover_state.clear()
                        menu_level = pending_menu_level
                    hover_state = make_hover_state(current_menu_items())

                # Panel de archivos (nivel 2)
                panel_pending_menu_level = None
                panel_action_triggered = False

                if menu_level == 2:
                    # Stems y páginas
                    stems = list_canvas_stems()
                    total_pages = max(1, int(np.ceil(max(0, len(stems)) / PANEL_ITEMS_PER_PAGE)))
                    panel_page = int(np.clip(panel_page, 0, total_pages - 1))

                    # Fondo panel con indicador de página
                    draw_panel_background(frame, w, h, page_cur=panel_page+1, page_tot=total_pages)

                    # Items del panel
                    panel_items = build_panel_items(stems, w, h)
                    # Sync hover
                    keys = {k for *_, k in panel_items}
                    if set(panel_hover_state.keys()) != keys:
                        panel_hover_state = make_hover_state(panel_items)

                    # Dibujar + interacción (con cambio de nivel DIFERIDO)
                    for label, (x1n, y1n, x2n, y2n), key in panel_items:
                        x1, y1, x2, y2 = int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (83, 92, 0), -1)
                        cv2.putText(frame, label[:26] + ("…" if len(label) > 26 else ""), (x1+8, y1+int(0.6*(y2-y1))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225,250,250), 2)

                        inside = (hand.landmark[12].x >= x1n and hand.landmark[12].x <= x2n and
                                  hand.landmark[12].y >= y1n and hand.landmark[12].y <= y2n)
                        hs = panel_hover_state[key]
                        now = time.time()
                        if inside:
                            if not hs["inside"]:
                                hs["inside"] = True
                                hs["t0"] = now
                                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 2)
                            else:
                                prog = min(1.0, (now - hs["t0"]) / HOVER_SECONDS)
                                bar_w = int((x2-x1) * prog)
                                cv2.rectangle(frame, (x1, y2-6), (x1+bar_w, y2-2), (0, 255, 255), -1)

                                if (now - hs["t0"]) >= HOVER_SECONDS:
                                    if key == "panel_back":
                                        # DEFER: volver al menú principal como en "back" de dibujo
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
                                        stem = time.strftime("lienzo_%Y%m%d_%H%M%S")
                                        save_canvas_both(canvas, stem)
                                        cv2.putText(frame, f"Guardado: {stem}", (int(0.56*w), int(0.95*h)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
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
                                                    cv2.putText(frame, f"Cargado: {stem}", (int(0.56*w), int(0.95*h)),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                            else:  # save -> sobrescribe
                                                save_canvas_both(canvas, stem)
                                                cv2.putText(frame, f"Sobrescrito: {stem}", (int(0.56*w), int(0.95*h)),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                                    hs["t0"] = now + 10
                                    panel_action_triggered = True
                                    break
                        else:
                            hs["inside"] = False

                # ====== aplicar cambios DIFERIDOS del panel (fuera del for) ======
                if panel_action_triggered:
                    if panel_pending_menu_level is not None:
                        menu_level = panel_pending_menu_level
                        menu_active = True  # deja activo el menú principal, como pediste
                        hover_state = make_hover_state(current_menu_items())
                        panel_hover_state.clear()

            # Recorte de viewport
            x, y = viewport_top_left
            view = canvas[y:y+VIEW_H, x:x+VIEW_W]
            if view.shape[0] != VIEW_H or view.shape[1] != VIEW_W:
                pad = np.ones((VIEW_H, VIEW_W, 3), dtype=np.uint8) * 255
                pad[:view.shape[0], :view.shape[1]] = view
                view = pad

            # Combinado
            alpha = 0.5
            combined = frame.copy()
            roi = combined[0:VIEW_H, 0:VIEW_W]
            blended = cv2.addWeighted(roi, 1-alpha, view, alpha, 0)
            combined[0:VIEW_H, 0:VIEW_W] = blended

            # Overlay estado
            mode_text = f"Dibujo: {'ON' if draw_mode_enabled else 'OFF'} | Color:{COLOR_PRESETS[draw_color_idx]} | Grosor:{thickness}px | Suav:{SMOOTHING_FACTOR:.1f}"
            cv2.putText(combined, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if menu_active and menu_level in (0, 1):
                cv2.putText(combined, "MENU ACTIVO (mantén indice ~1s sobre la opción)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                if menu_level == 1:
                    cv2.putText(combined, "SUBMENU DIBUJO", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # --- Toast (si aplica) ---
            if toast_text and time.time() < toast_expire:
                txt = toast_text
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x_toast = (combined.shape[1] - tw) // 2
                y_toast = combined.shape[0] - 20
                cv2.rectangle(combined, (x_toast-10, y_toast-th-10), (x_toast+tw+10, y_toast+10), (0, 0, 0), -1)
                cv2.putText(combined, txt, (x_toast, y_toast), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # === PUNTERO ROJO ENCIMA DE LOS MENÚS ===
            if index_xy is not None:
                cv2.line(combined, (index_xy[0]-10, index_xy[1]), (index_xy[0]+10, index_xy[1]), (0,0,255), 2)
                cv2.line(combined, (index_xy[0], index_xy[1]-10), (index_xy[0], index_xy[1]+10), (0,0,255), 2)

            cv2.imshow("Hoja de Trabajo", combined)

            # Teclas rápidas (opcionales)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
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

            app.processEvents()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
