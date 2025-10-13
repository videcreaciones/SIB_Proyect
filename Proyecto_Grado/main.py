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
    menu_level = 0  # 0: principal, 1: submenú dibujo
    menu_active = False

    main_menu_items = [
        ("Guardar", (0.88, 0.20, 0.98, 0.30), "save"),
        ("Cargar",  (0.88, 0.35, 0.98, 0.45), "load"),
        ("Dibujo",  (0.88, 0.50, 0.98, 0.60), "draw_menu"),
        ("Limpiar", (0.88, 0.70, 0.98, 0.80), "clear"),
    ]
    draw_menu_items = [
        ("Color",        (0.78, 0.18, 0.98, 0.30), "color"),
        ("Grosor",       (0.78, 0.33, 0.98, 0.45), "thick"),
        ("Estabilizador",(0.78, 0.48, 0.98, 0.60), "smooth"),
        ("Regresar",     (0.78, 0.72, 0.98, 0.84), "back"),
    ]

    def current_menu_items():
        return main_menu_items if menu_level == 0 else draw_menu_items

    def make_hover_state(items):
        return {key: {"inside": False, "t0": 0.0} for _, _, key in items}

    hover_state = make_hover_state(current_menu_items())

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

                # Borrador
                if is_thumb_inside_palm(hand) and mano_completa:
                    adj = (index_xy[0] + viewport_top_left[0],
                           index_xy[1] + viewport_top_left[1])
                    cv2.circle(canvas, adj, 30, (255, 255, 255), -1)
                    cv2.circle(frame, index_xy, 30, (0, 0, 0), 2)

                # Dibujo
                if draw_mode_enabled and mid_up and idx_up and (not ring_up) and (not pnk_up) and dist_im < PROXIMITY_THRESHOLD:
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

                # Puntero
                cv2.line(frame, (index_xy[0]-10, index_xy[1]), (index_xy[0]+10, index_xy[1]), (0,0,255), 2)
                cv2.line(frame, (index_xy[0], index_xy[1]-10), (index_xy[0], index_xy[1]+10), (0,0,255), 2)

                # ---------- Activación/desactivación del menú ----------
                if (hand.landmark[12].x > 0.9) and mano_completa:
                    if not menu_active:
                        menu_active = True
                        hover_state = make_hover_state(current_menu_items())  # reset al entrar
                elif hand.landmark[12].x < 0.85:
                    if menu_active:
                        menu_active = False
                        hover_state = make_hover_state(current_menu_items())  # reset al salir

                # ---------- Render y lógica de menús ----------
                pending_menu_level = None
                action_triggered = False

                if menu_active:
                    items = current_menu_items()
                    item_keys = {key for _, _, key in items}
                    if set(hover_state.keys()) != item_keys:
                        hover_state = make_hover_state(items)  # sincroniza claves

                    for label, (x1, y1, x2, y2), key in items:
                        p1 = (int(x1*w), int(y1*h))
                        p2 = (int(x2*w), int(y2*h))

                        # fondo del botón
                        cv2.rectangle(frame, p1, p2, (64, 41, 4), -1)
                        cv2.putText(frame, label, (p1[0]+6, p1[1]+int(0.6*(p2[1]-p1[1]))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225,250,250), 2)

                        # Indicadores de estado en el submenú
                        if menu_level == 1:
                            if key == "color":
                                cv2.circle(frame, (p2[0]-22, p1[1]+18), 10, draw_color, -1)
                            elif key == "thick":
                                cv2.putText(frame, f"{thickness}px", (p2[0]-90, p1[1]+22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                            elif key == "smooth":
                                cv2.putText(frame, f"{SMOOTHING_FACTOR:.1f}", (p2[0]-70, p1[1]+22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                        # Hover con temporizador
                        inside = (x1 <= hand.landmark[12].x <= x2) and (y1 <= hand.landmark[12].y <= y2)
                        hs = hover_state.get(key)
                        if hs is None:
                            # Protección extra (no debería ocurrir tras la sincronización)
                            continue

                        now = time.time()
                        if inside:
                            if not hs["inside"]:
                                hs["inside"] = True
                                hs["t0"] = now
                                # resalta borde al entrar
                                cv2.rectangle(frame, (p1[0]-2, p1[1]-2), (p2[0]+2, p2[1]+2), (0, 255, 255), 2)
                            else:
                                # barra de progreso visual (simple)
                                prog = min(1.0, (now - hs["t0"]) / HOVER_SECONDS)
                                bar_w = int((p2[0]-p1[0]) * prog)
                                cv2.rectangle(frame, (p1[0], p2[1]-6), (p1[0]+bar_w, p2[1]-2), (0, 255, 255), -1)

                                if (now - hs["t0"]) >= HOVER_SECONDS:
                                    # --- Acciones por opción ---
                                    if menu_level == 0:
                                        if key == "save":
                                            SaveCanvasDialog(canvas).exec()
                                        elif key == "load":
                                            dlg = LoadCanvasDialog()
                                            if dlg.exec() and dlg.loaded is not None:
                                                loaded = dlg.loaded
                                                if loaded.shape[:2] != (CANVAS_H, CANVAS_W):
                                                    loaded = cv2.resize(loaded, (CANVAS_W, CANVAS_H))
                                                canvas[:] = loaded
                                        elif key == "draw_menu":
                                            pending_menu_level = 1  # ← diferir el cambio de menú
                                        elif key == "clear":
                                            canvas.fill(255)
                                    else:
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
                                            pending_menu_level = 0  # ← diferir el cambio de menú

                                    hs["t0"] = now + 10  # anti-repetición
                                    action_triggered = True
                                    break  # ← sal del for para aplicar cambios sin mezclar claves
                        else:
                            hs["inside"] = False

                # Si hubo acción que cambia de menú, aplícala ahora (fuera del for)
                if action_triggered:
                    if pending_menu_level is not None:
                        menu_level = pending_menu_level
                    hover_state = make_hover_state(current_menu_items())

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
            mode_text = f"Dibujo: {'ON' if draw_mode_enabled else 'OFF'} | Color:{draw_color} | Grosor:{thickness}px | Suav:{SMOOTHING_FACTOR:.1f}"
            cv2.putText(combined, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if menu_active:
                cv2.putText(combined, "MENU ACTIVO (mantén indice ~1s sobre la opción)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                if menu_level == 1:
                    cv2.putText(combined, "SUBMENU DIBUJO", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("Hoja de Trabajo", combined)

            # Teclas rápidas
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
    # Opcional: silenciar warnings molestos de protobuf (visual)
    # import warnings
    # warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
    main()
