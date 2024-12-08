from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt6.QtGui import QPixmap, QPainter, QImage, QFont, QColor
from PyQt6.QtCore import Qt
import cv2
import numpy as np
import os
import sys
from math import sqrt
import mediapipe as mp
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QListWidget, QMessageBox
)
import threading

def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_thumb_inside_palm(hand_landmarks):
    """
    Verifica si el pulgar está dentro de la región formada por los puntos 5, 18, 0 y 2.
    """
    thumb_tip = hand_landmarks.landmark[4]  # Punta del pulgar
    point_5 = hand_landmarks.landmark[5]
    point_18 = hand_landmarks.landmark[18]
    point_0 = hand_landmarks.landmark[0]
    point_2 = hand_landmarks.landmark[2]

    # Coordenadas en píxeles del pulgar
    thumb_x, thumb_y = thumb_tip.x, thumb_tip.y

    # Calcular límites de la región
    min_x = min(point_5.x, point_18.x, point_0.x, point_2.x)
    max_x = max(point_5.x, point_18.x, point_0.x, point_2.x)
    min_y = min(point_5.y, point_18.y, point_0.y, point_2.y)
    max_y = max(point_5.y, point_18.y, point_0.y, point_2.y)

    # Verificar si el pulgar está dentro de los límites
    return min_x <= thumb_x <= max_x and min_y <= thumb_y <= max_y


CANVAS_DIR = "lienzos"

# Crear el directorio si no existe
if not os.path.exists(CANVAS_DIR):
    os.makedirs(CANVAS_DIR)

canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gestión de Lienzos")

        # Layout principal
        self.layout = QVBoxLayout()

        # Botones de menú
        self.save_button = QPushButton("Guardar Lienzo")
        self.save_button.clicked.connect(self.save_canvas_window)
        self.layout.addWidget(self.save_button)

        self.load_button = QPushButton("Cargar Lienzo")
        self.load_button.clicked.connect(self.load_canvas_window)
        self.layout.addWidget(self.load_button)

        self.quit_button = QPushButton("Salir")
        self.quit_button.clicked.connect(self.close)
        self.layout.addWidget(self.quit_button)

        # Configuración del widget principal
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def save_canvas_window(self):
        self.save_window = SaveCanvasWindow()
        self.save_window.show()

    def load_canvas_window(self):
        self.load_window = LoadCanvasWindow()
        self.load_window.show()

class SaveCanvasWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guardar Lienzo")

        # Layout principal
        self.layout = QVBoxLayout()

        # Selección de formato
        self.format_label = QLabel("Seleccione el formato para guardar (png/npy):")
        self.layout.addWidget(self.format_label)

        self.png_button = QPushButton("PNG")
        self.png_button.clicked.connect(lambda: self.save_canvas("png"))
        self.layout.addWidget(self.png_button)

        self.npy_button = QPushButton("NPY")
        self.npy_button.clicked.connect(lambda: self.save_canvas("npy"))
        self.layout.addWidget(self.npy_button)

        # Entrada para nombre del archivo
        self.name_label = QLabel("Ingrese el nombre del lienzo:")
        self.layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.layout.addWidget(self.name_input)

        # Configuración del widget principal
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def save_canvas(self, format_choice):
        global canvas
        filename = self.name_input.text().strip()
        if not filename:
            QMessageBox.warning(self, "Error", "El nombre del archivo no puede estar vacío.")
            return

        filepath = os.path.join(CANVAS_DIR, f"{filename}.{format_choice}")
        if format_choice == "png":
            cv2.imwrite(filepath, canvas)
        elif format_choice == "npy":
            np.save(filepath, canvas)

        QMessageBox.information(self, "Éxito", f"Lienzo guardado como {filepath}")
        self.close()

class LoadCanvasWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cargar Lienzo")

        # Layout principal
        self.layout = QVBoxLayout()

        # Lista de lienzos disponibles
        self.list_label = QLabel("Seleccione el número del lienzo que desea cargar:")
        self.layout.addWidget(self.list_label)

        self.lienzo_list = QListWidget()
        self.layout.addWidget(self.lienzo_list)

        # Rellenar la lista
        self.load_canvas_list()

        # Botón de carga
        self.load_button = QPushButton("Cargar")
        self.load_button.clicked.connect(self.load_selected_canvas)
        self.layout.addWidget(self.load_button)

        # Configuración del widget principal
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def load_canvas_list(self):
        files = [f for f in os.listdir(CANVAS_DIR) if f.endswith(".npy") or f.endswith(".png")]
        self.lienzo_list.addItems(files)

    def load_selected_canvas(self):
        global canvas
        selected_item = self.lienzo_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Error", "Debe seleccionar un lienzo.")
            return

        filepath = os.path.join(CANVAS_DIR, selected_item.text())
        if filepath.endswith(".npy"):
            canvas = np.load(filepath)
        elif filepath.endswith(".png"):
            canvas = cv2.imread(filepath)

        QMessageBox.information(self, "Éxito", f"Lienzo cargado desde {filepath}")
        self.close()

# Función para mostrar el video y el lienzo
def show_video():
    global canvas
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        combined_view = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Hoja de Trabajo", combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Salir con 'ESC'
            break
        elif key == ord('c'):  # Limpiar lienzo
            canvas.fill(255)
        elif key == ord('s'):  # Guardar lienzo
            app = QApplication.instance() or QApplication(sys.argv)
            save_window = SaveCanvasWindow()
            save_window.show()
            app.exec()
        elif key == ord('l'):  # Cargar lienzo
            app = QApplication.instance() or QApplication(sys.argv)
            load_window = LoadCanvasWindow()
            load_window.show()
            app.exec()

    cap.release()
    cv2.destroyAllWindows()


def is_shaka_gesture(hand_landmarks, width, height):
    """Detecta si la mano hace la seña 🤙."""
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    pinky_tip = hand_landmarks.landmark[20]
    pinky_base = hand_landmarks.landmark[18]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]

    # Convertir a píxeles
    thumb_tip = (int(thumb_tip.x * width), int(thumb_tip.y * height))
    pinky_tip = (int(pinky_tip.x * width), int(pinky_tip.y * height))
    thumb_base = (int(thumb_base.x * width), int(thumb_base.y * height))
    pinky_base = (int(pinky_base.x * width), int(pinky_base.y * height))

    # Verificar si pulgar y meñique están extendidos
    thumb_extended = thumb_tip[1] < thumb_base[1]
    pinky_extended = pinky_tip[1] < pinky_base[1]

    # Verificar si los otros dedos están doblados
    index_bent = hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y
    middle_bent = hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y
    ring_bent = hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y

    return thumb_extended and pinky_extended and index_bent and middle_bent and ring_bent

# Parámetros
SMOOTHING_FACTOR = 0.5  # Estabilizador
CANVAS_SIZE = (1200, 1800, 3)  # Tamaño del lienzo (alto, ancho, canales)
VIEWPORT_SIZE = (480, 640)  # Tamaño del área visible
PROXIMITY_THRESHOLD = 35  # Umbral para los dedos índice y medio
SCROLL_STEP = 20  # Paso de desplazamiento

# Inicializar lienzo y variables
canvas = np.ones(CANVAS_SIZE, dtype=np.uint8) * 255  # Lienzo blanco
viewport_top_left = [0, 0]  # Coordenadas de la esquina superior izquierda del área visible
is_drawing = False
last_point = None
start_point = None
last_hand_center = None

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with (mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.5) as hands):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Procesar resultados de detección
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Detectar la seña 🤙
                if is_shaka_gesture(hand_landmarks, width, height):
                    hand_center_x = int(hand_landmarks.landmark[9].x * width)
                    hand_center_y = int(hand_landmarks.landmark[9].y * height)

                    if last_hand_center is not None:
                        dx = hand_center_x - last_hand_center[0]
                        dy = hand_center_y - last_hand_center[1]
                        viewport_top_left[0] = np.clip(viewport_top_left[0] - dx, 0, CANVAS_SIZE[1] - VIEWPORT_SIZE[1])
                        viewport_top_left[1] = np.clip(viewport_top_left[1] - dy, 0, CANVAS_SIZE[0] - VIEWPORT_SIZE[0])

                    last_hand_center = (hand_center_x, hand_center_y)
                else:
                    last_hand_center = None

                # Coordenadas del índice y medio
                index_coords = (int(hand_landmarks.landmark[8].x * width),
                                int(hand_landmarks.landmark[8].y * height))
                middle_coords = (int(hand_landmarks.landmark[12].x * width),
                                 int(hand_landmarks.landmark[12].y * height))

                # Calcular distancia entre índice y medio
                distance = calculate_distance(index_coords, middle_coords)


                """________ Verificar Posición De Los Dedos ________"""

                Ring_up = None
                Thumb_up = None
                Pinki_up = None
                Index_up = None
                Midle_up = None


                #Comprobar si el dedo gordo esta levantado o no

                if hand_landmarks.landmark[14].y > hand_landmarks.landmark[4].y:
                    Thumb_up = True
                    print("Gordo" + str (Thumb_up))
                else:
                    Thumb_up = False
                   # print(Thumb_up)

                #Comprobar si el dedo anular esta levantado o no

                if hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y:
                    Ring_up = True
                else:
                    Ring_up = False

                #Comprobar si el dedo meñique esta levantado o no

                if hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y:
                    Pinki_up = True
                else:
                    Pinki_up = False

                #comprobar si el medio esta levantado o no

                if hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y:
                    Midle_up = True
                else:
                    Midle_up = False

                #Comprobar si el dedo indice esta levantado o no

                if hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y:
                    Index_up = True
                else:
                    Index_up = False

                # Lógica de borrado
                thumb_inside = is_thumb_inside_palm(hand_landmarks)
                all_fingers_up = Index_up and Midle_up and Ring_up and Pinki_up

                if thumb_inside and all_fingers_up:
                    adjusted_index_coords = (
                        index_coords[0] + viewport_top_left[0],
                        index_coords[1] + viewport_top_left[1]
                    )
                    cv2.circle(canvas, adjusted_index_coords, 30, (255, 255, 255), -1)
                    cv2.circle(frame, index_coords, 30, (0, 0, 255), 2)  # Indicador visual

                # Activar dibujo si índice y medio están juntos
                if Midle_up == True and Index_up == True and Thumb_up == False and Ring_up == False and Pinki_up == False and distance < PROXIMITY_THRESHOLD:
                    is_drawing = True
                else:
                    is_drawing = False
                    last_point = None
                    start_point = None

                # Dibujar según el estado actual
                # Ajustar las coordenadas del índice al área visible actual
                adjusted_index_coords = (
                    index_coords[0] + viewport_top_left[0],
                    index_coords[1] + viewport_top_left[1]
                )

                # Dibujar según el estado actual
                if is_drawing:
                    if start_point is None:
                        start_point = adjusted_index_coords
                    else:
                        start_point = (
                            int(start_point[0] + (adjusted_index_coords[0] - start_point[0]) * SMOOTHING_FACTOR),
                            int(start_point[1] + (adjusted_index_coords[1] - start_point[1]) * SMOOTHING_FACTOR)
                        )
                        cv2.line(canvas, last_point if last_point else start_point, start_point, (0, 0, 0), 5)
                    last_point = start_point

                # Dibujar un puntero flotante
                cv2.line(frame, (index_coords[0] - 10, index_coords[1]),
                         (index_coords[0] + 10, index_coords[1]), (0, 0, 255), 2)
                cv2.line(frame, (index_coords[0], index_coords[1] - 10),
                         (index_coords[0], index_coords[1] + 10), (0, 0, 255), 2)


        # Extraer el área visible del lienzo
        x, y = viewport_top_left
        x_end = x + VIEWPORT_SIZE[1]
        y_end = y + VIEWPORT_SIZE[0]
        viewport = canvas[y:y_end, x:x_end]

        # Combinar la vista flotante con el área visible
        combined_view = frame.copy()
        combined_view[0:VIEWPORT_SIZE[0], 0:VIEWPORT_SIZE[1]] = cv2.addWeighted(
            frame[0:VIEWPORT_SIZE[0], 0:VIEWPORT_SIZE[1]],
            0.5,
            viewport,
            0.5,   # Modificar opacidad del lienzo
            0
        )

        # Mostrar la vista combinada en una ventana OpenCV
        cv2.imshow("Hoja de Trabajo", combined_view)

        # Manejo de teclas para interacciones
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Salir con 'ESC'
            break
        elif key == ord('c'):  # Limpiar lienzo
            canvas.fill(255)
        elif key == ord('s'):  # Guardar lienzo
            app = QApplication.instance() or QApplication(sys.argv)
            save_window = SaveCanvasWindow()
            save_window.show()
            app.exec()
        elif key == ord('l'):  # Cargar lienzo
            app = QApplication.instance() or QApplication(sys.argv)
            load_window = LoadCanvasWindow()
            load_window.show()
            app.exec()
