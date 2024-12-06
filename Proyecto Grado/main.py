from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt6.QtGui import QPixmap, QPainter, QImage, QFont, QColor
from PyQt6.QtCore import Qt
import cv2
import numpy as np
import os
import sys
from math import sqrt
import mediapipe as mp

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


def save_canvas(canvas):
    """Guarda el lienzo actual en formato .png o .npy con un nombre personalizado."""
    format_choice = input("Seleccione el formato para guardar (png/npy): ").strip().lower()
    if format_choice not in ["png", "npy"]:
        print("Formato no válido. Se usará .npy por defecto.")
        format_choice = "npy"

    # Pedir nombre del archivo sin extensión
    filename = input("Ingrese el nombre del lienzo (sin extensión): ").strip()
    if not filename:
        print("Nombre inválido. Usando 'lienzo_guardado' por defecto.")
        filename = "lienzo_guardado"

    # Agregar extensión al nombre del archivo
    filepath = os.path.join(CANVAS_DIR, f"{filename}.{format_choice}")

    if format_choice == "png":
        cv2.imwrite(filepath, canvas)
        print(f"Lienzo guardado como imagen en {filepath}")
    elif format_choice == "npy":
        np.save(filepath, canvas)
        print(f"Lienzo guardado en {filepath}")


def list_canvases():
    """Lista todos los lienzos disponibles en el directorio en formatos .npy y .png."""
    files = [f for f in os.listdir(CANVAS_DIR) if f.endswith(".npy") or f.endswith(".png")]
    if not files:
        print("No hay lienzos guardados.")
        return []
    print("Lienzos disponibles:")
    for i, file in enumerate(files, start=1):
        print(f"{i}. {file}")
    return files

def load_canvas():
    """Permite seleccionar y cargar un lienzo desde el directorio en formatos .npy o .png."""
    files = list_canvases()
    if not files:
        return None
    try:
        choice = int(input("Seleccione el número del lienzo que desea cargar: "))
        if 1 <= choice <= len(files):
            filepath = os.path.join(CANVAS_DIR, files[choice - 1])
            if filepath.endswith(".npy"):
                loaded_canvas = np.load(filepath)
                print(f"Lienzo cargado desde {filepath}")
            elif filepath.endswith(".png"):
                loaded_canvas = cv2.imread(filepath)
                if loaded_canvas is not None:
                    print(f"Lienzo cargado desde {filepath}")
                else:
                    print(f"Error al cargar la imagen {filepath}")
                    return None
            return loaded_canvas
        else:
            print("Selección inválida.")
    except ValueError:
        print("Por favor, ingrese un número válido.")
    return None

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
            0.5,   #Modificar opacidad del lienzo
            0
        )

        # Mostrar la vista combinada
        cv2.imshow("Hoja de trabajo", combined_view)

        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Salir con 'ESC'
            break
        elif key == ord('c'):  # Limpiar lienzo
            canvas.fill(255)
        elif key == ord('s'):  # Guardar lienzo en formato .npy
            save_canvas(canvas)
        elif key == ord('l'):  # Cargar lienzo desde archivo .npy
            loaded_canvas = load_canvas()
            if loaded_canvas is not None:
                canvas = loaded_canvas




cap.release()
cv2.destroyAllWindows()