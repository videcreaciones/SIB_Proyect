import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def check_fingers_up(hand_landmarks, height, width):
    """Chequea si todos los dedos excepto el pulgar están levantados"""
    fingertips_ids = [8, 12, 16, 20]  # Índices de las yemas de los dedos (índice, medio, anular, meñique)
    for id in fingertips_ids:
        if hand_landmarks.landmark[id].y * height > hand_landmarks.landmark[id - 2].y * height:
            return False
    return True

# Factor de conversión de píxeles a centímetros (ajusta este valor según tu calibración)
FACTOR_CM_PER_PIXEL = 0.05

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Puntos de interés en la mano
index_finger_tip = 8  # Índice de la yema del dedo índice en MediaPipe
middle_finger_tip = 12  # Índice de la yema del dedo medio en MediaPipe

# Inicializar variables para el dibujo
is_drawing = False
is_erasing = False
last_point = None

# Crear una imagen en blanco para dibujar
drawing_board = np.zeros((480, 640, 3), dtype=np.uint8)

with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtener las coordenadas de las yemas de los dedos índice y medio
                index_finger_coords = (int(hand_landmarks.landmark[index_finger_tip].x * width),
                                       int(hand_landmarks.landmark[index_finger_tip].y * height))
                middle_finger_coords = (int(hand_landmarks.landmark[middle_finger_tip].x * width),
                                        int(hand_landmarks.landmark[middle_finger_tip].y * height))

                # Calcular la distancia en píxeles entre el índice y el medio
                distance_px = calculate_distance(index_finger_coords, middle_finger_coords)

                # Convertir la distancia a centímetros
                distance_cm = distance_px * FACTOR_CM_PER_PIXEL

                # Imprimir la distancia en la consola en centímetros
                print(f'Distancia entre índice y medio: {distance_cm:.2f} cm')

                # Dibujar las coordenadas de los dedos
                cv2.circle(frame, index_finger_coords, 5, (255, 0, 0), -1)  # Azul
                cv2.circle(frame, middle_finger_coords, 5, (0, 0, 255), -1)  # Rojo

                # Mostrar la distancia en la pantalla en centímetros
                cv2.putText(frame, f'Distancia: {distance_cm:.2f} cm',
                            (index_finger_coords[0], index_finger_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Dibujar conexiones de la mano
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Verificar si todos los dedos menos el pulgar están levantados
                all_fingers_up = check_fingers_up(hand_landmarks, height, width)

                # Activar o desactivar el dibujo según la distancia y los dedos levantados
                if all_fingers_up:
                    is_drawing = False
                    is_erasing = True
                    print("Borrador activado")
                elif distance_cm < 1.7:
                    is_drawing = True
                    is_erasing = False
                    print("Dibujo activado")
                else:
                    is_drawing = False
                    is_erasing = False
                    last_point = None  # Reiniciar la última posición
                    print("Dibujo y borrado desactivados")

                # Dibujar o borrar según el estado actual
                if is_drawing:
                    if last_point is not None:
                        cv2.line(drawing_board, last_point, index_finger_coords, (0, 255, 0), 15)
                    last_point = index_finger_coords
                elif is_erasing:
                    if last_point is not None:
                        cv2.line(drawing_board, last_point, middle_finger_coords, (0, 0, 0), 40)
                    last_point = index_finger_coords

        # Superponer la imagen de dibujo sobre el frame original
        combined_frame = cv2.addWeighted(frame, 1, drawing_board, 0.5, 1)

        cv2.imshow("Frame", combined_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'ESC' para salir
            break

cap.release()
cv2.destroyAllWindows()