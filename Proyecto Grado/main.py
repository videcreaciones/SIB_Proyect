import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
SMOOTHING_FACTOR = 0.35
CANVAS_SIZE = (1200, 1800, 3)  # Tamaño del lienzo (alto, ancho, canales)
VIEWPORT_SIZE = (480, 640)  # Tamaño del área visible
PROXIMITY_THRESHOLD = 30  # Umbral para los dedos índice y medio
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

with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.5) as hands:
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

                # Activar dibujo si índice y medio están juntos
                if distance < PROXIMITY_THRESHOLD:
                    is_drawing = True
                else:
                    is_drawing = False
                    last_point = None
                    start_point = None

                # Dibujar según el estado actual
                if is_drawing:
                    if start_point is None:
                        start_point = index_coords
                    else:
                        start_point = (
                            int(start_point[0] + (index_coords[0] - start_point[0]) * SMOOTHING_FACTOR),
                            int(start_point[1] + (index_coords[1] - start_point[1]) * SMOOTHING_FACTOR)
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
            0.5,
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
        elif key == ord('s'):  # Guardar imagen
            cv2.imwrite("hoja_de_trabajo.png", canvas)

cap.release()
cv2.destroyAllWindows()
