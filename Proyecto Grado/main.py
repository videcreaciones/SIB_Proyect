import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

# Ver si el dedo pulgar está levantado
def is_thumb_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo pulgar está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo pulgar está levantado, False en caso contrario.
    """
    tip = hand_landmarks.landmark[4]  # Punta del dedo pulgar
    pip = hand_landmarks.landmark[2]  # Articulación PIP del dedo pulgar

    # El pulgar está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

# Ver si el dedo anular está levantado
def is_ring_finger_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo anular está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo anular está levantado, False en caso contrario.
    """
    tip = hand_landmarks.landmark[16]  # Punta del dedo anular
    pip = hand_landmarks.landmark[14]  # Articulación PIP del dedo anular

    # El dedo anular está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

# Ver si el dedo meñique está levantado
def is_pinky_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo meñique está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo meñique está levantado, False en caso contrario.
    """
    tip = hand_landmarks.landmark[20]  # Punta del dedo meñique
    pip = hand_landmarks.landmark[18]  # Articulación PIP del dedo meñique

    # El dedo meñique está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

#ver si el dedo indice esta levantado
def is_index_finger_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo índice está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo índice está levantado, False en caso contrario.
    """
    # Coordenadas de las articulaciones relevantes del dedo índice
    tip = hand_landmarks.landmark[8]  # Punta del dedo índice
    pip = hand_landmarks.landmark[6]  # Articulación PIP del dedo índice

    # El dedo índice está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

#Ver si el dedo medio esta levantado
def is_middle_finger_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo medio está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo medio está levantado, False en caso contrario.
    """
    # Coordenadas de las articulaciones relevantes del dedo medio
    tip = hand_landmarks.landmark[12]  # Punta del dedo medio
    pip = hand_landmarks.landmark[10]  # Articulación PIP del dedo medio

    # El dedo medio está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y


# Función para calcular la distancia euclidiana entre dos puntos
def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)



def draw_on_canvas(canvas, start_point, end_point, color=(0, 0, 0), thickness=5):
    """
    Dibuja una línea en el lienzo entre dos puntos.

    Parámetros:
        canvas: El lienzo donde se dibuja.
        start_point: Coordenadas del punto inicial (x, y).
        end_point: Coordenadas del punto final (x, y).
        color: Color de la línea (por defecto negro).
        thickness: Grosor de la línea (por defecto 5).
    """
    if start_point is not None and end_point is not None:
        cv2.line(canvas, start_point, end_point, color, thickness)

def process_gesture(hand_landmarks, width, height, canvas, proximity_threshold=30):
    """
    Procesa los gestos de la mano para decidir si se debe dibujar en el lienzo.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.
        width: Ancho del cuadro de entrada (video/cámara).
        height: Altura del cuadro de entrada (video/cámara).
        canvas: Lienzo donde se realiza el dibujo.
        proximity_threshold: Distancia mínima entre los dedos índice y medio para dibujar.
    """
    # Coordenadas de las puntas de los dedos índice y medio
    index_tip = (int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height))
    middle_tip = (int(hand_landmarks.landmark[12].x * width), int(hand_landmarks.landmark[12].y * height))

    # Verificar si ambos dedos están levantados
    index_up = is_index_finger_up(hand_landmarks)
    middle_up = is_middle_finger_up(hand_landmarks)

    # Calcular la distancia entre los dedos
    distance = calculate_distance(index_tip, middle_tip)

    # Dibujar si ambos dedos están levantados y la distancia es menor al umbral
    if index_up and middle_up and distance < proximity_threshold:
        draw_on_canvas(canvas, index_tip, middle_tip, color=(0, 0, 255), thickness=3)


import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

# Ver si el dedo pulgar está levantado
def is_thumb_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo pulgar está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo pulgar está levantado, False en caso contrario.
    """
    tip = hand_landmarks.landmark[4]  # Punta del dedo pulgar
    pip = hand_landmarks.landmark[2]  # Articulación PIP del dedo pulgar

    # El pulgar está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

# Ver si el dedo anular está levantado
def is_ring_finger_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo anular está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo anular está levantado, False en caso contrario.
    """
    tip = hand_landmarks.landmark[16]  # Punta del dedo anular
    pip = hand_landmarks.landmark[14]  # Articulación PIP del dedo anular

    # El dedo anular está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

# Ver si el dedo meñique está levantado
def is_pinky_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo meñique está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo meñique está levantado, False en caso contrario.
    """
    tip = hand_landmarks.landmark[20]  # Punta del dedo meñique
    pip = hand_landmarks.landmark[18]  # Articulación PIP del dedo meñique

    # El dedo meñique está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

#ver si el dedo indice esta levantado
def is_index_finger_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo índice está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo índice está levantado, False en caso contrario.
    """
    # Coordenadas de las articulaciones relevantes del dedo índice
    tip = hand_landmarks.landmark[8]  # Punta del dedo índice
    pip = hand_landmarks.landmark[6]  # Articulación PIP del dedo índice

    # El dedo índice está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y

#Ver si el dedo medio esta levantado
def is_middle_finger_up(hand_landmarks) -> bool:
    """
    Detecta si el dedo medio está levantado.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.

    Devuelve:
        bool: True si el dedo medio está levantado, False en caso contrario.
    """
    # Coordenadas de las articulaciones relevantes del dedo medio
    tip = hand_landmarks.landmark[12]  # Punta del dedo medio
    pip = hand_landmarks.landmark[10]  # Articulación PIP del dedo medio

    # El dedo medio está levantado si la punta está por encima de la articulación PIP
    return tip.y < pip.y


# Función para calcular la distancia euclidiana entre dos puntos
def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)



def draw_on_canvas(canvas, start_point, end_point, color=(0, 0, 0), thickness=5):
    """
    Dibuja una línea en el lienzo entre dos puntos.

    Parámetros:
        canvas: El lienzo donde se dibuja.
        start_point: Coordenadas del punto inicial (x, y).
        end_point: Coordenadas del punto final (x, y).
        color: Color de la línea (por defecto negro).
        thickness: Grosor de la línea (por defecto 5).
    """
    if start_point is not None and end_point is not None:
        cv2.line(canvas, start_point, end_point, color, thickness)

def process_gesture(hand_landmarks, width, height, canvas, proximity_threshold=30):
    """
    Procesa los gestos de la mano para decidir si se debe dibujar en el lienzo.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.
        width: Ancho del cuadro de entrada (video/cámara).
        height: Altura del cuadro de entrada (video/cámara).
        canvas: Lienzo donde se realiza el dibujo.
        proximity_threshold: Distancia mínima entre los dedos índice y medio para dibujar.
    """
    # Coordenadas de las puntas de los dedos índice y medio
    index_tip = (int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height))
    middle_tip = (int(hand_landmarks.landmark[12].x * width), int(hand_landmarks.landmark[12].y * height))

    # Verificar si ambos dedos están levantados
    index_up = is_index_finger_up(hand_landmarks)
    middle_up = is_middle_finger_up(hand_landmarks)

    # Calcular la distancia entre los dedos
    distance = calculate_distance(index_tip, middle_tip)

    # Dibujar si ambos dedos están levantados y la distancia es menor al umbral
    if index_up and middle_up and distance < proximity_threshold:
        draw_on_canvas(canvas, index_tip, middle_tip, color=(0, 0, 255), thickness=3)

def Movimiento_En_Lienzo(hand_landmarks, width, height):
    """
    Detecta si se hace el gesto donde el pulgar y el meñique están levantados,
    mientras que los otros dedos están doblados.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.
        width: Ancho de la imagen.
        height: Altura de la imagen.

    Devuelve:
        bool: True si se detecta el gesto, False en caso contrario.
    """
    # Obtener coordenadas en píxeles para los dedos relevantes
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    pinky_tip = hand_landmarks.landmark[20]
    pinky_base = hand_landmarks.landmark[18]

    # Convertir a píxeles
    thumb_tip = (int(thumb_tip.x * width), int(thumb_tip.y * height))
    pinky_tip = (int(pinky_tip.x * width), int(pinky_tip.y * height))
    thumb_base = (int(thumb_base.x * width), int(thumb_base.y * height))
    pinky_base = (int(pinky_base.x * width), int(pinky_base.y * height))

    # Verificar si el pulgar y el meñique están levantados
    thumb_extended = thumb_tip[1] < thumb_base[1]
    pinky_extended = pinky_tip[1] < pinky_base[1]

    # Verificar si los otros dedos están doblados
    index_bent = hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y
    middle_bent = hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y
    ring_bent = hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y

    # Retornar True si el gesto coincide
    return thumb_extended and pinky_extended and index_bent and middle_bent and ring_bent

def Movimiento_En_Lienzo_Funcionalidad(hand_landmarks, width, height, viewport_top_left, canvas_size, viewport_size, last_hand_center):
    """Controla el movimiento en el lienzo cuando se detecta la seña 🤙."""
    hand_center_x = int(hand_landmarks.landmark[9].x * width)
    hand_center_y = int(hand_landmarks.landmark[9].y * height)

    # Si hay una posición previa, calcula el desplazamiento
    if last_hand_center is not None:
        dx = hand_center_x - last_hand_center[0]
        dy = hand_center_y - last_hand_center[1]

        # Actualizar las coordenadas del viewport dentro de los límites del lienzo
        viewport_top_left[0] = np.clip(viewport_top_left[0] - dx, 0, canvas_size[1] - viewport_size[1])
        viewport_top_left[1] = np.clip(viewport_top_left[1] - dy, 0, canvas_size[0] - viewport_size[0])

    # Actualizar la última posición de la mano
    last_hand_center = (hand_center_x, hand_center_y)
    return viewport_top_left, last_hand_center

# Función para procesar el lienzo y la vista
def process_canvas(frame, results, canvas, viewport_top_left, params):
    """
    Procesa el lienzo y la vista basándose en los gestos detectados.

    Parámetros:
        frame: Fotograma actual del video.
        results: Resultados de los landmarks de MediaPipe.
        canvas: Lienzo donde se dibuja.
        viewport_top_left: Posición inicial del viewport sobre el lienzo.
        params: Diccionario con parámetros como suavizado y estado del dibujo.

    Devuelve:
        viewport: Vista recortada del lienzo.
        last_hand_center: Última posición del centro de la mano.
        last_point: Última posición del punto dibujado.
        start_point: Posición inicial del dibujo.
        is_drawing: Estado de si se está dibujando o no.
    """
    height, width, _ = frame.shape
    is_drawing = params["is_drawing"]
    last_point = params["last_point"]
    start_point = params["start_point"]
    last_hand_center = params["last_hand_center"]

    SMOOTHING_FACTOR = params["smoothing_factor"]
    PROXIMITY_THRESHOLD = params["proximity_threshold"]
    VIEWPORT_SIZE = params["viewport_size"]
    CANVAS_SIZE = params["canvas_size"]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Verificar si los dedos índice y medio están levantados
            index_up = is_index_finger_up(hand_landmarks)
            middle_up = is_middle_finger_up(hand_landmarks)

            # Solo activar el dibujo si ambos dedos están levantados
            if index_up and middle_up:
                index_coords = (int(hand_landmarks.landmark[8].x * width),
                                int(hand_landmarks.landmark[8].y * height))
                middle_coords = (int(hand_landmarks.landmark[12].x * width),
                                 int(hand_landmarks.landmark[12].y * height))

                distance = calculate_distance(index_coords, middle_coords)

                # Dibujar si la distancia entre los dedos es menor al umbral
                if distance < PROXIMITY_THRESHOLD:
                    is_drawing = True
                else:
                    is_drawing = False
                    last_point = None
                    start_point = None

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

                # Dibujar marcador para la punta del índice
                cv2.line(frame, (index_coords[0] - 10, index_coords[1]),
                         (index_coords[0] + 10, index_coords[1]), (0, 0, 255), 2)
                cv2.line(frame, (index_coords[0], index_coords[1] - 10),
                         (index_coords[0], index_coords[1] + 10), (0, 0, 255), 2)

    x, y = viewport_top_left
    x_end = x + VIEWPORT_SIZE[1]
    y_end = y + VIEWPORT_SIZE[0]
    viewport = canvas[y:y_end, x:x_end]
    return viewport, last_hand_center, last_point, start_point, is_drawing


# Función principal
def main():
    SMOOTHING_FACTOR = 0.35
    CANVAS_SIZE = (1200, 1800, 3)
    VIEWPORT_SIZE = (480, 640)
    PROXIMITY_THRESHOLD = 30

    canvas = np.ones(CANVAS_SIZE, dtype=np.uint8) * 255
    viewport_top_left = [0, 0]

    params = {
        "smoothing_factor": SMOOTHING_FACTOR,
        "proximity_threshold": PROXIMITY_THRESHOLD,
        "canvas_size": CANVAS_SIZE,
        "viewport_size": VIEWPORT_SIZE,
        "is_drawing": False,
        "last_point": None,
        "start_point": None,
        "last_hand_center": None
    }

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            viewport, params["last_hand_center"], params["last_point"], params["start_point"], params[
                "is_drawing"] = process_canvas(
                frame, results, canvas, viewport_top_left, params)

            combined_view = frame.copy()
            combined_view[0:VIEWPORT_SIZE[0], 0:VIEWPORT_SIZE[1]] = cv2.addWeighted(
                frame[0:VIEWPORT_SIZE[0], 0:VIEWPORT_SIZE[1]],
                0.5,
                viewport,
                0.5,
                0
            )

            cv2.imshow("Hoja de trabajo", combined_view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                canvas.fill(255)
            elif key == ord('s'):
                cv2.imwrite("hoja_de_trabajo.png", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


def Movimiento_En_Lienzo(hand_landmarks, width, height):
    """
    Detecta si se hace el gesto donde el pulgar y el meñique están levantados,
    mientras que los otros dedos están doblados.

    Parámetros:
        hand_landmarks: Estructura de landmarks de MediaPipe para una mano.
        width: Ancho de la imagen.
        height: Altura de la imagen.

    Devuelve:
        bool: True si se detecta el gesto, False en caso contrario.
    """
    # Obtener coordenadas en píxeles para los dedos relevantes
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    pinky_tip = hand_landmarks.landmark[20]
    pinky_base = hand_landmarks.landmark[18]

    # Convertir a píxeles
    thumb_tip = (int(thumb_tip.x * width), int(thumb_tip.y * height))
    pinky_tip = (int(pinky_tip.x * width), int(pinky_tip.y * height))
    thumb_base = (int(thumb_base.x * width), int(thumb_base.y * height))
    pinky_base = (int(pinky_base.x * width), int(pinky_base.y * height))

    # Verificar si el pulgar y el meñique están levantados
    thumb_extended = thumb_tip[1] < thumb_base[1]
    pinky_extended = pinky_tip[1] < pinky_base[1]

    # Verificar si los otros dedos están doblados
    index_bent = hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y
    middle_bent = hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y
    ring_bent = hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y

    # Retornar True si el gesto coincide
    return thumb_extended and pinky_extended and index_bent and middle_bent and ring_bent

def Movimiento_En_Lienzo_Funcionalidad(hand_landmarks, width, height, viewport_top_left, canvas_size, viewport_size, last_hand_center):
    """Controla el movimiento en el lienzo cuando se detecta la seña 🤙."""
    hand_center_x = int(hand_landmarks.landmark[9].x * width)
    hand_center_y = int(hand_landmarks.landmark[9].y * height)

    # Si hay una posición previa, calcula el desplazamiento
    if last_hand_center is not None:
        dx = hand_center_x - last_hand_center[0]
        dy = hand_center_y - last_hand_center[1]

        # Actualizar las coordenadas del viewport dentro de los límites del lienzo
        viewport_top_left[0] = np.clip(viewport_top_left[0] - dx, 0, canvas_size[1] - viewport_size[1])
        viewport_top_left[1] = np.clip(viewport_top_left[1] - dy, 0, canvas_size[0] - viewport_size[0])

    # Actualizar la última posición de la mano
    last_hand_center = (hand_center_x, hand_center_y)
    return viewport_top_left, last_hand_center

# Función para procesar el lienzo y la vista


# Función principal
def main():
    SMOOTHING_FACTOR = 0.35
    CANVAS_SIZE = (1200, 1800, 3)
    VIEWPORT_SIZE = (480, 640)
    PROXIMITY_THRESHOLD = 30

    canvas = np.ones(CANVAS_SIZE, dtype=np.uint8) * 255
    viewport_top_left = [0, 0]

    params = {
        "smoothing_factor": SMOOTHING_FACTOR,
        "proximity_threshold": PROXIMITY_THRESHOLD,
        "canvas_size": CANVAS_SIZE,
        "viewport_size": VIEWPORT_SIZE,
        "is_drawing": False,
        "last_point": None,
        "start_point": None,
        "last_hand_center": None
    }

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            viewport, params["last_hand_center"], params["last_point"], params["start_point"], params[
                "is_drawing"] = process_canvas(
                frame, results, canvas, viewport_top_left, params)

            combined_view = frame.copy()
            combined_view[0:VIEWPORT_SIZE[0], 0:VIEWPORT_SIZE[1]] = cv2.addWeighted(
                frame[0:VIEWPORT_SIZE[0], 0:VIEWPORT_SIZE[1]],
                0.5,
                viewport,
                0.5,
                0
            )

            cv2.imshow("Hoja de trabajo", combined_view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                canvas.fill(255)
            elif key == ord('s'):
                cv2.imwrite("hoja_de_trabajo.png", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
