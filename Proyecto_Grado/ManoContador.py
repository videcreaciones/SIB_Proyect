import cv2
import numpy as np

# Configuración de las zonas del menú
menu_zones = [
    {"name": "Opción 1", "x1": 0, "x2": 100, "y1": 0, "y2": 100},
    {"name": "Opción 2", "x1": 0, "x2": 100, "y1": 101, "y2": 200},
    {"name": "Opción 3", "x1": 0, "x2": 100, "y1": 201, "y2": 300},
]

# Variable para registrar la opción seleccionada
selected_option = None
hold_time = 0  # Tiempo que el puntero permanece en una zona
hold_threshold = 30  # Frames necesarios para seleccionar una opción


# Función para verificar si el puntero está en una zona
def check_menu_collision(x, y):
    for zone in menu_zones:
        if zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]:
            return zone["name"]
    return None


# Simulación de bucle principal (reemplaza con tu detección de puntero)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Simulación de puntero (reemplázalo con las coordenadas de tu mano)
    # En este caso, el puntero se mueve con el ratón para pruebas.
    pointer_x, pointer_y = cv2.getTrackbarPos('X', 'Test'), cv2.getTrackbarPos('Y', 'Test')

    # Dibuja el menú en la pantalla
    for zone in menu_zones:
        cv2.rectangle(frame, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), (255, 0, 0), 2)
        cv2.putText(frame, zone["name"], (zone["x1"] + 5, zone["y1"] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # Dibuja el puntero
    cv2.circle(frame, (pointer_x, pointer_y), 10, (0, 255, 0), -1)

    # Detecta si el puntero está en una zona
    current_zone = check_menu_collision(pointer_x, pointer_y)
    if current_zone:
        hold_time += 1
        cv2.putText(frame, f"Seleccionando: {current_zone}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        if hold_time >= hold_threshold:
            selected_option = current_zone
            print(f"Opción seleccionada: {selected_option}")
            hold_time = 0  # Reinicia el tiempo de espera
    else:
        hold_time = 0

    # Muestra el video con el menú
    cv2.imshow("Menu Interactivo", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
