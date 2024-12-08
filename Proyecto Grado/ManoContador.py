from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QListWidget, QMessageBox
)
from PyQt6.QtCore import Qt
import cv2
import numpy as np
import threading
import os
import sys

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Crear y mostrar la ventana principal en un hilo separado
    main_window = MainWindow()

    video_thread = threading.Thread(target=show_video, daemon=True)
    video_thread.start()

    main_window.show()
    sys.exit(app.exec())
