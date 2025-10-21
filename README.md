<p align="center">
  <img src="https://github.com/videcreaciones/SIB_Proyect/blob/main/banner.png?raw=true" alt="Sign Intelligent Board" width="800"/>
</p>

# Sign Intelligent Board - S.I.B

**Sign Intelligent Board (S.I.B)** es un proyecto que implementa un **tablero inteligente con Inteligencia Artificial** para el reconocimiento de señas, con el objetivo de fomentar el uso de las TIC en el aula.  

El sistema interpreta gestos mediante una cámara y ofrece una experiencia interactiva entre docentes y estudiantes, optimizando las dinámicas de enseñanza-aprendizaje.

---

## 🚀 Tecnologías utilizadas

- Python 3  
- OpenCV (`cv2`)  
- MediaPipe (detección de manos)  
- NumPy  
- PyQt6  
- Raspberry Pi OS / Windows / Linux  

---

## 📁 Estructura del repositorio

```
SIB_Proyect/
├── Proyecto_Grado/
│   └── main.py         ← Archivo principal de ejecución
├── Documentos/         ← Información del proyecto de investigación
├── README.md           ← Este archivo
└── ...
```

---

## ⚙️ Instalación y ejecución

### 🔹 Opción 1: En Raspberry Pi

1. **Clonar el repositorio**
   ```bash
   git clone git@github.com:videcreaciones/SIB_Proyect.git
   ```

2. **Crear y activar un entorno virtual**
   ```bash
   python3 -m venv mano
   source mano/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   python3 -m pip install --upgrade pip
   pip install mediapipe numpy opencv-python pyqt6
   ```

4. **Ejecutar el proyecto**
   ```bash
   cd Proyecto_Grado
   python3 main.py
   ```

5. **Desactivar entorno (opcional)**
   ```bash
   deactivate
   ```

---

### 🔹 Opción 2: En cualquier computadora (Windows/Linux)

1. **Descargar o extraer el repositorio**
   - Si descargaste el archivo `.zip`, extráelo en una carpeta local.  
   - O clónalo directamente:
     ```bash
     git clone https://github.com/videcreaciones/SIB_Proyect.git
     ```

2. **Crear un entorno virtual**
   - En Windows:
     ```bash
     python -m venv mano
     mano\Scripts\activate
     ```
   - En Linux/Mac:
     ```bash
     python3 -m venv mano
     source mano/bin/activate
     ```

3. **Instalar las librerías necesarias**
   ```bash
   pip install --upgrade pip
   pip install mediapipe numpy opencv-python pyqt6
   ```

4. **Ejecutar el programa**
   ```bash
   cd Proyecto_Grado
   python main.py
   ```

5. **Cerrar el entorno**
   ```bash
   deactivate
   ```

---

## 🖐️ ¿Cómo se usa?

El sistema funciona únicamente por **señas**.  
La cámara reconoce los movimientos de la mano en tiempo real y la interfaz gráfica responde según el gesto detectado.

- ✋ **Palma abierta** → abrir menú lateral  
- 🤙 **Shake** → desplazarse en el tablero  
- 👆 **Seña del indice y medio levantados** → dibujar ( separar los dedos para dejar de hacer el trazo.
- ✊ **Puño cerrado** → borrar  

Para más información sobre los gestos y funcionalidades, consulta la interfaz o los documentos del proyecto.

---

## 👨‍💻 Autor

**Salomón Jarro Cerón**  
📧 [chovengo2018@gmail.com](mailto:chovengo2018@gmail.com)  

---
