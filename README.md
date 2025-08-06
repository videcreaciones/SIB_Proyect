
```
   _____ _____ ____  
  / ____|_   _|  _ \ 
 | (___   | | | |_) |
  \___ \  | | |  _ < 
  ____) |_| |_| |_) |
 |_____/|_____|____/ 
                    
```

# Sign Inteligent Board

Este proyecto consiste en el desarrollo de un **Tablero Inteligente** que utiliza **Inteligencia Artificial de reconocimiento de imágenes** para fomentar el uso de las TIC en el aula. Funciona mediante **señas captadas por una cámara**, y está diseñado especialmente para facilitar dinámicas interactivas entre docentes y estudiantes.

El sistema corre sobre una **Raspberry Pi**, e integra procesamiento de imagen en tiempo real con una interfaz gráfica intuitiva.

---

## 🚀 Tecnologías utilizadas

- **Python 3**
- **OpenCV (cv2)**
- **MediaPipe** (detección de manos)
- **NumPy**
- **PyQt6**
- **Raspberry Pi OS**

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

## ⚙️ Instalación en Raspberry Pi

1. **Clonar el repositorio**
   ```bash
   git clone git@github.com:videcreaciones/PR-Grado.git
   ```

2. **Crear y activar entorno virtual**
   ```bash
   python3 -m venv mano
   source mano/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   python3 -m pip install --upgrade pip
   pip install mediapipe
   pip install numpy
   pip install pyqt6 --only-binary :all:
   ```

4. **Desactivar entorno**
   ```bash
   deactivate
   ```

5. **Actualizar e instalar herramientas**
   ```bash
   sudo apt update
   sudo apt install build-essential curl file git
   ```

6. **Permitir escritura temporalmente en /bin**
   ```bash
   sudo chmod o+w /bin
   ```

7. **Agregar archivo `mutagen` en `/bin`**  
   *(Consulta el repositorio para saber cuál es el archivo correcto)*

8. **Restablecer permisos de /bin**
   ```bash
   sudo chmod o-w /bin
   ```

9. **Sincronizar con Mutagen**
   ```bash
   mutagen daemon start
   mutagen sync create --sync-mode=one-way-safe "D:\Mis documentos\Escritorio\Proyectos\PR-Grado\Proyecto_Grado" vyc@raspberrypi.local:/home/vyc/PR-Grado/Proyecto_Grado
   ```

---

## 🧠 ¿Cómo se usa?

El tablero inteligente se utiliza **únicamente mediante señas**. Las manos del usuario son captadas por la cámara y procesadas en tiempo real con MediaPipe. La interfaz gráfica muestra indicaciones e instrucciones específicas según la seña detectada.

> Para más detalles sobre los gestos soportados, revisa la interfaz gráfica del proyecto o los documentos incluidos.

---

## 🤝 Autor

**Chovengo**  
📧 chovengo2018@gmail.com  
📞 +57 321 484 4125

---

## 📜 Licencia

Este proyecto aún no tiene una licencia definida. Puedes modificar o distribuir el código bajo tu propio riesgo.  
(Si deseas agregar una licencia específica como MIT, GPL, etc., puedo ayudarte a incluirla).

---

¡Gracias por visitar el proyecto!  
Cualquier colaboración, sugerencia o mejora es bienvenida.
