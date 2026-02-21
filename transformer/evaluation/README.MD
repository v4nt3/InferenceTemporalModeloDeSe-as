# Sign Language Inference

Script temporal para realizar inferencia de reconocimiento de Lengua de Señas usando un modelo entrenado basado en landmarks.

## Requisitos

* Python 3.8+
* PyTorch
* OpenCV
* NumPy

Instalación rápida:

```bash
pip install -r requirements.txt    
```

---

## Uso

```bash
python inference.py --mode <modo> --checkpoint <modelo> [opciones]
```

### Argumentos principales

* `--mode` : `sign` | `sentence` | `camera`
* `--checkpoint` : ruta al modelo entrenado (requerido)
* `--video` : ruta al video (requerido para `sign` y `sentence`)
* `--labels` : archivo de etiquetas (opcional)
* `--device` : `cpu` o `cuda` (opcional)
* `--camera` : índice de cámara (default: 0)

---

## Modos

### 1. Predicción de una seña

```bash
python inference.py --mode sign --video testing/video4.mp4 --checkpoint outputs/checkpoints/best_model.pt
```

Salida:

* Clase predicha
* Confianza
* Top-K predicciones

---

### 2. Predicción de secuencia

```bash
python inference.py --mode sentence --video testing/video4.mp4 --labels data/labels.json --config outputs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Salida:

* Oración detectada
* Frames de inicio y fin por seña

---

### 3. Tiempo real (cámara)

```bash
python inference.py --mode camera --checkpoint outputs/checkpoints/best_model.pt --labels data/labels.json --config outputs/config.yaml
```

Opcional:

```bash
--camera 1
```

Presiona **Q** para salir.

---

## Notas

* El procesamiento inicia desde el primer frame donde se detectan manos.
* Usa GPU automáticamente si está disponible (`--device cuda`).
