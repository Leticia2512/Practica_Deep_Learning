# Predicción de Engagement en Puntos de Interés Turísticos (POIs)

**Módulo Deep Learning - Bootcamp KeepCoding**

---

## 📝 Descripción

Proyecto de *Deep Learning* para predecir el **nivel de engagement (bajo/medio/alto)** de POIs combinando:

- **Imágenes:** características visuales extraídas con ResNet-18.  
- **Metadatos:** coordenadas, categorías, etiquetas, popularidad y métricas de engagement.

El enfoque **multimodal (CNN + MLP)** mejora la predicción frente a modelos unidimensionales.

---

## 🗂️ Dataset

- **Fuente:** Plataforma Artgonuts  
- **POIs:** 1 569 puntos de interés con imagen principal y metadatos  
- **Variable objetivo:** `engagement_level` (0=bajo, 1=medio, 2=alto)

---

## ⚙️ Reproducibilidad

- Semillas fijadas para `random`, `numpy` y `torch`.  
- Dependencias en [`requirements.txt`](requirements.txt)  
- Entrenamiento y evaluación reproducibles en CPU/GPU  
- Modelo final: `models/best_model_final.pth` 

---

## 🔧 Preprocesamiento

- Metadatos estandarizados y codificados  
- Imágenes redimensionadas a 224×224, normalizadas y aumentadas moderadamente (rotación, color, crop)  

---

## 🏗️ Arquitectura del Modelo

- **Rama visual:** ResNet-18 preentrenada (ImageNet), capa final eliminada para embeddings (512-D).  
- **Rama metadatos:** MLP de dos capas con BatchNorm1d, activación y Dropout.  
- **Fusión:** Concatenación de ambas ramas → MLP de clasificación → salida de 3 clases.

---

## 🔧 Entrenamiento y Optimización

- **Batch size:** 256  
- **Pérdida:** CrossEntropyLoss con `class_weights` + `label_smoothing`  
- **Optimizador:** AdamW + ReduceLROnPlateau + Early Stopping  

**Fases de entrenamiento:**

| Fase               | Hiperparámetros relevantes                                                   | Mejor val_acc | Observaciones                                                         |
| ------------------ | ---------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------- |
| Baseline           | lr=1e-3, dropout=0.4, label_smoothing=0.05                                  | 86.36%        | Backbone congelado, aprende características desde la cabeza.         |
| Fine-tuning        | lr_head=5e-4, lr_backbone=5e-5                                             | 90.91%        | Ajuste parcial del backbone mejora generalización.                   |
| **Optuna (final)** | lr_head=9.52e-4, lr_backbone=1.19e-4, dropout=0.316, label_smoothing=0.04 | 93.94%        | Optimización sistemática, mejora significativa frente a fine-tuning. |

---

## 📊 Resultados

### Rendimiento global

| Fase               | Val. Accuracy | Test Accuracy | Macro-F1 (Test) |
| ------------------ | ------------: | ------------: | --------------: |
| Baseline           | 87.9 %        | 84.5 %        | 0.809           |
| Fine-tuning        | 90.9 %        | —             | —               |
| **Optuna (final)** | **93.4 %**    | **90.13 %**   | 0.8875          |

### Métricas por clase

| Clase     | Precisión | Recall | F1-score |
| --------- | --------: | -----: | -------: |
| 0 (bajo)  | 0.974     | 0.910  | 0.941    |
| 1 (medio) | 0.875     | 0.845  | 0.860    |
| 2 (alto)  | 0.794     | 0.943  | 0.862    |

**Insight visual:** Grad-CAM muestra que la rama visual se enfoca en rasgos distintivos de los POIs y permite analizar errores de predicción.

**Conclusión:**  
*El modelo multimodal permite predecir de manera precisa y confiable el engagement en POIs, facilitando mejoras futuras y aplicaciones en turismo y análisis de datos.*

---

## 🚀 Uso

1. Clonar el repositorio  
2. Instalar dependencias:

```bash
pip install -r requirements.txt

3. [📓 Abrir notebook](https://github.com/Leticia2512/Practica_Deep_Learning/blob/main/Notebooks/Practica_Deep_Learning.ipynb)

4. Cargar modelo para inferencia:

from models import POIMultimodalModel

model = POIMultimodalModel(...)
model.load_state_dict(torch.load("models/best_model_final.pth", map_location="cpu"))
model.eval()






