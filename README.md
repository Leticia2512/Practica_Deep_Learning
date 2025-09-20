# Predicción de Engagement en Puntos de Interés Turísticos (POIs)

Módulo Deep Learning - Bootcamp KeepCoding

---

## 📝 Descripción

Proyecto de *Deep Learning* para predecir el **nivel de engagement (bajo/medio/alto)** de puntos de interés turísticos (POIs) combinando:

- **Imágenes**: características visuales extraídas con ResNet-18.
- **Metadatos**: coordenadas, categorías, etiquetas, popularidad, métricas de gamificación y engagement.

El enfoque **multimodal** (CNN + MLP) mejora la predicción frente a modelos unidimensionales.


---

## 🗂️ Dataset

- **Fuente:** Plataforma Artgonuts.
- **POIs:** 1 569 puntos de interés con imagen principal (`main_image_path`) y metadatos asociados.
- **Variable objetivo:** `engagement_level` (bajo=0, medio=1, alto=2) generada a partir de un *engagement_score* compuesto.

---

## ⚙️ Reproducibilidad

- Semillas fijadas para `random`, `numpy` y `torch`.
- Dependencias en `requirements.txt`.
- Entrenamiento y evaluación reproducibles con GPU si está disponible.
- Modelo final guardado en `models/best_model_final.pth`.

---

## 🔧 Preparación y Preprocesamiento

- Eliminación de variables irrelevantes o que requerían NLP avanzado.

- Transformación logarítmica y normalización de métricas de engagement.

- Codificación multi-hot de categorías y selección de top-tags.

- Estandarización de variables numéricas con StandardScaler.

- Imágenes redimensionadas a 224×224, normalización ImageNet y data augmentation moderado (rotaciones, alteración de color).


---

## 🏗️ Arquitectura del Modelo

Modelo multimodal con dos ramas:

- **Rama visual**: ResNet-18 preentrenada (ImageNet), capa final eliminada para extraer embeddings (512-D).

- **Rama metadatos**: MLP de dos capas fully-connected con BatchNorm1d, activación ReLU/ELU y Dropout.

- Fusión: Concatenación de ambas salidas → bloque adicional de clasificación con normalización y Dropout → salida de 3 clases.


---

## 🏗️ Entrenamiento y optimización

- Batch size: 256.
- **Pérdida:** CrossEntropyLoss con `class_weights` + `label_smoothing`.
- **Optimizador:** AdamW con ReduceLROnPlateau + Early Stopping.
- Fases:

	1. Baseline (backbone congelado)

	2. Fine-tuning parcial con LR discriminativo

	3. Optimización de hiperparámetros con Optuna (40 trials)

---

## 📊 Resultados


| Fase               | Val. Accuracy | Test Accuracy | Macro-F1 (Test) |
| ------------------ | ------------: | ------------: | --------------: |
| Baseline           |        87.9 % |        84.5 % |           0.809 |
| Fine-tuning        |        90.9 % |             — |               — |
| **Optuna (final)** |    **93.4 %** |    **88.8 %** |       **0.875** |


| Clase     | Precisión | Recall |    F1 |
| --------- | --------: | -----: | ----: |
| 0 (bajo)  |     0.965 |  0.893 | 0.928 |
| 1 (medio) |     0.887 |  0.810 | 0.847 |
| 2 (alto)  |     0.761 |  0.962 | 0.850 |


Grad-CAM aplicado a la rama visual muestra que la red se enfoca en rasgos estéticos distintivos en las predicciones correctas.

---

## 📝 Conclusiones y Propuesta futuras

- El modelo multimodal mejora sustancialmente los enfoques unidimensionales (93.4 % val, 88.8 % test).

- Técnicas clave: LR discriminativo, AdamW + ReduceLROnPlateau, label smoothing y búsqueda de hiperparámetros con Optuna.

- Grad-CAM valida que la rama visual aprende patrones relevantes.

Mejoras potenciales:

- Augmentación dirigida para clases minoritarias.

- Estrategias de freeze/unfreeze progresivas (One-Cycle LR, cosine annealing).

- Técnicas de reweighting focal para mejorar recall minoritario.

- Análisis de outliers con Grad-CAM para refinar etiquetado y arquitectura.

---

## 🚀 Uso

1. Clonar el repositorio.
2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt

3. [👉 Abrir notebook](Practica_Deep_Learning.ipynb) para reproducir entrenamiento y evaluación.

4. Cargar models/best_model_final.pth para inferencia:

model = POIMultimodalModel(...)

model.load_state_dict(torch.load("Models/best_model_final.pth", map_location="cpu"))

model.eval()








