# Proyecto 2: Clasificación de Especies usando Ecoacústica

## Estado del proyecto

**Terminado.**

Proyecto final de Machine Learning orientado a la clasificación automática de especies faunísticas a partir de señales eco-acústicas representadas mediante 64 características Mel (`mel_0` a `mel_63`).

El proyecto incluye:

- Preprocesamiento completo del dataset.
- Reducción dimensional con PCA y UMAP.
- Clustering no supervisado con GMM y DBSCAN.
- Clasificación supervisada con MLP y modelos de ensamble.
- Política de umbrales para mitigación de riesgos.
- Aplicación informativa en Streamlit.
- Informe técnico final en LaTeX/PDF.

---

## Descripción general

El objetivo del proyecto es desarrollar un pipeline integral de Machine Learning para clasificar especies silvestres a partir de datos tabulares eco-acústicos. Cada muestra está representada por un vector de 64 características Mel, y la variable objetivo es `species_id`.

Las especies clasificadas son:

| Species ID | Nombre científico |
|---:|---|
| 10 | *Leptodactylus discodactylus* |
| 12 | *Osteocephalus taurinus* |
| 17 | *Chiroxiphia lineata* |
| 18 | *Saltator grossus* |
| 23 | *Pheucticus chrysopeplus* |

---

## Estructura del proyecto

```text
Proyecto_2/
├── 01_integrante_1_preprocessing_pca_umap.ipynb
├── 02_integrante_2_clustering_mlp_xgboost.ipynb
├── Parte3Y4.py
├── app.py
├── README.md
├── requirements.txt
├── Documentacion_de_dataset.pdf
├── eco_acoustic_train.csv
├── eco_acoustic_test.csv
│
├── data/
│   └── processed/
│       ├── X_dev.csv
│       ├── X_dev_scaled.csv
│       ├── X_val.csv
│       ├── X_val_scaled.csv
│       ├── X_test_official.csv
│       ├── X_test_official_scaled.csv
│       ├── X_train_full_scaled.csv
│       ├── y_dev.csv
│       ├── y_val.csv
│       ├── y_test_official.csv
│       ├── y_train_full.csv
│       ├── recording_dev.csv
│       ├── recording_val.csv
│       ├── recording_test_official.csv
│       ├── class_mapping.csv
│       └── pca_umap_coordinates.csv
│
├── models/
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   ├── xgboost_model.pkl
│   ├── best_ensemble_model.pkl
│   ├── best_mlp_state_dict.pth
│   ├── best_mlp_model_full.pkl
│   ├── lightgbm_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── random_forest_model.pkl
│
├── outputs/
│   ├── figures/
│   │   ├── pca_2d.png
│   │   ├── umap_2d.png
│   │   ├── fig4_gmm_silhouette_bic.png
│   │   ├── fig4b_metricas_internas.png
│   │   └── fig5_dbscan_kdist.png
│   └── tables/
│       ├── class_distribution.csv
│       ├── dimensionality_reduction_results.csv
│       └── clustering_internal_metrics.csv
│
├── results/
│   └── clasificacion/
│       ├── tabla_completa_resultados.csv
│       ├── comparacion_matrices_confusion.png
│       ├── comparacion_metricas_todos.png
│       ├── comparacion_tiempos.png
│       ├── curvas_aprendizaje_comparativas.png
│       ├── curvas_aprendizaje_detalladas.png
│       ├── curvas_aprendizaje_zoom.png
│       ├── politica_umbrales.png
│       ├── politica_umbrales_composicion.png
│       ├── todas_matrices_confusion.png
│       └── tradeoff_costo_rendimiento.png
│
└── report/
    ├── informe_P2.tex
    ├── informe_P2.pdf
    └── figs/
        └── fig_arquitectura.png
```

---

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv venv
```

En Windows:

```bash
venv\Scripts\activate
```

En Linux/Mac:

```bash
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Dependencias principales

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `umap-learn`
- `joblib`
- `xgboost`
- `lightgbm`
- `torch`
- `torchvision`
- `torchaudio`
- `streamlit`

---

## Flujo del proyecto

### 1. Preprocesamiento y reducción dimensional

Notebook:

```text
01_integrante_1_preprocessing_pca_umap.ipynb
```

Incluye:

- Carga de `eco_acoustic_train.csv` y `eco_acoustic_test.csv`.
- Validación de columnas, nulos y distribución de clases.
- Separación de `X` e `y`.
- División interna `dev/val`.
- Escalamiento con `StandardScaler`.
- Codificación de etiquetas con `LabelEncoder`.
- PCA 2D/3D.
- UMAP 2D.
- Generación de tablas y figuras base.

Resultados principales:

| Método | Dimensiones | Tiempo (s) | Varianza explicada | Trustworthiness |
|---|---:|---:|---:|---:|
| PCA | 2 | 0.1599 | 0.6714 | 0.8662 |
| PCA | 3 | 0.0034 | 0.7677 | 0.9403 |
| UMAP | 2 | 26.0173 | — | 0.9777 |

---

### 2. Clustering no supervisado

Notebook:

```text
02_integrante_2_clustering_mlp_xgboost.ipynb
```

Incluye:

- Aplicación de GMM.
- Aplicación de DBSCAN.
- Selección de hiperparámetros.
- Evaluación con métricas internas:
  - Silhouette.
  - Davies-Bouldin.
  - Calinski-Harabasz.
- Comparación externa referencial con ARI.

Resultados principales:

| Método | Paradigma | Grupos | Silhouette | ARI |
|---|---|---:|---:|---:|
| GMM | Probabilístico | 2 | 0.182 | 0.026 |
| DBSCAN | Densidad | 2 | — | 0.003 |

El clustering evidencia que las especies no forman cinco grupos naturalmente separables en el espacio Mel.

---

### 3. Clasificación supervisada

Script:

```text
Parte3Y4.py
```

Incluye:

- Entrenamiento de variantes de MLP.
- Comparación de Dropout y Batch Normalization.
- Entrenamiento de modelos de ensamble:
  - XGBoost.
  - LightGBM.
  - Gradient Boosting.
  - Random Forest.
- Comparación mediante F1-Macro, Accuracy, matrices de confusión y tiempos.
- Guardado de modelos finales en `models/`.

Resultados principales:

| Modelo | Tipo | F1-Macro | Accuracy | Inferencia (s) |
|---|---|---:|---:|---:|
| MLP + Dropout | MLP | 0.4650 | 0.5157 | 0.0013 |
| XGBoost | Ensamble | 0.4519 | 0.5052 | 0.0089 |
| LightGBM | Ensamble | 0.4351 | 0.4864 | 0.0211 |
| Gradient Boosting | Ensamble | 0.4364 | 0.4927 | 0.0116 |
| Random Forest | Ensamble | 0.3842 | 0.4759 | 0.1852 |

El mejor modelo global fue **MLP + Dropout**, mientras que el mejor ensamble fue **XGBoost**.

---

### 4. Política de umbrales

Se implementó una política operativa basada en la probabilidad máxima predicha:

| Zona | Condición | Acción |
|---|---|---|
| Confianza | `P >= 0.85` | Clasificación automática |
| Incertidumbre | `0.40 <= P < 0.85` | Revisión asistida |
| Rechazo | `P < 0.40` | Descarte automático |

Resultados sobre test:

| Zona | Muestras | Porcentaje | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Confianza | 64 | 13.4 % | 0.6250 | 0.4934 |
| Incertidumbre | 375 | 78.6 % | 0.4960 | 0.4521 |
| Rechazo | 38 | 8.0 % | 0.3947 | 0.3880 |

---

## Aplicación Streamlit

El proyecto incluye una aplicación web informativa desarrollada con Streamlit.

Archivo principal:

```text
app.py
```

Para ejecutarla:

```bash
streamlit run app.py
```

O también:

```bash
python -m streamlit run app.py
```

La app permite:

- Seleccionar el modelo a utilizar.
- Cargar un CSV con 64 columnas Mel.
- Usar el dataset de prueba oficial.
- Obtener predicción de especie.
- Visualizar probabilidades por clase.
- Aplicar la política de umbrales:
  - confianza,
  - incertidumbre,
  - rechazo.

---

## Informe final

El informe técnico se encuentra en:

```text
report/informe_P2.pdf
```

Fuente LaTeX:

```text
report/informe_P2.tex
```

El informe incluye:

- Definición del problema.
- Arquitectura del pipeline.
- PCA y UMAP.
- Clustering con GMM y DBSCAN.
- Clasificación MLP vs modelos de ensamble.
- Política de umbrales.
- Deploy informativo.
- Contribution Statement.

---

## Cómo reproducir el proyecto

### Ejecutar preprocesamiento y reducción dimensional

Abrir y ejecutar:

```text
01_integrante_1_preprocessing_pca_umap.ipynb
```

### Ejecutar clustering

Abrir y ejecutar:

```text
02_integrante_2_clustering_mlp_xgboost.ipynb
```

### Ejecutar clasificación y generación de modelos

```bash
python Parte3Y4.py
```

### Ejecutar la app

```bash
streamlit run app.py
```

---

## Integrantes

- **Marco Esau Apolinario Lainez**: Preprocesamiento, validación del dataset, escalamiento, PCA, UMAP, figuras y tablas base.
- **Diego Roger Illescas Castillo**: Clustering no supervisado con GMM y DBSCAN, selección de hiperparámetros y métricas internas.
- **Lucia Jimena Cartagena Miranda**: Clasificación supervisada, MLP, modelos de ensamble, guardado de modelos, política de umbrales y app Streamlit.

---

## Notas importantes

- Se usa `RANDOM_STATE = 42` para reproducibilidad.
- El escalador se ajusta únicamente sobre el conjunto de desarrollo para evitar data leakage.
- Las características predictoras son exclusivamente `mel_0` a `mel_63`.
- `species_id` se usa como variable objetivo.
- `recording_id`, `songtype_id` e `is_tp` se tratan como metadatos.
- Los modelos finales están guardados en `models/`.
- Los resultados oficiales de clasificación están en `results/clasificacion/`.

---

## Referencias

- Documentación del dataset: `Documentacion_de_dataset.pdf`
- Scikit-learn: https://scikit-learn.org/
- UMAP: https://umap-learn.readthedocs.io/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- Streamlit: https://streamlit.io/

---

**Última actualización:** 2026-07-03  
**Estado del proyecto:** Terminado