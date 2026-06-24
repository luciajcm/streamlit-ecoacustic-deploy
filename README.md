# Proyecto 2: Clasificación de Especies usando Ecoacústica

## Descripción General

Este proyecto se enfoca en la **clasificación de especies silvestres** utilizando características de ecoacústica (análisis de grabaciones de sonido). El objetivo es predecir qué especie produce un determinado sonido basándose en características mel-espectrograma y metadatos de audio.

### Especies Clasificadas
- **Leptodactylus discodactylus** (Species ID: 10)
- **Osteocephalus taurinus** (Species ID: 12)
- **Chiroxiphia lineata** (Species ID: 17)
- **Saltator grossus** (Species ID: 18)
- **Pheucticus chrysopeplus** (Species ID: 23)

---

## Estructura del Proyecto

```
Proyecto_2/
├── 01_integrante_1_preprocessing_pca_umap.ipynb  # Notebook principal: preprocesamiento y DR
├── eco_acoustic_train.csv                         # Dataset de entrenamiento
├── eco_acoustic_test.csv                          # Dataset de prueba oficial
├── requirements.txt                               # Dependencias del proyecto
├── Documentacion_de_dataset.pdf                  # Documentación del dataset
│
├── data/                                          # Datos procesados
│   └── processed/
│       ├── X_dev.csv                              # Características dev (sin escalar)
│       ├── X_dev_scaled.csv                       # Características dev (escaladas)
│       ├── X_val.csv                              # Características val (sin escalar)
│       ├── X_val_scaled.csv                       # Características val (escaladas)
│       ├── X_test_official.csv                    # Características test (sin escalar)
│       ├── X_test_official_scaled.csv             # Características test (escaladas)
│       ├── y_dev.csv                              # Etiquetas dev
│       ├── y_val.csv                              # Etiquetas val
│       ├── y_test_official.csv                    # Etiquetas test
│       ├── recording_dev.csv                      # IDs de grabación dev
│       ├── recording_val.csv                      # IDs de grabación val
│       ├── recording_test_official.csv            # IDs de grabación test
│       ├── class_mapping.csv                      # Mapeo de clases encoded → species_id
│       └── pca_umap_coordinates.csv               # Coordenadas para visualización
│
├── models/                                        # Modelos y escalers guardados
│   ├── scaler.pkl                                 # StandardScaler entrenado
│   └── label_encoder.pkl                          # LabelEncoder para clases
│
├── outputs/
│   ├── figures/                                   # Gráficos generados
│   │   └── [figuras de visualización]
│   └── tables/
│       ├── class_distribution.csv                 # Distribución de clases
│       └── dimensionality_reduction_results.csv   # Resultados de reducción
│
└── README.md                                      # Este archivo

```

---

## Configuración y Instalación

### Requisitos
- Python 3.x
- Librerías especificadas en `requirements.txt`

### Instalación

1. **Crear un entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencias Principales
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `scikit-learn` - Machine learning (PCA, escalado, codificación)
- `matplotlib` - Visualización
- `umap-learn` - Reducción dimensional (UMAP)
- `joblib` - Serialización de modelos

---

## Flujo de Procesamiento

### 1. **Carga de Datos**
- Se cargan dos archivos CSV principales:
  - `eco_acoustic_train.csv`: Datos de entrenamiento (~80% del dataset original)
  - `eco_acoustic_test.csv`: Datos de prueba oficial

### 2. **Validación del Dataset**
- Verificación de columnas esperadas:
  - Metadatos: `recording_id`, `songtype_id`, `is_tp`
  - Características: 64 bandas mel (`mel_0` a `mel_63`)
  - Etiqueta: `species_id`
- Chequeo de valores faltantes y duplicados
- Validación de distribución de clases

### 3. **Separación de Datos**
- Uso de `train_test_split` con stratification en `species_id`
- División: 80% dev (entrenamiento) + 20% val (validación)
- Test official: conjunto separado de prueba

**Distribuciones resultantes:**
- **X_dev**: Características para entrenamiento
- **X_val**: Características para validación
- **X_test_official**: Características para prueba oficial
- Correspondientes **y_*** con las etiquetas

### 4. **Escalamiento de Características**
- `StandardScaler` ajustado en `X_dev`
- Se aplica a dev, val y test para consistencia
- Genera versiones tanto escaladas como sin escalar

### 5. **Codificación de Etiquetas**
- `LabelEncoder` transforma `species_id` a valores numéricos (0-4)
- Se guarda el mapeo en `class_mapping.csv` para interpretación posterior
- Incluye nombres científicos de las especies

### 6. **Reducción Dimensional**
Se exploran dos métodos:

**PCA (Principal Component Analysis)**
- PCA 2D: Para visualización bidimensional
- PCA 3D: Para visualización tridimensional
- Se calcula:
  - Tiempo de ejecución
  - Varianza explicada
  - Trustworthiness (calidad de la reducción)

**UMAP (Uniform Manifold Approximation and Projection)**
- Método no lineal de reducción dimensional
- Típicamente preserva mejor la estructura local que PCA

---

## Archivos Generados

### Datos Procesados (`data/processed/`)

| Archivo | Descripción |
|---------|-------------|
| `X_dev.csv` | 64 características mel para dev (original) |
| `X_dev_scaled.csv` | 64 características mel para dev (escaladas) |
| `X_val.csv` | 64 características mel para val (original) |
| `X_val_scaled.csv` | 64 características mel para val (escaladas) |
| `X_test_official.csv` | 64 características mel para test (original) |
| `X_test_official_scaled.csv` | 64 características mel para test (escaladas) |
| `y_dev.csv` | Etiquetas para dev (species_id y encoded) |
| `y_val.csv` | Etiquetas para val (species_id y encoded) |
| `y_test_official.csv` | Etiquetas para test (species_id y encoded) |
| `recording_dev.csv` | Mapeo de recording_id → split para dev |
| `recording_val.csv` | Mapeo de recording_id → split para val |
| `recording_test_official.csv` | Mapeo de recording_id → split para test |
| `class_mapping.csv` | Mapeo: encoded_label → species_id → scientific_name |
| `pca_umap_coordinates.csv` | Coordenadas para visualización 2D y 3D |

### Modelos Guardados (`models/`)

| Archivo | Descripción |
|---------|-------------|
| `scaler.pkl` | StandardScaler entrenado en X_dev |
| `label_encoder.pkl` | LabelEncoder para mapeo species_id |

### Salidas Tabulares (`outputs/tables/`)

| Archivo | Descripción |
|---------|-------------|
| `class_distribution.csv` | Distribución de clases en train/test |
| `dimensionality_reduction_results.csv` | Resultados de PCA y UMAP |

### Visualizaciones (`outputs/figures/`)
- Gráficos de distribución de clases
- Visualizaciones de reducción dimensional (PCA 2D/3D, UMAP)

---

## Cómo Ejecutar

### Ejecutar el Notebook Completo

1. Abrir VS Code o Jupyter:
   ```bash
   # Opción 1: Con VS Code (recomendado)
   code .
   # Luego abrir: 01_integrante_1_preprocessing_pca_umap.ipynb

   # Opción 2: Con Jupyter
   jupyter notebook
   ```

2. Ejecutar las celdas en orden o ejecutar todo:
   - `Shift + Enter` para ejecutar una celda
   - `Ctrl + Shift + Alt + Enter` para ejecutar todo

### Procesamiento Esperado

El notebook ejecutará secuencialmente:
1. Importación de librerías
2. Creación de carpetas necesarias
3. Carga y validación del dataset
4. Separación train/dev/val/test
5. Escalamiento de características
6. Codificación de etiquetas
7. Guardado de archivos procesados
8. Análisis de distribución de clases
9. Aplicación de PCA 2D/3D
10. Aplicación de UMAP (si está disponible)
11. Generación de visualizaciones

---

## Próximos Pasos

Este proyecto continuará con:
- [ ] Desarrollo de modelos de clasificación (SVM, Random Forest, XGBoost, etc.)
- [ ] Validación cruzada y ajuste de hiperparámetros
- [ ] Análisis de importancia de características
- [ ] Evaluación de métricas (precisión, recall, F1-score, AUC-ROC)
- [ ] Comparación de modelos
- [ ] Análisis de errores y casos difíciles
- [ ] Reporte final de resultados

---

## Integrantes

- **Marco Apolinario**: Preprocesamiento y Análisis de Dimensionalidad (PCA, UMAP)

---

## Notas Importantes

- El `RANDOM_STATE = 42` se utiliza para reproducibilidad
- El escalador se ajusta **solo en X_dev** para evitar data leakage
- Las etiquetas se codifican usando `LabelEncoder` (0-4) para entrenamiento
- Se preserva el mapeo original (species_id) en `class_mapping.csv`
- Las características originales (sin escalar) también se guardan para referencia

---

## Referencias

- Documentación del dataset: Ver `Documentacion_de_dataset.pdf`
- Scikit-learn: https://scikit-learn.org/
- UMAP: https://umap-learn.readthedocs.io/
- Pandas: https://pandas.pydata.org/

---

**Última actualización:** 2026-06-24  
**Estado del proyecto:** En construcción
