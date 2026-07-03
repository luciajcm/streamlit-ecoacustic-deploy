"""
=============================================================================
APP STREAMLIT - CLASIFICACIÓN ECO-ACOUSTIC
Sistema de Despliegue con Política de Umbrales Operativos
=============================================================================
Ejecutar con: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de la página
st.set_page_config(page_title="Clasificador Eco-Acoustic", page_icon="🐸", layout="wide")

# Forzar tamaño de fuente (Política > 14)
plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
})

# ══════════════════════════════════════════════════════════════
# 1. DEFINICIONES DE ARQUITECTURAS MLP (Requeridas para cargar pesos)
# ══════════════════════════════════════════════════════════════
INPUT_DIM = 64
N_CLASSES = 5

class MLP_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32);         self.fc4 = nn.Linear(32, N_CLASSES)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))

class MLP_Dropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32);         self.fc4 = nn.Linear(32, N_CLASSES)
        self.relu = nn.ReLU(); self.drop = nn.Dropout(p=p)
    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x))); x = self.drop(self.relu(self.fc2(x)))
        return self.fc4(self.drop(self.relu(self.fc3(x))))

class MLP_BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64);        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32);         self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, N_CLASSES);  self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc4(self.bn3(self.relu(self.fc3(self.bn2(self.relu(self.fc2(self.bn1(self.relu(self.fc1(x))))))))))

class MLP_Drop_BN(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64);        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32);         self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, N_CLASSES);  self.relu = nn.ReLU(); self.drop = nn.Dropout(p=p)
    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x))); x = self.bn1(x)
        x = self.drop(self.relu(self.fc2(x))); x = self.bn2(x)
        x = self.drop(self.relu(self.fc3(x))); x = self.bn3(x)
        return self.fc4(x)

class MLP_BN_Drop(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64);        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32);         self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, N_CLASSES);  self.relu = nn.ReLU(); self.drop = nn.Dropout(p=p)
    def forward(self, x):
        x = self.drop(self.bn1(self.relu(self.fc1(x))))
        x = self.drop(self.bn2(self.relu(self.fc2(x))))
        return self.fc4(self.drop(self.bn3(self.relu(self.fc3(x)))))

# Diccionario para instanciar el modelo correcto según el JSON
MLP_CLASSES = {
    "MLP Base (Sin Regularización)": MLP_Base,
    "MLP + Dropout (post-activación)": MLP_Dropout,
    "MLP + BatchNorm (post-activación)": MLP_BatchNorm,
    "MLP + Dropout → BatchNorm": MLP_Drop_BN,
    "MLP + BatchNorm → Dropout": MLP_BN_Drop
}

# ══════════════════════════════════════════════════════════════
# 2. CARGA DE RECURSOS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_resources():
    models_dir = Path("./models")
    data_dir = Path("./data/processed")
    
    with open(models_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    # Cargar Modelos de Ensamble
    models = {
        "XGBoost": joblib.load(models_dir / "xgboost_model.pkl"),
        "LightGBM": joblib.load(models_dir / "lightgbm_model.pkl"),
        "Gradient Boosting": joblib.load(models_dir / "gradient_boosting_model.pkl"),
        "Random Forest": joblib.load(models_dir / "random_forest_model.pkl"),
        "⭐ Mejor Ensamble": joblib.load(models_dir / "best_ensemble_model.pkl"),
    }
    
    # Cargar el Mejor MLP de forma segura usando state_dict
    best_mlp_name = metadata["best_mlp_name"]
    mlp_class = MLP_CLASSES.get(best_mlp_name, MLP_Drop_BN) # Fallback por seguridad
    
    mlp_model = mlp_class()
    # map_location="cpu" evita errores si se entrenó en GPU y se ejecuta en CPU
    state_dict = torch.load(models_dir / "best_mlp_state_dict.pth", map_location="cpu")
    mlp_model.load_state_dict(state_dict)
    mlp_model.eval() # Modo evaluación (desactiva Dropout, etc.)
    
    models["⭐ Mejor MLP"] = mlp_model
    
    return metadata, scaler, models

try:
    metadata, scaler, models = load_resources()
    species_names = metadata["species_names"]
    species_names_str = {str(k): v for k, v in species_names.items()}
    LOAD_SUCCESS = True
except Exception as e:
    st.error(f"❌ Error al cargar los modelos: {e}")
    LOAD_SUCCESS = False

# ══════════════════════════════════════════════════════════════
# 3. LÓGICA DE PREDICCIÓN Y UMBRALES
# ══════════════════════════════════════════════════════════════
def predict_sample(model, sample_scaled, is_mlp=False):
    if is_mlp:
        model.eval()
        with torch.no_grad():
            tensor_input = torch.tensor(sample_scaled, dtype=torch.float32)
            logits = model(tensor_input)
            probs = torch.softmax(logits, dim=1).numpy()
    else:
        probs = model.predict_proba(sample_scaled)
    return probs

def get_threshold_zone(p_max):
    if p_max >= 0.85:
        return "Confianza", "green", "✅ Clasificación Automática", "El sistema despliega una alerta visual verde confirmando la detección automática de la especie con alta fiabilidad."
    elif p_max >= 0.40:
        return "Incertidumbre", "orange", "⚠️ Clasificación Asistida", "El sistema despliega una alerta visual amarilla catalogando el evento como dudoso. Se sugiere el envío a cola de auditoría por un experto humano."
    else:
        return "Rechazo", "red", "❌ Descarte Automático", "El evento es omitido por el software para mitigar el impacto del ruido ambiental."

# ══════════════════════════════════════════════════════════════
# 4. INTERFAZ DE USUARIO (UI)
# ══════════════════════════════════════════════════════════════
if LOAD_SUCCESS:
    st.title("🐸 Sistema de Clasificación Eco-Acoustic")
    st.markdown("**Pipeline de Machine Learning para identificación de especies faunísticas basado en coeficientes MFCC (64 dimensiones)**")
    
    st.sidebar.header("⚙️ Configuración")
    model_choice = st.sidebar.selectbox(
        "Seleccione el Modelo a utilizar:",
        options=list(models.keys()),
        index=list(models.keys()).index("⭐ Mejor Ensamble")
    )
    
    is_mlp = "MLP" in model_choice
    selected_model = models[model_choice]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Modelo Activo:**\n{model_choice}\n\n**Clases:** {metadata['n_classes']} especies")

    st.header("📁 Ingesta de Datos")
    tab1, tab2 = st.tabs(["Subir Archivo CSV", "Usar Dataset de Prueba Oficial"])
    
    df_input = None
    
    with tab1:
        st.markdown("Suba un archivo CSV que contenga las 64 columnas de características MFCC (`mel_0` a `mel_63`).")
        uploaded_file = st.file_uploader("Seleccione su archivo CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.success(f"Archivo cargado correctamente. {len(df_input)} muestras detectadas.")
            except Exception as e:
                st.error(f"Error al leer el CSV: {e}")
                
    with tab2:
        st.markdown("Use el conjunto de prueba oficial del proyecto para validar el sistema.")
        if st.button("Cargar Test Oficial", use_container_width=True):
            test_path = Path("./data/processed/X_test_official.csv")
            if test_path.exists():
                df_input = pd.read_csv(test_path)
                st.success(f"Dataset de prueba cargado. {len(df_input)} muestras.")
            else:
                st.warning("No se encontró `X_test_official.csv` en la carpeta de datos procesados.")

    if df_input is not None:
        expected_cols = [f"mel_{i}" for i in range(64)]
        if not all(col in df_input.columns for col in expected_cols):
            missing = [col for col in expected_cols if col not in df_input.columns]
            st.error(f"Faltan columnas de características en el CSV: {missing[:5]}...")
        else:
            st.header("🔍 Resultados de Clasificación y Moderación")
            
            X_raw = df_input[expected_cols].values
            X_scaled = scaler.transform(X_raw)
            
            probs = predict_sample(selected_model, X_scaled, is_mlp=is_mlp)
            pred_classes = np.argmax(probs, axis=1)
            p_maxs = np.max(probs, axis=1)
            
            results_df = pd.DataFrame({
                "Predicción (ID)": pred_classes,
                "Especie Predicha": [species_names_str.get(str(p), "Desconocida") for p in pred_classes],
                "Probabilidad Máxima (P)": np.round(p_maxs, 4),
                "Zona Operativa": [get_threshold_zone(pm)[0] for pm in p_maxs]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            st.subheader("📊 Detalle por Muestra y Alertas Visuales")
            
            samples_to_show = results_df.index.tolist()[:10] 
            if len(results_df) > 10:
                st.info("Mostrando el detalle de las primeras 10 muestras. La tabla anterior contiene el resultado completo.")
                
            for idx in samples_to_show:
                p_max = p_maxs[idx]
                zone_name, color, title_msg, desc_msg = get_threshold_zone(p_max)
                pred_class = pred_classes[idx]
                pred_species = species_names_str.get(str(pred_class), "Desconocida")
                
                alert_html = f"""
                <div style="background-color: {color}15; border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                    <h4 style="color: {color}; margin-top:0;">{title_msg} - Muestra {idx}</h4>
                    <p style="color: black; margin-bottom:5px;"><b>Especie:</b> {pred_species} (Clase {pred_class}) | <b>Confianza:</b> {p_max*100:.2f}%</p>
                    <p style="color: gray; font-size: 14px; margin-bottom:0;"><i>{desc_msg}</i></p>
                </div>
                """
                st.markdown(alert_html, unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                classes = list(species_names.values())
                y_pos = np.arange(len(classes))
                
                bar_colors = ["#cccccc"] * len(classes)
                bar_colors[pred_class] = color
                
                ax.barh(y_pos, probs[idx], color=bar_colors, edgecolor="black", height=0.6)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(classes, fontsize=14)
                ax.set_xlim(0, 1.1)
                ax.set_xlabel("Probabilidad", fontsize=14)
                ax.set_title(f"Distribución de Probabilidades - Muestra {idx}", fontsize=16, fontweight="bold")
                ax.grid(True, axis="x", alpha=0.3)
                
                st.pyplot(fig, clear_figure=True)
                st.markdown("---")

    else:
        st.info("Por favor, carga un archivo CSV o utiliza el dataset de prueba para comenzar la clasificación.")
        
    st.sidebar.markdown("---")
    st.sidebar.caption("Desarrollado para el proyecto de ML Eco-Acoustic.\nPolíticas de moderación basadas en umbrales (P≥85%, 40%≤P<85%, P<40%).")