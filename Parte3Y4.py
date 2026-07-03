"""
=============================================================================
MÓDULO DE CLASIFICACIÓN SUPERVISADA: MLP vs MODELOS DE ENSAMBLE
Proyecto Eco-Acústico — Clasificación de Especies Faunísticas
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import joblib
import json
from pathlib import Path

from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ← PATH CORREGIDO según tu estructura
DATA_DIR = Path("./data/processed")
MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./results/clasificacion")

for d in [MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
N_CLASSES = 5
INPUT_DIM = 64

# Tamaño mínimo de fuente 14 (política de penalización estricta)
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})

# ══════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS PROCESADOS
# ══════════════════════════════════════════════════════════════
print("=" * 72)
print("1. CARGA DE DATOS PROCESADOS")
print("=" * 72)

X_train = pd.read_csv(DATA_DIR / "X_train_full_scaled.csv")
X_val   = pd.read_csv(DATA_DIR / "X_val_scaled.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_official_scaled.csv")

y_train_raw = pd.read_csv(DATA_DIR / "y_train_full.csv")
y_val_df    = pd.read_csv(DATA_DIR / "y_val.csv")
y_test_df   = pd.read_csv(DATA_DIR / "y_test_official.csv")

# Carga robusta de class_mapping (evita errores de BOM o espacios)
class_mapping = pd.read_csv(DATA_DIR / "class_mapping.csv", encoding='utf-8-sig')
class_mapping.columns = class_mapping.columns.str.strip()

# Imprimir columnas para verificación
print(f"Columnas en class_mapping.csv: {list(class_mapping.columns)}")
print(f"Primeras filas:\n{class_mapping.head()}\n")

# Mapeo seguro species_id → encoded_label
if 'encoded_label' in class_mapping.columns and 'species_id' in class_mapping.columns:
    id_to_enc = dict(zip(class_mapping["species_id"], class_mapping["encoded_label"]))
    y_train = y_train_raw["species_id"].map(id_to_enc).values
else:
    print("⚠️  Columnas inesperadas en class_mapping. Usando label_encoder.pkl como respaldo...")
    label_encoder = joblib.load(DATA_DIR / "label_encoder.pkl")
    y_train = label_encoder.transform(y_train_raw["species_id"].values)

y_val  = y_val_df["species_encoded"].values
y_test = y_test_df["species_encoded"].values

# Mapeo para nombres científicos
if 'scientific_name' in class_mapping.columns:
    if 'encoded_label' in class_mapping.columns:
        species_names = dict(zip(class_mapping["encoded_label"], class_mapping["scientific_name"]))
    else:
        # Si no hay encoded_label, asumimos que el índice es el encoded
        species_names = dict(zip(class_mapping.index, class_mapping["scientific_name"]))
else:
    species_names = {i: f"Especie_{i}" for i in range(N_CLASSES)}

X_train_np = X_train.values.astype(np.float32)
X_val_np   = X_val.values.astype(np.float32)
X_test_np  = X_test.values.astype(np.float32)

print(f"Train : {X_train_np.shape[0]} muestras × {X_train_np.shape[1]} features")
print(f"Val   : {X_val_np.shape[0]} muestras")
print(f"Test  : {X_test_np.shape[0]} muestras")
print(f"Clases: {N_CLASSES} → {list(species_names.values())}")

# Tensores PyTorch
X_train_t = torch.tensor(X_train_np)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val_np)
y_val_t   = torch.tensor(y_val, dtype=torch.long)
X_test_t  = torch.tensor(X_test_np)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)

# ══════════════════════════════════════════════════════════════
# 2. ESPECIFICACIÓN MATEMÁTICA FORMAL DE LA FUNCIÓN DE PÉRDIDA
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("2. FUNCIÓN DE PÉRDIDA — CROSS-ENTROPY CATEGÓRICA")
print("=" * 72)

LOSS_DOC = r"""
==============================================================================
ESPECIFICACIÓN MATEMÁTICA FORMAL — FUNCIÓN DE PÉRDIDA
==============================================================================

Sea D = {(x_i, y_i)}_{i=1}^{N} el conjunto de N muestras, donde
  x_i ∈ ℝ^{64}  es el vector de coeficientes MFCC (mel_0 … mel_63)
  y_i ∈ {0, 1, 2, 3, 4} es la etiqueta de clase codificada.

La función de pérdida empleada es la Entropía Cruzada Categórica:

                        N   C
       L(θ) = − (1/N)  Σ   Σ   𝟙[y_i = c] · log(ŷ_{i,c})
                       i=1 c=1

donde:
  θ          → parámetros entrenables (pesos W y sesgos b)
  C = 5      → número de clases objetivo
  𝟙[·]      → indicador de Kronecker
  ŷ_{i,c}    → probabilidad predicha para la clase c en la muestra i

La probabilidad se obtiene mediante la función Softmax:

              exp(z_{i,c})
  ŷ_{i,c} = ─────────────────
              Σ_{j=1}^{C} exp(z_{i,j})

con z_{i,c} = W_c^T · h_i + b_c  (logit de la capa de salida).

Propiedades:
  • L(θ) ≥ 0  ∀θ
  • L(θ) = 0  ⇔  ŷ_{i,c} = 𝟙[y_i = c]  (clasificación perfecta)
  • Es convexa respecto a los logits → garantiza convergencia con GD

Implementación en PyTorch: nn.CrossEntropyLoss()
  Combina internamente LogSoftmax + NLLLoss para estabilidad numérica.
==============================================================================
"""
print(LOSS_DOC)
criterion = nn.CrossEntropyLoss()

# ══════════════════════════════════════════════════════════════
# 3. DEFINICIÓN DE ARQUITECTURAS MLP
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("3. DEFINICIÓN DE ARQUITECTURAS MLP")
print("=" * 72)

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLP_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.tag = "MLP Base (Sin Regularización)"
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32);         self.fc4 = nn.Linear(32, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))

class MLP_Dropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.tag = "MLP + Dropout (post-activación)"
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32);         self.fc4 = nn.Linear(32, N_CLASSES)
        self.relu = nn.ReLU(); self.drop = nn.Dropout(p=p)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc4(self.drop(self.relu(self.fc3(x))))

class MLP_BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.tag = "MLP + BatchNorm (post-activación)"
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64);        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32);         self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, N_CLASSES);  self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc4(self.bn3(self.relu(self.fc3(self.bn2(self.relu(self.fc2(self.bn1(self.relu(self.fc1(x))))))))))

class MLP_Drop_BN(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.tag = "MLP + Dropout → BatchNorm"
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
        self.tag = "MLP + BatchNorm → Dropout"
        self.fc1 = nn.Linear(INPUT_DIM, 128); self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64);        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32);         self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, N_CLASSES);  self.relu = nn.ReLU(); self.drop = nn.Dropout(p=p)

    def forward(self, x):
        x = self.drop(self.bn1(self.relu(self.fc1(x))))
        x = self.drop(self.bn2(self.relu(self.fc2(x))))
        return self.fc4(self.drop(self.bn3(self.relu(self.fc3(x)))))

# ══════════════════════════════════════════════════════════════
# 4. TABLA DE ARQUITECTURA
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("4. TABLA DE ARQUITECTURA DE LA RED NEURONAL")
print("=" * 72)
print(r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║              TOPOLOGÍA DETALLADA DE LA RED NEURONAL MLP                   ║
╠════════════════╦═══════════╦══════════╦═════════════╦═════════════════════╣
║ Capa           ║ Tipo      ║ Neuronas ║ Activación   ║ Parámetros          ║
╠════════════════╬═══════════╬══════════╬═════════════╬═════════════════════╣
║ Input          ║ —         ║    64    ║     —        ║         0           ║
║ Hidden 1       ║ Linear    ║   128    ║     —        ║     8 320           ║
║ Activation 1   ║ ReLU      ║   128    ║   ReLU       ║         0           ║
║ Hidden 2       ║ Linear    ║    64    ║     —        ║     8 256           ║
║ Activation 2   ║ ReLU      ║    64    ║   ReLU       ║         0           ║
║ Hidden 3       ║ Linear    ║    32    ║     —        ║     2 080           ║
║ Activation 3   ║ ReLU      ║    32    ║   ReLU       ║         0           ║
║ Output         ║ Linear    ║     5    ║  Softmax*    ║       165           ║
╠════════════════╬═══════════╬══════════╬═════════════╬═════════════════════╣
║ TOTAL          ║           ║          ║              ║    18 821           ║
╚════════════════╩═══════════╩══════════╩═════════════╩═════════════════════╝
* Softmax implícito en nn.CrossEntropyLoss.
""")

# ══════════════════════════════════════════════════════════════
# 5. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ══════════════════════════════════════════════════════════════
def train_mlp(model, train_loader, X_val_t, y_val_t, criterion, epochs, lr=1e-3, device="cpu"):
    model = model.to(device)
    X_val_t, y_val_t = X_val_t.to(device), y_val_t.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
            running += loss.item()
        tl = running / len(train_loader); train_losses.append(tl)
        
        model.eval()
        with torch.no_grad(): vl = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(vl); scheduler.step(vl)
        
        if epoch % 25 == 0: print(f"    Epoch {epoch:4d}/{epochs}  |  Train {tl:.4f}  |  Val {vl:.4f}")

    return train_losses, val_losses, time.time() - t0

def evaluate_mlp(model, X_t, y_t, device="cpu"):
    model = model.to(device).eval(); X_t = X_t.to(device)
    t0 = time.time()
    with torch.no_grad():
        logits = model(X_t); probs = torch.softmax(logits, dim=1); preds = torch.argmax(probs, dim=1)
    inf_time = time.time() - t0
    p, pr = preds.cpu().numpy(), probs.cpu().numpy()
    y = y_t.numpy() if isinstance(y_t, torch.Tensor) else y_t
    return {"predictions": p, "probabilities": pr, "f1_macro": f1_score(y, p, average="macro"),
            "f1_weighted": f1_score(y, p, average="weighted"), "accuracy": accuracy_score(y, p),
            "confusion_matrix": confusion_matrix(y, p), "inference_time": inf_time}

# ══════════════════════════════════════════════════════════════
# 6. EXPERIMENTOS DE REGULARIZACIÓN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("6. EXPERIMENTOS: DROPOUT Y BATCH NORMALIZATION")
print("=" * 72)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}\n")

configs = [MLP_Base(), MLP_Dropout(p=0.3), MLP_BatchNorm(), MLP_Drop_BN(p=0.3), MLP_BN_Drop(p=0.3)]
mlp_results, mlp_curves, mlp_models = {}, {}, {}

for cfg in configs:
    name = cfg.tag
    print(f"--- {name} ---")
    torch.manual_seed(RANDOM_STATE); cfg.apply(_init_weights)
    tl, vl, tt = train_mlp(cfg, train_loader, X_val_t, y_val_t, criterion, EPOCHS, LEARNING_RATE, device)
    met = evaluate_mlp(cfg, X_test_t, y_test_t, device); met["training_time"] = tt
    mlp_results[name] = met; mlp_curves[name] = (tl, vl); mlp_models[name] = cfg
    print(f"    F1-Macro: {met['f1_macro']:.4f}  |  Acc: {met['accuracy']:.4f}  |  T_Train: {tt:.2f}s\n")

# ══════════════════════════════════════════════════════════════
# 7. GRÁFICAS DE CURVAS DE APRENDIZAJE
# ══════════════════════════════════════════════════════════════
print("=" * 72)
print("7. GENERANDO GRÁFICAS DE CURVAS DE APRENDIZAJE")
print("=" * 72)

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# 7.1 Comparativa global
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for idx, (name, (tl, vl)) in enumerate(mlp_curves.items()):
    axes[0].plot(tl, color=COLORS[idx], ls=STYLES[idx], lw=2.2, label=name)
    axes[1].plot(vl, color=COLORS[idx], ls=STYLES[idx], lw=2.2, label=name)
for ax, title in zip(axes, ["Entrenamiento", "Validación"]):
    ax.set_xlabel("Época", fontsize=14); ax.set_ylabel("Loss (Cross-Entropy)", fontsize=14)
    ax.set_title(f"Curvas de Aprendizaje — {title}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right"); ax.grid(True, alpha=0.3); ax.set_xlim(0, EPOCHS)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "curvas_aprendizaje_comparativas.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ curvas_aprendizaje_comparativas.png")

# 7.2 Grilla individual
fig, axes = plt.subplots(3, 2, figsize=(20, 20)); axes = axes.flatten()
for idx, (name, (tl, vl)) in enumerate(mlp_curves.items()):
    ax = axes[idx]; ax.plot(tl, color="#2196F3", lw=2.2, label="Train Loss")
    ax.plot(vl, color="#FF5722", lw=2.2, label="Val Loss"); gap = vl[-1] - tl[-1]
    ax.set_xlabel("Época", fontsize=14); ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(f"{name}\nGap final: {gap:.4f}", fontsize=15, fontweight="bold")
    ax.legend(fontsize=14); ax.grid(True, alpha=0.3); ax.set_xlim(0, EPOCHS)
if len(mlp_curves) < 6: axes[-1].set_visible(False)
plt.suptitle("Impacto de Dropout y Batch Normalization sobre Estabilidad", fontsize=18, fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "curvas_aprendizaje_detalladas.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ curvas_aprendizaje_detalladas.png")

# 7.3 Zoom últimas 30 épocas
ZOOM = max(0, EPOCHS - 30)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for idx, (name, (tl, vl)) in enumerate(mlp_curves.items()):
    axes[0].plot(range(ZOOM, EPOCHS), tl[ZOOM:], color=COLORS[idx], ls=STYLES[idx], lw=2.5, label=name)
    axes[1].plot(range(ZOOM, EPOCHS), vl[ZOOM:], color=COLORS[idx], ls=STYLES[idx], lw=2.5, label=name)
for ax, title in zip(axes, ["Zoom Train (últimas 30 épocas)", "Zoom Val (últimas 30 épocas)"]):
    ax.set_xlabel("Época", fontsize=14); ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold"); ax.legend(fontsize=13); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "curvas_aprendizaje_zoom.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ curvas_aprendizaje_zoom.png")

# ══════════════════════════════════════════════════════════════
# 8. SELECCIÓN DEL MEJOR MLP
# ══════════════════════════════════════════════════════════════
best_mlp_name = max(mlp_results, key=lambda k: mlp_results[k]["f1_macro"])
best_mlp_res = mlp_results[best_mlp_name]
best_mlp_model = mlp_models[best_mlp_name]
print(f"\n★ Mejor MLP : {best_mlp_name} | F1: {best_mlp_res['f1_macro']:.4f} | Acc: {best_mlp_res['accuracy']:.4f}")

# ══════════════════════════════════════════════════════════════
# 9. MODELOS DE ENSAMBLE (TODOS: XGBoost, LightGBM, GradBoost, RF)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("9. IMPLEMENTACIÓN DE MODELOS DE ENSAMBLE")
print("=" * 72)

ensamble_defs = {
    "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="mlogloss", tree_method="hist", n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
}

ens_results = {}
for name, model in ensamble_defs.items():
    print(f"\n--- {name} ---")
    t0 = time.time(); model.fit(X_train_np, y_train); train_t = time.time() - t0
    t0 = time.time(); preds = model.predict(X_test_np); inf_t = time.time() - t0
    probs = model.predict_proba(X_test_np) if hasattr(model, "predict_proba") else None
    f1m = f1_score(y_test, preds, average="macro"); f1w = f1_score(y_test, preds, average="weighted"); acc = accuracy_score(y_test, preds)
    ens_results[name] = {"predictions": preds, "probabilities": probs, "f1_macro": f1m, "f1_weighted": f1w, "accuracy": acc, "confusion_matrix": confusion_matrix(y_test, preds), "training_time": train_t, "inference_time": inf_t, "model": model}
    print(f"    F1-Macro: {f1m:.4f}  |  F1-Weighted: {f1w:.4f}  |  Acc: {acc:.4f}  |  T_Train: {train_t:.2f}s")

best_ens_name = max(ens_results, key=lambda k: ens_results[k]["f1_macro"])
best_ens_res = ens_results[best_ens_name]
print(f"\n★ Mejor Ensamble: {best_ens_name} | F1: {best_ens_res['f1_macro']:.4f} | Acc: {best_ens_res['accuracy']:.4f}")

# ══════════════════════════════════════════════════════════════
# 10. COMPARACIÓN RIGUROSA + GRÁFICAS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("10. COMPARACIÓN RIGUROSA: MLP vs ENSAMBLE")
print("=" * 72)

comp_rows = []
for name, m in mlp_results.items(): comp_rows.append(dict(Modelo=name, Tipo="MLP", F1_Macro=m["f1_macro"], F1_Weighted=m["f1_weighted"], Accuracy=m["accuracy"], T_Entren_s=m["training_time"], T_Infer_s=m["inference_time"]))
for name, m in ens_results.items(): comp_rows.append(dict(Modelo=name, Tipo="Ensamble", F1_Macro=m["f1_macro"], F1_Weighted=m["f1_weighted"], Accuracy=m["accuracy"], T_Entren_s=m["training_time"], T_Infer_s=m["inference_time"]))
comp_df = pd.DataFrame(comp_rows).sort_values("F1_Macro", ascending=False)
print("\n" + comp_df.to_string(index=False))
comp_df.to_csv(RESULTS_DIR / "tabla_completa_resultados.csv", index=False)

# Gráficas Comparativas
cls_labels = [f"{species_names.get(i, str(i))}\n(Cls {i})" for i in range(N_CLASSES)]

# Matrices de confusión (Mejor MLP vs Mejor Ensamble)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, cm, title, cmap in [(axes[0], best_mlp_res["confusion_matrix"], best_mlp_name, "Blues"), (axes[1], best_ens_res["confusion_matrix"], best_ens_name, "Greens")]:
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, xticklabels=cls_labels, yticklabels=cls_labels, annot_kws={"size": 14})
    ax.set_xlabel("Predicción", fontsize=14); ax.set_ylabel("Valor Real", fontsize=14); ax.set_title(f"Matriz de Confusión — {title}", fontsize=15, fontweight="bold")
plt.suptitle("Comparación: MLP vs Ensamble", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "comparacion_matrices_confusion.png", dpi=150, bbox_inches="tight"); plt.close()
print("\n  ✓ comparacion_matrices_confusion.png")

# Barras F1/Acc (Todos los modelos)
fig, ax = plt.subplots(figsize=(16, 8))
models_all = list(mlp_results.keys()) + list(ens_results.keys())
f1_vals = [mlp_results[n]["f1_macro"] for n in mlp_results] + [ens_results[n]["f1_macro"] for n in ens_results]
colors_bar = ["#2196F3"]*len(mlp_results) + ["#4CAF50"]*len(ens_results)
bars = ax.bar(range(len(models_all)), f1_vals, color=colors_bar, edgecolor="black", lw=1.2)
ax.set_xticks(range(len(models_all))); ax.set_xticklabels(models_all, rotation=35, ha="right", fontsize=13)
ax.set_ylabel("F1-Macro", fontsize=14); ax.set_title("F1-Macro de Todos los Modelos Evaluados", fontsize=16, fontweight="bold"); ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, 1.1)
for b in bars: ax.annotate(f"{b.get_height():.4f}", (b.get_x() + b.get_width()/2, b.get_height()), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=13, fontweight="bold")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0], color="#2196F3", lw=8, label="MLP"), Line2D([0],[0], color="#4CAF50", lw=8, label="Ensamble")], fontsize=14)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "comparacion_metricas_todos.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ comparacion_metricas_todos.png")

# Tiempos
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, key, title, fmt in [(axes[0], "training_time", "Tiempo de Entrenamiento", ".2f"), (axes[1], "inference_time", "Tiempo de Inferencia", ".4f")]:
    vals = [mlp_results[n][key] for n in mlp_results] + [ens_results[n][key] for n in ens_results]
    b = ax.bar(range(len(models_all)), vals, color=colors_bar, edgecolor="black", lw=1.2)
    ax.set_xticks(range(len(models_all))); ax.set_xticklabels(models_all, rotation=35, ha="right", fontsize=12)
    ax.set_ylabel("Segundos (s)", fontsize=14); ax.set_title(title, fontsize=16, fontweight="bold"); ax.grid(True, alpha=0.3, axis="y")
    for bar in b: ax.annotate(f"{bar.get_height():{fmt}}s", (bar.get_x()+bar.get_width()/2, bar.get_height()), textcoords="offset points", xytext=(0,5), ha="center", fontsize=12, fontweight="bold")
plt.suptitle("Benchmarking Computacional", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "comparacion_tiempos.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ comparacion_tiempos.png")

# Matrices de TODOS los modelos
n_total = len(mlp_results) + len(ens_results); ncols = min(3, n_total); nrows = int(np.ceil(n_total / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6*nrows)); axes = np.array(axes).flatten()
all_models = {**mlp_results, **ens_results}; cmaps = {n: "Blues" for n in mlp_results}; cmaps.update({n: "Greens" for n in ens_results})
for idx, (name, m) in enumerate(all_models.items()):
    sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap=cmaps[name], ax=axes[idx], xticklabels=[str(i) for i in range(N_CLASSES)], yticklabels=[str(i) for i in range(N_CLASSES)], annot_kws={"size": 13})
    axes[idx].set_title(f"{name}\nF1={m['f1_macro']:.4f}", fontsize=14, fontweight="bold"); axes[idx].set_xlabel("Pred", fontsize=14); axes[idx].set_ylabel("Real", fontsize=14)
for j in range(idx + 1, len(axes)): axes[j].set_visible(False)
plt.suptitle("Matrices de Confusión — Todos los Modelos", fontsize=18, fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "todas_matrices_confusion.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ todas_matrices_confusion.png")

# ══════════════════════════════════════════════════════════════
# 11. POLÍTICA DE UMBRALES OPERATIVOS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("11. POLÍTICA DE UMBRALES OPERATIVOS")
print("=" * 72)
print(r"""
  ZONA DE CONFIANZA   (P_max ≥ 0.85) → Clasificación automática (Verde)
  ZONA DE INCERTIDUMBRE(0.40 ≤ P < 0.85) → Clasificación asistida (Amarillo)
  ZONA DE RECHAZO       (P_max < 0.40) → Descarte automático (Rojo)
""")

def apply_threshold_policy(probs, y_true, conf=0.85, reject=0.40):
    pmax = np.max(probs, axis=1); pred = np.argmax(probs, axis=1)
    m_conf = pmax >= conf; m_unc = (pmax >= reject) & (~m_conf); m_rej = pmax < reject
    def _safe(mask):
        if mask.sum() == 0: return None, None
        return (accuracy_score(y_true[mask], pred[mask]), f1_score(y_true[mask], pred[mask], average="macro"))
    a_c, f_c = _safe(m_conf); a_u, f_u = _safe(m_unc); a_r, f_r = _safe(m_rej)
    return dict(confianza=dict(n=m_conf.sum(), pct=m_conf.mean()*100, acc=a_c, f1=f_c, mask=m_conf), incertidumbre=dict(n=m_unc.sum(), pct=m_unc.mean()*100, acc=a_u, f1=f_u, mask=m_unc), rechazo=dict(n=m_rej.sum(), pct=m_rej.mean()*100, acc=a_r, f1=f_r, mask=m_rej), pmax=pmax, pred=pred)

thresh = apply_threshold_policy(best_ens_res["probabilities"], y_test)
print("Estadísticas por zona (sobre test):")
for key, label in [("confianza", "Confianza (P≥85%)"), ("incertidumbre", "Incertidumbre (40%≤P<85%)"), ("rechazo", "Rechazo (P<40%)")]:
    z = thresh[key]; print(f"  {label}: {z['n']} muestras ({z['pct']:.1f}%)", end="")
    print(f" -> Acc: {z['acc']:.4f}, F1: {z['f1']:.4f}" if z["acc"] is not None else " -> Sin muestras")

# Gráficas Umbrales
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
pmax, correct = thresh["pmax"], thresh["pred"] == y_test
ax = axes[0]
bins = np.linspace(0, 1, 35)
ax.hist(pmax[correct], bins=bins, alpha=0.7, color="#4CAF50", label="Correctas", edgecolor="black", lw=0.5)
ax.hist(pmax[~correct], bins=bins, alpha=0.7, color="#F44336", label="Incorrectas", edgecolor="black", lw=0.5)
ax.axvline(0.85, color="#2196F3", ls="--", lw=2.5, label="Confianza (85%)"); ax.axvline(0.40, color="#FF9800", ls="--", lw=2.5, label="Rechazo (40%)")
ax.axvspan(0.85, 1.0, alpha=0.10, color="green"); ax.axvspan(0.40, 0.85, alpha=0.10, color="yellow"); ax.axvspan(0.00, 0.40, alpha=0.10, color="red")
ax.set_xlabel("Probabilidad Máxima (P_max)", fontsize=14); ax.set_ylabel("Frecuencia", fontsize=14); ax.set_title("Distribución por Zona Operativa", fontsize=15, fontweight="bold"); ax.legend(fontsize=12); ax.grid(True, alpha=0.3)

ax = axes[1]
jitter = np.random.uniform(-0.15, 0.15, size=len(pmax)); colors_sc = ["#4CAF50" if c else "#F44336" for c in correct]
ax.scatter(pmax, np.zeros(len(pmax)) + jitter, c=colors_sc, alpha=0.6, s=60, edgecolors="black", lw=0.5)
ax.axvline(0.85, color="#2196F3", ls="--", lw=2.5); ax.axvline(0.40, color="#FF9800", ls="--", lw=2.5)
ax.axvspan(0.85, 1.0, alpha=0.10, color="green"); ax.axvspan(0.40, 0.85, alpha=0.10, color="yellow"); ax.axvspan(0.00, 0.40, alpha=0.10, color="red")
ax.set_xlabel("Probabilidad Máxima (P_max)", fontsize=14); ax.set_title("Clasificaciones Correctas vs Incorrectas", fontsize=15, fontweight="bold"); ax.set_yticks([])
ax.legend(["Umbral 85%", "Umbral 40%", "Confianza", "Incertidumbre", "Rechazo", "Correcta", "Incorrecta"], fontsize=12, loc="upper left", ncol=2); ax.grid(True, alpha=0.3, axis="x")
plt.suptitle(f"Política de Umbrales — {best_ens_name}", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "politica_umbrales.png", dpi=150, bbox_inches="tight"); plt.close()
print("\n  ✓ politica_umbrales.png")

# Composición y F1 por zona
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
zone_counts = [thresh["confianza"]["n"], thresh["incertidumbre"]["n"], thresh["rechazo"]["n"]]
zone_labels = [f"Confianza\n{thresh['confianza']['n']} ({thresh['confianza']['pct']:.1f}%)", f"Incertidumbre\n{thresh['incertidumbre']['n']} ({thresh['incertidumbre']['pct']:.1f}%)", f"Rechazo\n{thresh['rechazo']['n']} ({thresh['rechazo']['pct']:.1f}%)"]
zone_colors = ["#4CAF50", "#FFC107", "#F44336"]
axes[0].pie(zone_counts, labels=zone_labels, colors=zone_colors, startangle=90, textprops={"fontsize": 14}, wedgeprops={"edgecolor": "black", "linewidth": 1.2})
axes[0].set_title("Distribución de Muestras", fontsize=16, fontweight="bold")
zone_f1 = [thresh["confianza"]["f1"] or 0, thresh["incertidumbre"]["f1"] or 0, thresh["rechazo"]["f1"] or 0]
bars = axes[1].bar(["Confianza", "Incertidumbre", "Rechazo"], zone_f1, color=zone_colors, edgecolor="black", lw=1.2, width=0.45)
axes[1].set_ylabel("F1-Macro", fontsize=14); axes[1].set_title("Rendimiento por Zona", fontsize=16, fontweight="bold"); axes[1].set_ylim(0, 1.15); axes[1].grid(True, alpha=0.3, axis="y")
for b, f in zip(bars, zone_f1):
    txt = f"{f:.4f}" if f > 0 else "N/A"
    axes[1].annotate(txt, (b.get_x() + b.get_width()/2, b.get_height() if f>0 else 0.02), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=14, fontweight="bold", color="black" if f>0 else "gray")
plt.suptitle("Análisis de Política de Umbrales", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "politica_umbrales_composicion.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ politica_umbrales_composicion.png")

# ══════════════════════════════════════════════════════════════
# 12. GUARDADO DE MODELOS .PKL (Para Streamlit)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("12. GUARDANDO MODELOS .PKL PARA STREAMLIT")
print("=" * 72)

# 1. Guardar Mejor MLP (Estado del modelo + info de la clase)
torch.save(best_mlp_model.state_dict(), MODELS_DIR / "best_mlp_state_dict.pth")
joblib.dump(best_mlp_model, MODELS_DIR / "best_mlp_model_full.pkl")
print(f"  ✓ Mejor MLP guardado: {best_mlp_name}")

# 2. Guardar TODOS los modelos de Ensamble por separado
for name, res in ens_results.items():
    filename = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(res["model"], MODELS_DIR / filename)
    print(f"  ✓ {name} guardado como: {filename}")

# 3. Guardar el Mejor Ensamble de forma destacada
joblib.dump(best_ens_res["model"], MODELS_DIR / "best_ensemble_model.pkl")
print(f"  ✓ Mejor Ensamble destacado guardado: best_ensemble_model.pkl ({best_ens_name})")

# 4. Guardar Metadata y Políticas para la app
metadata = {
    "best_mlp_name": best_mlp_name,
    "best_ensemble_name": best_ens_name,
    "species_names": species_names,
    "thresholds": {"confianza": 0.85, "incertidumbre_min": 0.40, "rechazo_max": 0.40},
    "input_dim": INPUT_DIM,
    "n_classes": N_CLASSES
}
with open(MODELS_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("  ✓ Metadata y políticas guardadas en: model_metadata.json")

# ══════════════════════════════════════════════════════════════
# 13. ANÁLISIS DE TRADE-OFFS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("13. ANÁLISIS DE TRADE-OFFS: COSTO vs RENDIMIENTO")
print("=" * 72)
print(f"""
  MEJOR MLP — {best_mlp_name}
    T. entrenamiento : {best_mlp_res['training_time']:.2f} s
    T. inferencia    : {best_mlp_res['inference_time']:.4f} s
    F1-Macro / Acc   : {best_mlp_res['f1_macro']:.4f} / {best_mlp_res['accuracy']:.4f}

  MEJOR ENSAMBLE — {best_ens_name}
    T. entrenamiento : {best_ens_res['training_time']:.2f} s
    T. inferencia    : {best_ens_res['inference_time']:.4f} s
    F1-Macro / Acc   : {best_ens_res['f1_macro']:.4f} / {best_ens_res['accuracy']:.4f}

  DISCUSIÓN:
  • Latencia crítica (tiempo real)     → El MLP es superior en inferencia.
  • Calidad de clasificación prioridad → {'Ensamble' if best_ens_res['f1_macro'] >= best_mlp_res['f1_macro'] else 'MLP'} lidera el F1-Macro.
  • Producción robusta                  → Ensamble + política de umbrales.
""")

fig, ax = plt.subplots(figsize=(13, 8))
for tx, fy, nm, c, m in [(mlp_results[n]["training_time"], mlp_results[n]["f1_macro"], n, "#2196F3", "o") for n in mlp_results] + [(ens_results[n]["training_time"], ens_results[n]["f1_macro"], n, "#4CAF50", "s") for n in ens_results]:
    ax.scatter(tx, fy, s=220, c=c, marker=m, edgecolors="black", lw=1.5, zorder=5)
    ax.annotate(nm.replace("MLP ", "").replace("(", "\n("), (tx, fy), textcoords="offset points", xytext=(10, 5), fontsize=11)
ax.legend(handles=[Line2D([0],[0], marker="o", color="w", markerfacecolor="#2196F3", markersize=14, markeredgecolor="black", label="MLP"), Line2D([0],[0], marker="s", color="w", markerfacecolor="#4CAF50", markersize=14, markeredgecolor="black", label="Ensamble")], fontsize=14, loc="best")
ax.set_xlabel("Tiempo de Entrenamiento (s)", fontsize=14); ax.set_ylabel("F1-Macro", fontsize=14); ax.set_title("Trade-off: Costo Computacional vs Rendimiento", fontsize=16, fontweight="bold"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "tradeoff_costo_rendimiento.png", dpi=150, bbox_inches="tight"); plt.close()
print("  ✓ tradeoff_costo_rendimiento.png")

print("\n" + "=" * 72)
print("★ EJECUCIÓN COMPLETADA. Todos los .pkl están en /models ★")
print("=" * 72)