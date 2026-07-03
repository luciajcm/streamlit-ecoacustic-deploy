from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

Path("report/figs").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")

COLOR_MAIN = "#EAF3FF"
COLOR_ANALYSIS = "#EAF8EA"
COLOR_MODEL = "#FFF4E6"
COLOR_DEPLOY = "#F3EAFB"
COLOR_EDGE = "#2B2B2B"


def add_box(x, y, text, width=2.3, height=0.85, facecolor="white"):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.12,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=COLOR_EDGE,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.15,
    )


def add_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.3,
        color=COLOR_EDGE,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def add_elbow_arrow(points):
    for i in range(len(points) - 2):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        ax.plot([x1, x2], [y1, y2], color=COLOR_EDGE, linewidth=1.3)

    x1, y1 = points[-2]
    x2, y2 = points[-1]
    add_arrow(x1, y1, x2, y2)


# Título
ax.text(
    8,
    7.45,
    "Arquitectura general del pipeline de clasificación eco-acústica",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
)

# Fila principal
y_main = 6.15
h = 0.85
w = 2.25

add_box(0.5, y_main, "CSV originales\ntrain / test", w, h, COLOR_MAIN)
add_box(3.1, y_main, "Preprocesamiento\nX, y, split, scaler", w, h, COLOR_MAIN)
add_box(5.7, y_main, "Datos procesados\nX escalado + etiquetas", w, h, COLOR_MAIN)
add_box(8.3, y_main, "Modelado supervisado\nMLP + ensambles", w, h, COLOR_MODEL)
add_box(10.9, y_main, "Modelo final\n+ probabilidades", w, h, COLOR_MODEL)

add_arrow(2.75, y_main + h / 2, 3.1, y_main + h / 2)
add_arrow(5.35, y_main + h / 2, 5.7, y_main + h / 2)
add_arrow(7.95, y_main + h / 2, 8.3, y_main + h / 2)
add_arrow(10.55, y_main + h / 2, 10.9, y_main + h / 2)

# Rama exploratoria y clustering
add_box(2.2, 4.05, "Reducción dimensional\nPCA / UMAP", 2.6, h, COLOR_ANALYSIS)
add_box(2.2, 2.65, "Visualización\nestructura geométrica", 2.6, h, COLOR_ANALYSIS)

add_box(5.8, 4.05, "Clustering\nGMM / DBSCAN", 2.6, h, COLOR_ANALYSIS)
add_box(5.8, 2.65, "Métricas internas\nSilhouette, DB, CH", 2.6, h, COLOR_ANALYSIS)

processed_center_x = 5.7 + w / 2
processed_bottom_y = y_main
branch_y = 5.45

# Baja desde "Datos procesados" hacia una línea común
ax.plot(
    [processed_center_x, processed_center_x],
    [processed_bottom_y, branch_y],
    color=COLOR_EDGE,
    linewidth=1.3,
)

# Rama hacia PCA/UMAP
add_elbow_arrow(
    [
        (processed_center_x, branch_y),
        (3.5, branch_y),
        (3.5, 4.9),
    ]
)

# Rama hacia clustering
add_elbow_arrow(
    [
        (processed_center_x, branch_y),
        (7.1, branch_y),
        (7.1, 4.9),
    ]
)

# Flechas internas
add_arrow(3.5, 4.05, 3.5, 3.5)
add_arrow(7.1, 4.05, 7.1, 3.5)

# Rama MLOps / deploy
add_box(
    10.7,
    4.05,
    "Política de umbrales\nconfianza / incertidumbre / rechazo",
    3.0,
    h,
    COLOR_DEPLOY,
)

add_box(
    10.7,
    2.65,
    "App Streamlit\ninferencia informativa",
    3.0,
    h,
    COLOR_DEPLOY,
)

model_center_x = 10.9 + w / 2

add_arrow(model_center_x, y_main, model_center_x, 4.9)
add_arrow(model_center_x, 4.05, model_center_x, 3.5)

# Etiquetas de secciones
ax.text(1.6, 5.65, "Ingesta", ha="center", fontsize=10, fontweight="bold")
ax.text(6.0, 5.65, "Base común", ha="center", fontsize=10, fontweight="bold")
ax.text(9.4, 5.65, "Clasificación", ha="center", fontsize=10, fontweight="bold")
ax.text(12.2, 5.65, "MLOps / Deploy", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("report/figs/fig_arquitectura.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figura creada en: report/figs/fig_arquitectura.png")