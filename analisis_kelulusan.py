"""Analisis prediksi kelulusan tepat waktu mahasiswa.

Letakkan dataset_mahasiswa.csv pada folder yang sama, lalu jalankan:
    pip install pandas numpy matplotlib scikit-learn
    python analisis_kelulusan.py

Output grafik dan tabel tersimpan dalam folder output_ml/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_mahasiswa.csv"
OUTPUT_DIR = BASE_DIR / "output_ml"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Memuat data
df = pd.read_csv(DATA_PATH)
target = "Lulus_Tepat_Waktu"
fitur_numerik = ["IPK", "Kehadiran", "Jam_Belajar"]
fitur_kategorik = [
    "Organisasi", "Penghasilan_Ortu", "Jenis_Kelamin", "Status_Beasiswa"
]
X = df[fitur_numerik + fitur_kategorik].copy()
y = (df[target] == "Ya").astype(int)  # Ya=1, Tidak=0

print("Ukuran dataset:", df.shape)
print("\nMissing value:\n", df.isna().sum())
print("\nDistribusi target:\n", df[target].value_counts())

# 2. Train-test split 80:20 dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Kompatibel dengan scikit-learn versi lama dan baru
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("numerik", SimpleImputer(strategy="median"), fitur_numerik),
    ("kategori", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ]), fitur_kategorik),
])

model_defs = {
    "Decision Tree depth 3": DecisionTreeClassifier(
        max_depth=3, random_state=RANDOM_STATE
    ),
    "Decision Tree penuh": DecisionTreeClassifier(
        max_depth=None, random_state=RANDOM_STATE
    ),
    "Gaussian Naive Bayes": GaussianNB(),
}

# 3. Pelatihan dan evaluasi
pipelines, predictions, results = {}, {}, []
for nama, model in model_defs.items():
    pipe = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    pipelines[nama] = pipe
    predictions[nama] = pred
    results.append({
        "Model": nama,
        "Akurasi": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "Akurasi Latih": accuracy_score(y_train, pipe.predict(X_train)),
    })

hasil = pd.DataFrame(results).set_index("Model")
print("\nHasil evaluasi:\n", hasil.round(4))
hasil.to_csv(OUTPUT_DIR / "hasil_evaluasi.csv")

# 4. Confusion matrix tiga model
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (nama, pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Tidak", "Ya"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(nama)
fig.suptitle("Confusion Matrix – Perbandingan 3 Model")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 5. Bar chart metrik
metric_cols = ["Akurasi", "Precision", "Recall", "F1 Score"]
ax = hasil[metric_cols].T.plot(kind="bar", figsize=(10, 6), ylim=(0, 1.08))
ax.set_title("Perbandingan Metrik Evaluasi 3 Model")
ax.set_xlabel("")
ax.set_ylabel("Nilai")
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "perbandingan_metrik.png", dpi=200, bbox_inches="tight")
plt.close()

# 6. Visualisasi Decision Tree max_depth=3
pipe_dt3 = pipelines["Decision Tree depth 3"]
feature_names = pipe_dt3.named_steps["preprocessor"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(
    pipe_dt3.named_steps["model"],
    feature_names=feature_names,
    class_names=["Tidak", "Ya"],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision Tree (max_depth=3)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "decision_tree_depth3.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 7. Lima data salah prediksi Decision Tree depth 3
pred_dt3 = predictions["Decision Tree depth 3"]
pos_salah = np.flatnonzero(pred_dt3 != y_test.to_numpy())[:5]
error = X_test.iloc[pos_salah].copy()
error.insert(0, "Baris_CSV", X_test.index[pos_salah] + 2)  # header dihitung
error["Aktual"] = np.where(y_test.iloc[pos_salah].to_numpy() == 1, "Ya", "Tidak")
error["Prediksi"] = np.where(pred_dt3[pos_salah] == 1, "Ya", "Tidak")
error.to_csv(OUTPUT_DIR / "lima_data_salah_prediksi.csv", index=False)
print("\nLima data salah prediksi:\n", error)

# 8. Simulasi IPK tinggi dan kehadiran rendah
simulasi = pd.DataFrame([{
    "IPK": 3.80,
    "Kehadiran": 65.0,
    "Jam_Belajar": float(X_train["Jam_Belajar"].median()),
    "Organisasi": X_train["Organisasi"].mode()[0],
    "Penghasilan_Ortu": X_train["Penghasilan_Ortu"].mode()[0],
    "Jenis_Kelamin": "P",
    "Status_Beasiswa": X_train["Status_Beasiswa"].mode()[0],
}])
sim_results = []
for nama, pipe in pipelines.items():
    pred = int(pipe.predict(simulasi)[0])
    p_ya = float(pipe.predict_proba(simulasi)[0, 1])
    sim_results.append({"Model": nama, "Prediksi": "Ya" if pred else "Tidak", "Peluang_Ya": p_ya})
pd.DataFrame(sim_results).to_csv(OUTPUT_DIR / "hasil_simulasi.csv", index=False)
print("\nHasil simulasi:\n", pd.DataFrame(sim_results).round(4))

# Ringkasan machine-readable
ringkasan = {
    "baris": int(len(df)),
    "train": int(len(X_train)),
    "test": int(len(X_test)),
    "target_ya": int((df[target] == "Ya").sum()),
    "target_tidak": int((df[target] == "Tidak").sum()),
    "hasil": hasil.reset_index().to_dict(orient="records"),
    "simulasi": sim_results,
}
with open(OUTPUT_DIR / "ringkasan.json", "w", encoding="utf-8") as f:
    json.dump(ringkasan, f, indent=2, ensure_ascii=False)

print(f"\nSelesai. Semua output tersimpan di: {OUTPUT_DIR}")
