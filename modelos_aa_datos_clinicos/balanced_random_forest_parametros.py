import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
import csv

# ----------------------------
# 1. Cargar y preparar datos
# ----------------------------
df = pd.read_csv("datos_filtrados.csv")

df["Complicación"] = df["Complicación"].map({"Sí": 1, "No": 0})
df["HTA"] = df["Factor_riesgo"].str.contains("HTA", case=False, na=False).astype(int)
df["Enfisema"] = df["Patología pulmonar"].str.contains("Enfisema", case=False, na=False).astype(int)
df["Tabaquismo"] = df["Factor_riesgo"].str.contains("Tabaquismo", case=False, na=False).astype(int)
df["Diabetes"] = df["Factor_riesgo"].str.contains("DM", case=False, na=False).astype(int)

for col in ["Sexo", "Tipo_cáncer"]:
    df[col] = LabelEncoder().fit_transform(df[col])

features = ["Edad", "Sexo", "Tipo_cáncer", "HTA", "Enfisema", "Tabaquismo", "Diabetes"]
X = df[features]
y = df["Complicación"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 2. Validación cruzada
# ----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# 3. Grid de parámetros
# ----------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'criterion': ['gini', 'entropy']
}
grid = list(ParameterGrid(param_grid))

# ----------------------------
# 4. Archivo para guardar resultados
# ----------------------------
with open("resultados_balanced_random_forest.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "n_estimators", "max_depth", "criterion",
        "F1_promedio", "Recall_promedio", "Accuracy_promedio"
    ])

    # ----------------------------
    # 5. Evaluar cada combinación
    # ----------------------------
    for i, params in enumerate(grid, 1):
        print(f"\n🌲 Probar combinación {i}/{len(grid)}: {params}")
        f1_scores, recall_scores, acc_scores = [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = BalancedRandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            f1_scores.append(report["1"]["f1-score"])
            recall_scores.append(report["1"]["recall"])
            acc_scores.append(accuracy_score(y_test, y_pred))

        # Guardar resultados en el CSV
        writer.writerow([
            params["n_estimators"], params["max_depth"], params["criterion"],
            round(np.mean(f1_scores), 4),
            round(np.mean(recall_scores), 4),
            round(np.mean(acc_scores), 4)
        ])
