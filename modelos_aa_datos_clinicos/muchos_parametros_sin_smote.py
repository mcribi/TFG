import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
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
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1.0, 1.25, 1.5, 1.75, 2.0]  # técnica de balanceo incorporada
}
grid = list(ParameterGrid(param_grid))

# ----------------------------
# 4. Crear archivo CSV de salida
# ----------------------------
with open("resultados_xgboost_weighted.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "scale_pos_weight", "F1_promedio",
        "Recall_promedio", "Accuracy_promedio"
    ])

    # ----------------------------
    # 5. Evaluar cada combinación
    # ----------------------------
    for i, params in enumerate(grid, 1):
        print(f"\n🔧 Probar combinación {i}/{len(grid)}: {params}")
        f1_scores, recall_scores, acc_scores = [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBClassifier(
                **params,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            f1_scores.append(report["1"]["f1-score"])
            recall_scores.append(report["1"]["recall"])
            acc_scores.append(accuracy_score(y_test, y_pred))

        # Guardar resultados
        writer.writerow([
            params["n_estimators"], params["max_depth"], params["learning_rate"],
            params["subsample"], params["colsample_bytree"], params["scale_pos_weight"],
            round(np.mean(f1_scores), 4), round(np.mean(recall_scores), 4), round(np.mean(acc_scores), 4)
        ])
