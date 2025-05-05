import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt

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
# 2. Cross Validation
# ----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Para acumular predicciones
y_true_total = []
y_pred_total = []

# Hiperparámetros óptimos
best_params = {
    'n_estimators': 300,
    'max_depth': 7,
    'criterion': 'gini'
}

# Para cada fold
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    print(f"\n🔄 Fold {fold}/5")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = BalancedRandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_true_total.extend(y_test)
    y_pred_total.extend(y_pred)

# ----------------------------
# 3. Mostrar resultados
# ----------------------------

# Matriz de confusión global
cm = confusion_matrix(y_true_total, y_pred_total)
print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No complicación", "Complicación"])
# disp.plot(cmap="Blues")
# plt.title("Matriz de Confusión (5-fold Cross Validation)")
# plt.show()

# Informe de clasificación global
print("\n📋 Classification Report (5-fold CV):")
print(classification_report(y_true_total, y_pred_total, target_names=["No complicación", "Complicación"]))

# Accuracy global
print(f"✅ Accuracy global: {accuracy_score(y_true_total, y_pred_total):.4f}")
