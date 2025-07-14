import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# leer datos del csv y preprocesar
ruta_csv = "datos_filtrados.csv"
df = pd.read_csv(ruta_csv)

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

# cross validate y smote
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
recall_scores = []
conf_matrices = []
feature_importances = []

#  hiperparametros
best_params = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 2.75
}

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        **best_params,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    f1_scores.append(report["1"]["f1-score"])
    recall_scores.append(report["1"]["recall"])
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    feature_importances.append(model.feature_importances_)

    print(f"\n Fold {fold} — F1: {report['1']['f1-score']:.2f}  |  Recall: {report['1']['recall']:.2f}")

# resultados
print("\n Resultados Finales (media de 5 folds)")
print(f"F1 promedio (clase 1):     {np.mean(f1_scores):.3f}")
print(f"Recall promedio (clase 1): {np.mean(recall_scores):.3f}")

total_cm = sum(conf_matrices)
print("\n Matriz de Confusión acumulada:")
print(total_cm)

# importancia de caracteristicas
importances = pd.DataFrame({
    "Característica": features,
    "Importancia media": np.mean(feature_importances, axis=0)
}).sort_values(by="Importancia media", ascending=False)

sns.barplot(data=importances, x="Importancia media", y="Característica")
plt.title("Importancia Media de Características (Modelo Final XGBoost)")
plt.tight_layout()
plt.show()
