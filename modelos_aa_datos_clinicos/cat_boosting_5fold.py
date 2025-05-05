import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold



# ----------------------------
# 1. Leer y preparar los datos
# ----------------------------
df = pd.read_csv("datos_filtrados.csv")

# Etiqueta binaria
df["Complicación"] = df["Complicación"].map({"Sí": 1, "No": 0})

# Variables derivadas de texto
df["HTA"] = df["Factor_riesgo"].str.contains("HTA", case=False, na=False).astype(int)
df["Enfisema"] = df["Patología pulmonar"].str.contains("Enfisema", case=False, na=False).astype(int)

# Variables a usar
features = ["Edad", "Sexo", "Tipo_cáncer", "HTA", "Enfisema"]
target = "Complicación"
X = df[features].copy()
y = df[target]

# Columnas categóricas
cat_features = ["Sexo", "Tipo_cáncer"]

# Limpiar NaNs en categóricas
for col in cat_features:
    X[col] = X[col].fillna("desconocido")

# ----------------------------
# 2. 5-Fold CV estratificado
# ----------------------------
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
f1_scores = []
recall_scores = []
conf_matrices = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="F1",
        class_weights=[1.0, 3.0],  # más peso a la clase minoritaria
        random_state=42,
        verbose=0
    )
    model.fit(train_pool)
    y_pred = model.predict(test_pool)

    report = classification_report(y_test, y_pred, output_dict=True)
    f1_scores.append(report["1"]["f1-score"])
    recall_scores.append(report["1"]["recall"])
    conf_matrices.append(confusion_matrix(y_test, y_pred))

    print(f"\n Fold {fold} — F1: {report['1']['f1-score']:.2f}  |  Recall: {report['1']['recall']:.2f}")

# ----------------------------
# 3. Resultados agregados
# ----------------------------
print("\n Resultados Finales (media de 5 folds)")
print(f"F1 promedio (clase 1):     {np.mean(f1_scores):.3f}")
print(f"Recall promedio (clase 1): {np.mean(recall_scores):.3f}")

# Guardar matriz promedio
total_cm = sum(conf_matrices)
np.savetxt("matriz_confusion_5fold.txt", total_cm.astype(int), fmt="%d")
print("\n Matriz de confusión acumulada guardada en 'matriz_confusion_5fold.txt'")
