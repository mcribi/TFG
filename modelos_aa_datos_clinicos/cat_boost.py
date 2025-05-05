import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 1. Leer y preparar el CSV
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

# Separación
X = df[features]
y = df[target]

# Columnas categóricas (tal como están en el CSV, sin codificar)
cat_features = ["Sexo", "Tipo_cáncer"]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# ----------------------------
# 2. Modelo CatBoost
# ----------------------------
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="F1",
    cat_features=cat_features,
    verbose=100,
    random_state=42,
    class_weights=[1.0, 2.5]  # más peso a "Con complicación"
)


model.fit(X_train, y_train)

# ----------------------------
# 3. Evaluación
# ----------------------------
y_pred = model.predict(X_test)

print("\n✅ Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Sin complicación", "Con complicación"]))

cm = confusion_matrix(y_test, y_pred)
print("\n✅ Matriz de confusión:")
print(cm)

# ----------------------------
# 4. Guardar matriz como texto
# ----------------------------
with open("matriz_confusion_catboost.txt", "w") as f:
    f.write("Matriz de Confusión (valores absolutos):\n")
    for fila in cm:
        f.write(" ".join(str(x) for x in fila) + "\n")
