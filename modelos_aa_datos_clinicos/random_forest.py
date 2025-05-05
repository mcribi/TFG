import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# ----------------------------
# 1. Leer el CSV
# ----------------------------
ruta_csv = "datos_filtrados.csv" 
df = pd.read_csv(ruta_csv)

# ----------------------------
# 2. Preprocesamiento
# ----------------------------

# Variable objetivo: binaria (1 = Sí, 0 = No)
df["Complicación"] = df["Complicación"].map({"Sí": 1, "No": 0})

# Variables binarias a partir de textos
df["HTA"] = df["Factor_riesgo"].str.contains("HTA", case=False, na=False).astype(int)
df["Enfisema"] = df["Patología pulmonar"].str.contains("Enfisema", case=False, na=False).astype(int)

# Codificar variables categóricas
label_cols = ["Sexo", "Tipo_cáncer", "TC"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Columnas seleccionadas para entrenamiento
features = ["Edad", "Sexo", "Tipo_cáncer", "HTA", "Enfisema"]
X = df[features]
y = df["Complicación"]

# Escalar las variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3. División de datos
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# ----------------------------
# 4. Random Forest
# ----------------------------
#model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)



model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
    #scale_pos_weight=2.0
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ----------------------------
# 5. Evaluación
# ----------------------------
print("\n Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Sin complicación", "Con complicación"]))

# hacemos la matriz de confusion con la libreria sklearn pero sin plot
cm = confusion_matrix(y_test, y_pred)
print("\n Matriz de confusión:")
print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sin complicación", "Con complicación"])
# disp.plot(cmap="Blues")
# plt.title("Matriz de Confusión")
# plt.show()

# ---------


# ----------------------------
# 6. Importancia de características
# ----------------------------
importances = pd.DataFrame({
    "Característica": features,
    "Importancia": model.feature_importances_
}).sort_values(by="Importancia", ascending=False)

sns.barplot(data=importances, x="Importancia", y="Característica")
plt.title("Importancia de Características (Random Forest)")
plt.show()
