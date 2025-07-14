import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#  cargamos datos
archivo = "../datos_clinicos/datos_limpios_sintipos.csv"
df = pd.read_csv(archivo)

print(" Datos cargados:", df.shape)
print(df.head())

#  separamos variables
X = df.drop(columns=["Id_paciente", "Complicacion_binaria"])
y = df["Complicacion_binaria"]

#   hiperparámetros a probar
param_grid = [
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 4},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2},
]

resultados_todos = []

# Función para calcular métricas
def calcular_metricas(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    gmean = np.sqrt(tpr * tnr)
    return acc, f1, tpr, tnr, gmean

#  bucle por cada combinación de hiperparámetros
for i, custom_params in enumerate(param_grid, 1):
    print(f"\n Probando configuración {i}/{len(param_grid)}: {custom_params}")

    # Cross-validation 5-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #guardamos métricas de los 5 folds
    accs, f1s, tprs, tnrs, gmeans = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=custom_params["n_estimators"],
            max_depth=custom_params["max_depth"],
            min_samples_split=custom_params["min_samples_split"],
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc, f1, tpr, tnr, gmean = calcular_metricas(y_test, y_pred)
        accs.append(acc)
        f1s.append(f1)
        tprs.append(tpr)
        tnrs.append(tnr)
        gmeans.append(gmean)

        print(f"  Fold {fold}: Acc={acc:.4f} | F1={f1:.4f} | TPR={tpr:.4f} | TNR={tnr:.4f} | G-Mean={gmean:.4f}")

    # calculamos las medias de los 5 folds
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    mean_tpr = np.mean(tprs)
    mean_tnr = np.mean(tnrs)
    mean_gmean = np.mean(gmeans)

    print(f"\n Resultado promedio para configuración {i}:")
    print(f"   Accuracy={mean_acc:.4f} | F1={mean_f1:.4f} | TPR={mean_tpr:.4f} | TNR={mean_tnr:.4f} | G-Mean={mean_gmean:.4f}")

    # guardamos resultados 
    resultados_todos.append({
        **custom_params,
        "Mean_Accuracy": mean_acc,
        "Mean_F1": mean_f1,
        "Mean_TPR": mean_tpr,
        "Mean_TNR": mean_tnr,
        "Mean_G-Mean": mean_gmean
    })

#  DataFrame final
df_resultados = pd.DataFrame(resultados_todos)

#  guardamos resultados
df_resultados.to_csv("resultados_randomforest_gridsearch_cv5.csv", index=False)
print("\n Todos los resultados guardados en 'resultados_randomforest_gridsearch_cv5.csv'")
