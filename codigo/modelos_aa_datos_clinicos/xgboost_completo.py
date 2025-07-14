import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import xgboost as xgb

# cargamos los datos clinicos
archivo = "../datos_clinicos/datos_limpios_sintipos.csv"
df = pd.read_csv(archivo)

print(" Datos cargados:", df.shape)
print(df.head())

#  separamos variables
X = df.drop(columns=["Id_paciente", "Complicacion_binaria"])
y = df["Complicacion_binaria"]

#  definimos las combinaciones de hiperparámetros a probar
param_grid = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.07},
    {"n_estimators": 250, "max_depth": 10, "learning_rate": 0.05},
    {"n_estimators": 350, "max_depth": 12, "learning_rate": 0.02},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.01},
    {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.02},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.2},
]


#resultados
resultados_todos = []

#  Función para métricas binarias
def calcular_metricas(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    gmean = np.sqrt(tpr * tnr)
    return acc, f1, tpr, tnr, gmean


for i, custom_params in enumerate(param_grid, 1):
    print(f"\nProbar configuración {i}/{len(param_grid)}: {custom_params}")

    # hiperparam comunes
    params = {
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "min_child_weight": 1,
        "scale_pos_weight": 1,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42
    }
    # hiperparm específicos 
    params.update(custom_params)

    # 5-fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # guardamos métricas de los 5 folds
    accs, f1s, tprs, tnrs, gmeans = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc, f1, tpr, tnr, gmean = calcular_metricas(y_test, y_pred)
        accs.append(acc)
        f1s.append(f1)
        tprs.append(tpr)
        tnrs.append(tnr)
        gmeans.append(gmean)

        print(f"  Fold {fold}: Acc={acc:.4f} | F1={f1:.4f} | TPR={tpr:.4f} | TNR={tnr:.4f} | G-Mean={gmean:.4f}")

    # calculamos medias de los 5 folds
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    mean_tpr = np.mean(tprs)
    mean_tnr = np.mean(tnrs)
    mean_gmean = np.mean(gmeans)

    print(f"\n Resultado promedio para configuración {i}:")
    print(f"   Accuracy={mean_acc:.4f} | F1={mean_f1:.4f} | TPR={mean_tpr:.4f} | TNR={mean_tnr:.4f} | G-Mean={mean_gmean:.4f}")

    #rguardamos
    resultados_todos.append({
        **params,
        "Mean_Accuracy": mean_acc,
        "Mean_F1": mean_f1,
        "Mean_TPR": mean_tpr,
        "Mean_TNR": mean_tnr,
        "Mean_G-Mean": mean_gmean
    })

#  a DataFrame final
df_resultados = pd.DataFrame(resultados_todos)

#  guardamos en CSV
df_resultados.to_csv("resultados_xgboost_gridsearch_cv5.csv", index=False)
print("\n Todos los resultados guardados en 'resultados_xgboost_gridsearch_cv5.csv'")
