"""
Script de Preprocesamiento de datos tabulares clínicos
- Eliminar columnas innecesarias
- Procesar columnas multietiqueta
- Normalizar edad con MinMaxScaler
- Convertir variable objetivo a binaria
- Guardar resultado en CSV limpio
"""

# librerias necesarias
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

#config
INPUT_CSV = "./../../datos_clinicos/datos_clinicos.csv"
OUTPUT_CSV = "./../../datos_clinicos/datos_clinicos_limpio_j.csv"

# cargamos datos 
train = pd.read_csv(INPUT_CSV, na_values="NaN", sep=",")
print(f"Datos cargados. Shape inicial: {train.shape}")

# eliminamos el tipo de cancer ya que vamos a hacer 
if "Tipo de cáncer" in train.columns:
    train.drop(columns=["Tipo de cáncer"], inplace=True)
    print("Columna 'Tipo de cáncer' eliminada.")

# funcion que preprocesa columnas multietiqueta
def procesar_columna_multietiqueta(df, columna, nombre_sin=None, reemplazos=None):
    df = df.copy()
    if reemplazos:
        for k, v in reemplazos.items():
            df[columna] = df[columna].str.replace(k, v)

    # convertimos los nulos o X en etiqueta explicita
    df[columna] = df[columna].fillna("X").replace("X", nombre_sin).str.strip()
    df[columna] = df[columna].str.replace(r"\s*,\s*", ",", regex=True)

    # separamos las etiquetas
    etiquetas_ordenadas = df[columna].apply(
        lambda x: sorted(set(x.split(","))) if x else [nombre_sin]
    )

    mlb = MultiLabelBinarizer()
    etiquetas_bin = pd.DataFrame(
        mlb.fit_transform(etiquetas_ordenadas),
        columns=[col.replace(" ", "_") for col in mlb.classes_],
        index=df.index
    )

    return pd.concat([df.drop(columns=[columna]), etiquetas_bin], axis=1), mlb.classes_


# procesamos columnas multietiqueta
if "Tipo de complicación" in train.columns:
    train, etiquetas_comp = procesar_columna_multietiqueta(
        train, "Tipo de complicación", nombre_sin="Sin_complicación"
    )
    print(f" Columna 'Tipo de complicación' procesada. Clases: {list(etiquetas_comp)}")

if "Factor de riesgo" in train.columns:
    train, etiquetas_riesgo = procesar_columna_multietiqueta(
        train, "Factor de riesgo", nombre_sin="Sin_factor_de_riesgo"
    )
    print(f" Columna 'Factor de riesgo' procesada. Clases: {list(etiquetas_riesgo)}")

if "Patología pulmonar" in train.columns:
    train, etiquetas_pato = procesar_columna_multietiqueta(
        train, "Patología pulmonar", nombre_sin="Sin_patología_pulmonar"
    )
    print(f" Columna 'Patología pulmonar' procesada. Clases: {list(etiquetas_pato)}")

# normalizamos la edad
if "Edad" in train.columns:
    scaler = MinMaxScaler()
    train["Edad"] = scaler.fit_transform(train[["Edad"]])
    print(" Columna 'Edad' normalizada con MinMaxScaler.")

# convertimos la variable Sexo a binaria
if "Sexo" in train.columns:
    train["Sexo_binaria"] = train["Sexo"].map({"Hombre": 1, "Mujer": 0})
    train.drop(columns=["Sexo"], inplace=True)
    print(" Columna 'Sexo' convertida a binaria.")


# convertimos la variable objetivo Complicación a binaria (es nuestra etiqueta de clasif)
if "Complicación" in train.columns:
    train["Complicacion_binaria"] = train["Complicación"].map({"Sí": 1, "No": 0})
    train.drop(columns=["Complicación"], inplace=True)
    print(" Columna 'Complicación' convertida a binaria.")

#resultado
print("\n Primeras filas del DataFrame procesado:")
print(train.head())

# guardamos csv
train.to_csv(OUTPUT_CSV, index=False)
print(f"\n DataFrame procesado guardado en: {OUTPUT_CSV}")
