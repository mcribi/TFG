import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

model = joblib.load("models/fixed/radiomic_clinical/model_config17358_fold_2.pkl")

data_path = "radiomic_data/cv_clinicos/extended_radiomics_features_resize_tiny_hu_m300_1400_separadas_varthresh_0.01/fold_2/test.csv"

df_test = pd.read_csv(data_path, index_col=0)

target_col = "label_complicacion"
feature_cols = [col for col in df_test.columns if not col.startswith("label_")]

X_test = df_test[feature_cols]
y_test = (df_test[target_col].values == 'S').astype(int)

import shap

explainer = shap.Explainer(model.predict, X_test, max_evals=2000)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.savefig("plots/shap_summary_plot_lighbmg_extended.pdf", dpi=300, bbox_inches='tight')
plt.close()

output_folder = "plots/shap_individual_plots"
shap_values_array = shap_values

# carpeta de salida
os.makedirs(output_folder, exist_ok=True)

y_pred = model.predict(X_test)

# plot por paciente
for i in range(len(X_test)):
    label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
    pred = y_pred[i]

    filename = f"patient_{i}_label_{label}_pred_{pred}.pdf"
    output_path = os.path.join(output_folder, filename)
    
    #waterfall plot
    shap.plots.waterfall(shap_values_array[i], show=False)
    
    # guardamos
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f" Saved {output_path}")