# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# pipeline = joblib.load("models/fixed/radiomic/model_config11228_fold_4.pkl")
# print(pipeline)
# tree_model = pipeline.named_steps['classifier']
# data_path = "radiomic_data/cv/extended_radiomics_features_resize_smaller_hu_m300_1400_separadas_pca_95/fold_4"

# df_train = pd.read_csv(os.path.join(data_path, "train.csv"), index_col=0)
# df_test = pd.read_csv(os.path.join(data_path, "test.csv"), index_col=0)

# target_col = "label_complicacion"
# feature_cols = [col for col in df_train.columns if not col.startswith("label_")]

# X_train = df_train[feature_cols]
# X_train = X_train.reset_index(drop=True)


# print(X_train.index)
# print(tree_model.n_features_in_)
# y_train = (df_train[target_col].values == 'S').astype(int)

# import numpy as np

# X_array = X_train[feature_cols].to_numpy(dtype=np.float32)


# from dtreeviz import dtreeviz

# viz = dtreeviz(
#     tree_model,
#     X_array,           
#     y_train,
#     target_name='complicacion',
#     feature_names=feature_cols,
#     class_names=['No complicacion', 'Complicacion']
# )

# viz.save("arbol_decision_viz.svg")
# viz.view()


import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

pipeline = joblib.load("models/fixed/radiomic/model_config11228_fold_4.pkl")
print(pipeline)
tree_model = pipeline.named_steps['classifier']
data_path = "radiomic_data/cv/extended_radiomics_features_resize_smaller_hu_m300_1400_separadas_pca_95/fold_4"

df_train = pd.read_csv(os.path.join(data_path, "train.csv"), index_col=0)
df_test = pd.read_csv(os.path.join(data_path, "test.csv"), index_col=0)

target_col = "label_complicacion"
feature_cols = [col for col in df_train.columns if not col.startswith("label_")]

X_train = df_train[feature_cols]
X_train = X_train.reset_index(drop=True)

from sklearn import tree
import graphviz

# exportamos el árbol a formato DOT
dot_data = tree.export_graphviz(
    tree_model, 
    out_file=None, 
    feature_names=feature_cols, 
    class_names=['No complicacion', 'Complicacion'], 
    filled=True, 
    rounded=True, 
    special_characters=True
) 

# gráfico
graph = graphviz.Source(dot_data)  
graph.render("arbol_decision", format='pdf', cleanup=True)
