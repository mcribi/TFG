# 🫁 TFG: Predicción de complicaciones en biopsias pulmonares con IA

Este repositorio contiene el desarrollo completo del Trabajo de Fin de Grado (TFG) de **María Cribillés Pérez**, dirigido por **Francisco Herrera Triguero** y **Juan Luis Suárez Díaz**, en el marco del doble grado en Ingeniería Informática y Matemáticas en la Universidad de Granada.

🔗 **[Página de resultados con visualizaciones interactivas](https://mariacribilles.github.io/TFG/)**

---

## 🧠 Resumen del proyecto

El objetivo de este TFG es desarrollar un sistema predictivo capaz de estimar si una biopsia pulmonar guiada por tomografía computarizada (TC) tendrá o no complicaciones, utilizando técnicas de inteligencia artificial aplicadas a imágenes médicas 3D y datos clínicos tabulares. Se proponen diferentes enfoques basados en Deep Learning, Radiómica y modelos híbridos, y se analizan mediante validación cruzada y técnicas de explicabilidad.

---

## 📖 Estructura de la memoria del TFG

### 🧮 Parte teórica

- **Procesamiento de imágenes y señales**: fundamentos sobre discretización, filtros y convoluciones.
- **Transformada de Fourier**: su papel en el análisis de frecuencias de imágenes médicas.
- **Radiómica**: extracción cuantitativa de características desde imágenes.
- **Optimización y aprendizaje automático**: descenso por gradiente, clasificación y funciones de pérdida.
- **Deep Learning**: redes convolucionales 2D/3D, preentrenamiento y transferencia.
- **Distance Metric Learning**: aprendizaje de distancias con LMNN y NCA.

### 🧪 Parte aplicada

#### Capítulo 6 — Planteamiento
Introducción clínica, definición del problema, descripción del dataset y contexto ético/legal.

#### Capítulo 7 — Preprocesado
- Datos volumétricos: normalización, segmentación con TotalSegmentator, resize y máscaras.
- Datos clínicos: limpieza, codificación, imputación y escalado.

#### Capítulo 8 — Modelos DL 2D/3D
- Arquitecturas como DenseNet121 y ResNet3D.
- Validación cruzada estratificada (5-fold).
- Fusión multimodal de imagen + datos clínicos.

#### Capítulo 9 — Radiómica y ML clásico
- Extracción con PyRadiomics.
- Modelos clásicos: Random Forest, XGBoost, KNN.
- Aprendizaje de métricas (LMNN, NCA) y fusión con datos clínicos.

#### Capítulo 10 — Resultados experimentales
- Comparativa entre enfoques: DL puro, híbridos, y radiómicos.
- Métricas: Accuracy, F1, TPR, TNR, G-Mean.
- Tablas con resultados por fold y análisis detallado.

#### Capítulo 11 — Explicabilidad (XAI)
- Visualización con **Grad-CAM** para modelos 3D.
- Interpretabilidad con **SHAP** para modelos tabulares y radiómicos.

#### Capítulo 12 — Conclusiones
- Análisis crítico de los resultados.
- Limitaciones del dataset.
- Líneas futuras: aumentar datos, mejorar segmentación, generalización multimodal.

---

## 📂 Estructura del repositorio
- codigo/: scripts de entrenamiento, validación, preprocesado y visualización de modelos deep learning, radiómicos y multimodales. Contiene el núcleo del sistema predictivo.

- defensa/: materiales utilizados para la defensa del TFG, como presentaciones, figuras y recursos visuales.

- memoria/latex/: código fuente completo en LaTeX de la memoria escrita del TFG, incluyendo figuras, tablas y bibliografía.

- resultados/: resultados obtenidos durante los experimentos, organizados en carpetas por tipo de modelo (DL3D, multimodal, radiómico, etc.). Incluye métricas, visualizaciones SHAP, mapas Grad-CAM y tablas HTML.

- index.html: página principal que carga el sitio web generado con GitHub Pages, mostrando los resultados interactivos.

- README.md: este archivo, que documenta el contenido y propósito del repositorio.

- .gitignore: archivo que especifica qué archivos/directorios deben ser ignorados por Git.


---

## 🖥️ Página de resultados (GitHub Pages)

Puedes explorar visualizaciones, métricas, gráficas y resultados detallados de los experimentos en la siguiente página:  
📊 **[https://mariacribilles.github.io/TFG/](https://mariacribilles.github.io/TFG/)**

---

## 📌 Tecnologías utilizadas

- 🧠 **Deep Learning**: PyTorch, MONAI
- 📊 **Machine Learning clásico**: scikit-learn, XGBoost, LightGBM
- 📈 **Radiómica**: PyRadiomics
- 🫁 **Segmentación**: TotalSegmentator
- 🎯 **Visualización y XAI**: SHAP, Grad-CAM, Matplotlib, Seaborn


## Resumen

La biopsia pulmonar guiada por tomografía computarizada (TC) es un procedimiento diagnóstico esencial para caracterizar nódulos pulmonares y determinar la presencia de neoplasias. Sin embargo, no está exenta de riesgos, presentando complicaciones como hemorragias o neumotórax en un porcentaje significativo de casos. Aunque existen numerosos estudios centrados en la clasificación de la benignidad o malignidad de los nódulos, apenas hay investigaciones que analicen la probabilidad de complicaciones antes de realizar la biopsia. Esta carencia motiva la necesidad de herramientas predictivas que permitan anticipar el riesgo y optimizar la selección de pacientes.

El presente trabajo propone el desarrollo de un sistema predictivo basado en técnicas de radiómica y aprendizaje profundo para estimar el riesgo de complicaciones en biopsias pulmonares guiadas por TC. Para sustentar el diseño del modelo, se estudian en detalle los fundamentos matemáticos necesarios, incluyendo el procesamiento de señales médicas, teoría de convolución, teoría de radiómica y los conceptos teóricos del aprendizaje automático y profundo. 

La metodología incluye el preprocesamiento de imágenes volumétricas con segmentación pulmonar y normalización de intensidades, la extracción de características radiómicas, el uso de redes neuronales convolucionales 3D y la integración de datos clínicos tabulares para construir modelos multimodales. Se emplean estrategias como el preentrenamiento (transfer learning), la validación cruzada estratificada y el análisis de interpretabilidad (Grad-CAM, SHAP) para garantizar robustez y facilitar la validación clínica.

Los resultados obtenidos muestran que, aunque la idea es prometedora, los modelos de aprendizaje profundo sobre imágenes 3D presentaron limitaciones para generalizar de forma sólida, probablemente debido al tamaño reducido y la heterogeneidad del conjunto de datos. Por el contrario, los enfoques clásicos de radiómica ofrecieron resultados más estables. Este trabajo representa así un primer paso en una línea de investigación novedosa, destacando la necesidad de recopilar más datos y refinar estrategias para mejorar la capacidad predictiva en futuros estudios.

**Palabras clave**: Biopsia pulmonar, Tomografía computarizada, Aprendizaje profundo, Inteligencia Artificial, Radiómica, Redes neuronales convolucionales, Predicción de complicaciones, Segmentación pulmonar, Datos clínicos.

## Summary
### Problem Description

Lung cancer remains the leading cause of cancer-related mortality worldwide, responsible for over 1.8 million deaths annually. Despite advances in low-dose CT screening enabling earlier detection of lesions, five-year survival rates remain limited due to late diagnoses in advanced stages. To confirm suspicion and characterize tumor subtype, CT-guided lung biopsy is essential. While minimally invasive, this procedure carries inherent risks, the most common being pneumothorax and pulmonary hemorrhage, with incidence rates of up to 22\% and 7\%, respectively. The severity of these complications varies from mild to severe and depends on multiple factors, including lesion location and size, needle path length, pulmonary parenchyma structure, and operator experience.

Currently, risk estimation prior to biopsy relies primarily on the subjective judgment of the interventional radiologist, who qualitatively assesses imaging and patient characteristics. There are no standardized clinical tools or quantitative predictive models that provide personalized, pre-procedural risk estimates. This gap limits the ability to plan preventive measures, tailor procedural techniques, or consider alternative diagnostic strategies for high-risk cases. Investigating the feasibility of developing a system to predict complication risk in lung biopsies using clinical and imaging data is therefore especially relevant.

This project frames the problem as a binary classification task aimed at predicting whether a patient will experience a complication after biopsy, combining structured clinical data and volumetric CT imaging. The goal is to equip clinicians with an objective, personalized risk assessment tool to improve patient safety and optimize medical resources. Beyond its immediate clinical relevance, this research is highly innovative, as there is virtually no prior work in the literature specifically addressing complication prediction in lung biopsies using AI techniques. This absence of references poses additional challenges, such as designing preprocessing, modeling, and validation strategies from scratch, but it also underscores the importance and potential impact of the proposal.

### Mathematical Framework

Developing an AI-based predictive system for anticipating complications in lung biopsies requires a solid theoretical foundation that blends mathematical and computational principles. From a mathematical perspective, medical images can be studied as functions carrying complex anatomical information. Transforms, such as the Fourier transform, enable analysis of the frequency components of these signals, facilitating filtering and enhancement of relevant features. Convolution operations are fundamental for image processing, allowing hierarchical extraction of local patterns. This concept underpins convolutional neural networks, which automatically learn these filters during training to identify discriminative features.

Radiomics leverages these mathematical principles to extract quantitative features from medical images. Using first-order statistics and texture metrics derived from co-occurrence matrices, radiomics generates numerical descriptors that summarize imaging information, capturing subtle patterns potentially linked to higher complication risk. This systematic feature extraction aims to overcome the inherent subjectivity of human visual assessment.

Mathematical optimization is another essential pillar in training machine learning models. The training process is formulated as the minimization of a loss function quantifying the discrepancy between model predictions and actual outcomes. Methods such as gradient descent and its stochastic variants allow efficient adjustment of millions of parameters. The backpropagation algorithm enables efficient calculation of partial derivatives, supporting iterative weight updates in the network.

Supervised machine learning provides the framework for addressing complication risk prediction as a binary classification problem. This approach requires labeled data indicating whether a complication occurred following biopsy and uses these examples to learn a model that generalizes to new cases. Evaluation metrics are particularly important in clinical contexts with significant class imbalance. Accuracy can be misleading when the majority class dominates, so more informative measures like the F1-score are used, along with sensitivity and specificity to assess detection of positives and negatives separately, and the G-Mean, which combines both to evaluate performance in imbalanced scenarios.

###  Practical Approach and Experimentation

This study relied on two main types of data: volumetric chest CT scans and structured clinical data for each patient. For the CT volumes, extensive preprocessing was performed, including intensity normalization to Hounsfield Units using a lung window. This step limited and rescaled intensities to highlight the pulmonary parenchyma, removing extreme values from irrelevant structures like bone or extrapulmonary air. Additionally, lung segmentation was performed using the TotalSegmentator tool to generate precise masks, effectively isolating the anatomical region of interest and reducing input noise. The segmented and normalized volumes were then resized to consistent dimensions to ensure sample homogeneity and facilitate processing in 3D convolutional networks.

Clinical tabular data also required careful cleaning and normalization. The original dataset contained heterogeneous variables, both categorical and numerical, with incomplete or inconsistent records. The data were curated through imputation or removal of missing values, categorical encoding, and numerical scaling. This process ensured a harmonized clinical dataset ready for machine learning, supporting integration with imaging-derived features in subsequent experiments.

A key preprocessing step was lung segmentation, performed with TotalSegmentator to produce accurate lung masks in each CT volume. This allowed effective isolation of the target anatomy while reducing unnecessary noise and variability in the model.

For the modeling phase, a 3D-adapted DenseNet121 was implemented and trained using MONAI and PyTorch. Stratified five-fold cross-validation served as the primary validation method to assess model performance robustly. To improve generalization with limited data, a strategy of pretraining on similar tasks followed by fine-tuning on the specific problem set was applied. This approach enabled transfer of previously acquired knowledge and its adaptation to predicting complication risk in lung biopsies.

In parallel, an alternative radiomics-based approach was developed. Quantitative features were extracted from the segmented volumes using PyRadiomics, including statistical and textural descriptors of the pulmonary parenchyma. These features were combined with patient clinical data and used as inputs for traditional machine learning models such as LightGBM. This pipeline enabled comparison of direct deep learning with a more classical feature-extraction approach.

During experimentation, multimodal integration techniques were applied to combine clinical data with imaging-derived information, enriching the system's predictive context. Additionally, interpretability tools such as Grad-CAM were used to visualize CT regions most influential in deep model predictions, while SHAP values analyzed the impact of each clinical or radiomic variable in classic models. This interpretability was crucial to validate results and ensure potential clinical applicability.

Experiments showed progressive performance improvements thanks to careful preprocessing, precise segmentation, and the use of pretraining and fine-tuning strategies. However, purely 3D deep learning models exhibited notable limitations, with less stable and generalizable results likely due to the task’s complexity and the small dataset size. In contrast, radiomics-based strategies involving systematic feature extraction and analysis via classic machine learning and deep metric learning delivered more consistent, robust results. These findings suggest that while the developed system represents an important first step toward personalized risk prediction, there remains substantial room for improvement in future work.

