# 🩺 Breast Cancer Prediction Model  

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)  
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn&logoColor=white)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)  

---

## 📌 Overview  

This project demonstrates a **Breast Cancer Prediction Model** built using **Machine Learning** techniques.  
Unlike most implementations, the project demonstrates **PCA (Principal Component Analysis)** from scratch for dimensionality reduction.  

After dimensionality reduction, I trained multiple ML models to classify tumors as **malignant or benign**, and compared their performance.  

---

## 📊 Project Highlights  

- 🔬 **Manual PCA Implementation** (retained 7 components → ~92% variance explained)  
- 📉 Reduced dimensionality from **30 → 7 features**  
- 🤖 Applied **SVM, KNN, Random Forest** using Scikit-learn  
- 📑 Evaluated with **classification reports** (Precision, Recall, F1-score)  
- 📒 Full implementation available in Jupyter Notebook  

---

## ⚙️ Workflow  

1. **Data Preprocessing**  
   - Standardized dataset features  
   - Computed covariance matrix, eigenvalues & eigenvectors manually  

2. **Dimensionality Reduction (PCA)**  
   - Selected **7 principal components**  
   - Preserved **92% of variance**  

3. **Model Training**  
   - SVM  
   - K-Nearest Neighbors (KNN)  
   - Random Forest  

4. **Evaluation**  
   - Generated **classification reports**  
   - Compared model performance
---

## 🛠️ Tech Stack  

- 🐍 Python (NumPy, Pandas, Matplotlib)  
- 🤖 Scikit-learn (SVM, KNN, Random Forest)  
- 📒 Jupyter Notebook  

