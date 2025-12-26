## Fraud Detection System

---

### Overview

This project implements a structured data preparation and baseline machine learning workflow for transaction fraud detection. The primary focus is on building a clear, reproducible data pipeline rather than optimizing model performance.

---

### Dataset

- Approximately 3,500 transaction records  
- Dataset is synthetic and intended for controlled experimentation  
- Used to validate data cleaning, feature engineering, and modeling workflows  

---

### Technical Architecture & Data Flow

The project follows a staged data pipeline:

1. Raw transaction data ingestion  
2. Data cleaning and validation  
3. Feature engineering  
4. Model training and evaluation  
5. Output of performance metrics  

Each stage operates on explicit inputs and outputs to keep the workflow modular and repeatable.

---

### Models Used

- Logistic Regression  
- Decision Tree  

These models are used as baselines to validate the data pipeline and feature preparation logic.

---

### Model Performance

The dataset used is synthetic, resulting in cleaner patterns than real-world transaction data.

Baseline results:
- Logistic Regression: ~78% accuracy  
- Decision Tree: ~75% accuracy  

These results reflect performance on controlled synthetic data and are not representative of real-world fraud detection accuracy.

---

### Tech Stack

- Python  
- Pandas  
- NumPy  
- scikit-learn  

---

### Notes

This project emphasizes data preparation discipline, pipeline structure, and reproducibility over advanced modeling or production deployment.
