# Fraud Detection System (Python + ETL + ML)

A structured data pipeline that processes transaction records, performs automated feature engineering, and trains baseline machine-learning models for fraud detection. The project emphasizes reproducibility, modular code structure, and engineering discipline in data workflows.

## Overview

This repository implements a clean and maintainable pipeline for processing bank transaction data. The workflow includes:

1. Ingesting raw transaction data (synthetic dataset).
2. Applying automated preprocessing: missing-value handling, type cleaning, and timestamp normalization.
3. Engineering statistical and behavioral features such as log-scaled amounts and per-account transaction frequency.
4. Training baseline ML models (Logistic Regression and Decision Tree Classifier).
5. Generating prediction outputs for downstream auditing or inspection.

The focus is on creating a clear, reproducible workflow rather than model sophistication.

---

## Technical Architecture

### Project Structure

```text
fraud-detection-system/
├── data/
│   ├── transactions.csv           # Raw input data
│   └── processed_transactions.csv # Cleaned and feature-engineered data
│
├── output/
│   └── fraud_predictions.csv      # Model inference output
│
├── src/
│   ├── preprocess.py              # ETL and feature engineering
│   └── train.py                   # Baseline ML model training
│
├── fraud_detection_colab.ipynb    # Colab notebook for reproducibility
├── README.md
└── requirements.txt
```

### Data Pipeline Flow

```text
Raw Data (CSV)
    ↓
Preprocessing: null handling, timestamp conversion, schema consistency
    ↓
Feature Engineering: log transforms, frequency features
    ↓
Model Training: Logistic Regression, Decision Tree
    ↓
Inference Output: fraud_predictions.csv
```

### Tech Stack

| Component | Tools |
|----------|-------|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Notebook Runtime | Google Colab |
| Version Control | Git, GitHub |

---

## Key Implementation Details

- **Automated preprocessing** using modular Python scripts inside `src/`, ensuring that the pipeline can be re-run on any consistent CSV input.
- **Feature engineering** includes:
  - `amount_log` to stabilize variance in transaction amounts.
  - `tx_count` to encode per-account activity frequency.
- **Baseline modeling** with Logistic Regression and Decision Tree to establish interpretable benchmarks.
- **Reproducibility**: All steps are scripted; the Colab notebook mirrors the pipeline but does not replace the scripted workflow.

---

## Model Performance

Trained on a synthetic dataset of approximately 3,500 transactions.

- **Logistic Regression:** ~0.78 accuracy  
- **Decision Tree:** ~0.75 accuracy  

Note: This pipeline uses a synthetic dataset for demonstration purposes. Synthetic patterns tend to be cleaner than real-world data and may produce higher accuracy scores during execution (e.g., above 90%). The 75–80% range documented here reflects a conservative real-world baseline rather than synthetic performance.

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Prudhveer-G/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run preprocessing:
   ```bash
   python src/preprocess.py
   ```

4. Train models and generate predictions:
   ```bash
   python src/train.py
   ```

---

## Future Improvements

- Add an unsupervised anomaly-detection module (e.g., Isolation Forest).
- Introduce automated data validation using a framework like Great Expectations.
- Containerize the pipeline for deployment using Docker.
- Include simple visual analytics dashboards for EDA.

---

## Author

**Garlapati Prudhveer Reddy**  
GitHub: https://github.com/Prudhveer-G  
LinkedIn: https://www.linkedin.com/in/garlapati-prudhveer-reddy-13401121a
