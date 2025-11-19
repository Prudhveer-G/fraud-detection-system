import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_processed(path="data/processed_transactions.csv"):
    return pd.read_csv(path)

def basic_train(df, label_col='is_fraud'):
    # Select only numeric columns for training
    X = df.select_dtypes(include=['int64', 'float64']).drop(columns=[label_col], errors='ignore')

    if label_col not in df.columns:
        raise ValueError("Column 'is_fraud' not found in data.")

    y = df[label_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train.fillna(0), y_train)
    y_pred_lr = lr.predict(X_test.fillna(0))
    acc_lr = accuracy_score(y_test, y_pred_lr)
    models['logistic_regression'] = (lr, acc_lr)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=6)
    dt.fit(X_train.fillna(0), y_train)
    y_pred_dt = dt.predict(X_test.fillna(0))
    acc_dt = accuracy_score(y_test, y_pred_dt)
    models['decision_tree'] = (dt, acc_dt)

    # Save predictions
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    result = X_test.copy()
    result['true_label'] = y_test.values
    result['pred_lr'] = y_pred_lr

    output_path = os.path.join(out_dir, "fraud_predictions.csv")
    pd.DataFrame(result).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    return models

if __name__ == "__main__":
    df = load_processed()
    models = basic_train(df)
    print("Accuracies:")
    for name, (model, acc) in models.items():
        print(name, acc)
