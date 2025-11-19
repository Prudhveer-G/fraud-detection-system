import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(filepath):
    return pd.read_csv(filepath)

def train_models(df):
    # Separate features and target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    print("--- Model Performance Report ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Clean formatted accuracy
        print(f"{name}: {acc:.4f}")

    print("--------------------------------")

if __name__ == "__main__":
    df = load_data('data/processed_transactions.csv')
    train_models(df)
