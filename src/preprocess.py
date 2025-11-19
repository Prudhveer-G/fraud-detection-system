import pandas as pd
import numpy as np
import os

def load_data(path="data/transactions.csv"):
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    # Fill missing amounts with 0 (simple baseline assumption)
    if 'amount' in df.columns:
        df['amount'] = df['amount'].fillna(0)

    # Convert timestamp column if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Fill missing categories
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('unknown')

    return df

def feature_engineer(df):
    # Log transform for amount
    if 'amount' in df.columns:
        df['amount_log'] = (df['amount'] + 1).apply(lambda x: np.log(x))

    # Frequency feature based on account_id
    if 'account_id' in df.columns:
        freq = df.groupby('account_id').size().rename('tx_count')
        df = df.merge(freq.reset_index(), on='account_id', how='left')

    return df

def save_processed(df, out_path="data/processed_transactions.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed file to {out_path}")

if __name__ == "__main__":
    df = load_data()
    df = basic_cleaning(df)
    df = feature_engineer(df)
    save_processed(df)
