import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold



def compute_admission_profits(df, admission_col='type_of_admission'):
    grouped = (
        df.groupby(admission_col)
        .agg(
            avg_los=('length_of_stay', 'mean'),
            avg_cost=('total_costs', 'mean'),
            avg_charge=('total_charges', 'mean'),
            std_charge=('total_charges', 'std')  # add std to use later
        )
        .assign(
            cost_per_day=lambda x: x['avg_cost'] / x['avg_los'],
            charge_per_day=lambda x: x['avg_charge'] / x['avg_los']
        )
    )

    # Calculate raw profit
    grouped['profit_per_day'] = grouped['charge_per_day'] - grouped['cost_per_day']

    # Apply your correction rule
    low_profit_mask = grouped['profit_per_day'] < 20
    grouped.loc[low_profit_mask, 'cost_per_day'] = (
        grouped.loc[low_profit_mask, 'charge_per_day'] / 2 +
        grouped.loc[low_profit_mask, 'std_charge'] / grouped.loc[low_profit_mask, 'avg_los']
    )
    # Recalculate profit after patch
    grouped['profit_per_day'] = grouped['charge_per_day'] - grouped['cost_per_day']

    return grouped['profit_per_day'].to_dict()



data1 = pd.read_csv("../../data/bia.csv", low_memory=False)

data1['length_of_stay'] = pd.to_numeric(data1['length_of_stay'], errors='coerce')
data1['total_costs'] = pd.to_numeric(data1['total_costs'], errors='coerce')
data1['total_charges'] = pd.to_numeric(data1['total_charges'], errors='coerce')

# Drop rows with missing or invalid values
data1.dropna(subset=['length_of_stay', 'total_costs', 'total_charges'], inplace=True)

# Generate and save the profit map
profit_map = compute_admission_profits(data1, admission_col='type_of_admission')
joblib.dump(profit_map, "../../code/profit_map.pkl")

print("✅ profit_map.pkl saved.")

df = pd.read_csv("../../code/test.csv")

def oof_target_encoding(df, col, target, n_splits=5):
    df[f"{col}_oof_avg"] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        mapping = train_fold.groupby(col)[target].mean()
        df.loc[df.index[val_idx], f"{col}_oof_avg"] = df.loc[df.index[val_idx], col].map(mapping)
    global_mean = df[target].mean()
    df[f"{col}_oof_avg"].fillna(global_mean, inplace=True)
    return df

# Apply to your training data
df = oof_target_encoding(df, 'apr_drg_code_enc', 'length_of_stay')
df = oof_target_encoding(df, 'apr_mdc_code_enc', 'length_of_stay')

# Save final mappings (after filling)
drg_oof_map = df.groupby('apr_drg_code_enc')['length_of_stay'].mean().to_dict()
mdc_oof_map = df.groupby('apr_mdc_code_enc')['length_of_stay'].mean().to_dict()

# Save them for inference
joblib.dump(drg_oof_map, '../../code/drg_oof_map.pkl')
joblib.dump(mdc_oof_map, '../../code/mdc_oof_map.pkl')