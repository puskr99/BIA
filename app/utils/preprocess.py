import joblib
import pandas as pd


def compute_admission_profits(df, admission_col='type_of_admission'):
    grouped = (
        df.groupby(admission_col)
        .agg(
            avg_los=('length_of_stay', 'mean'),
            avg_cost=('total_costs', 'mean'),
            avg_charge=('total_charges', 'mean')
        )
        .assign(
            cost_per_day=lambda x: x['avg_cost'] / x['avg_los'],
            charge_per_day=lambda x: x['avg_charge'] / x['avg_los'],
            profit_per_day=lambda x: x['charge_per_day'] - x['cost_per_day']
        )
    )
    return grouped['profit_per_day'].to_dict()


data = pd.read_csv("../../data/bia.csv", low_memory=False)

data['length_of_stay'] = pd.to_numeric(data['length_of_stay'], errors='coerce')
data['total_costs'] = pd.to_numeric(data['total_costs'], errors='coerce')
data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')

# Drop rows with missing or invalid values
data.dropna(subset=['length_of_stay', 'total_costs', 'total_charges'], inplace=True)

# Generate and save the profit map
profit_map = compute_admission_profits(data, admission_col='type_of_admission')
joblib.dump(profit_map, "../../code/profit_map.pkl")

print("✅ profit_map.pkl saved.")