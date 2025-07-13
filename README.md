## Features List
['apr_drg_code_oof_avg',
 'apr_severity_of_illness',
 'patient_disposition_Home or Self Care',
 'patient_disposition_Skilled Nursing Home',
 'apr_risk_of_mortality',
 'apr_mdc_code_oof_avg',
 'age_group',
 'Disease_Mental Health',
 'Black/African American',
 'is_rare_drg',
 'patient_disposition_Left Against Medical Advice',
 'disease_burden',
 'Admission_Emergency',
 'White',
 'apr_drg_code_enc',
 'gender',
 'Disease_Infectious Diseases',
 'patient_disposition_Inpatient Rehabilitation Facility',
 'patient_disposition_Home w/ Home Health Services',
 'Admission_Newborn',
 'Admission_Elective',
 'Admission_Urgent']

 ```python
 import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv('bia_data.csv')

# Helper function: OOF Target Encoding
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

# ----------------------------
# 1. OOF Target Encoded Features
# ----------------------------
data = oof_target_encoding(data, 'apr_drg_code', 'length_of_stay')
data = oof_target_encoding(data, 'apr_mdc_code', 'length_of_stay')

# ----------------------------
# 2. APR severity & risk (already ordinal) #USER INPUT
# ----------------------------

data['apr_severity_of_illness'] = 1
data['apr_risk_of_mortality'] = 1

# ----------------------------
# 3. Patient Disposition # User Input
# ---------------------------

# ----------------------------
# 4. Race Encoding # USER INPUT

# ----------------------------
# 5. Gender and Age Group # USER INPUT

# ----------------------------
# 6. Disease Categories (one-hot) # USER INPUT
# --------------------------

# ----------------------------
# 7. Admission Type One-Hot # USER INPUT
# ----------------------------

# ----------------------------
# 8. Other Engineered Features
# ----------------------------
# Rare DRG: Based on frequency threshold
drg_counts = data['apr_drg_code'].value_counts()
rare_drg_set = set(drg_counts[drg_counts < 1200].index)
data['is_rare_drg'] = data['apr_drg_code'].isin(rare_drg_set).astype(int)


data['disease_burden'] = data[
    [col for col in data.columns if col.startswith('Disease_')]
].sum(axis=1)

def smoothed_target_encode(df, col, target, min_samples=100, smoothing=10):
    averages = df.groupby(col)[target].agg(['mean', 'count'])
    global_mean = df[target].mean()
    smoothing_factor = 1 / (1 + np.exp(-(averages['count'] - min_samples) / smoothing))
    smooth_mean = global_mean * (1 - smoothing_factor) + averages['mean'] * smoothing_factor
    return df[col].map(smooth_mean)


data['apr_drg_code_enc'] = smoothed_target_encode(data, 'apr_drg_code', 'length_of_stay')

# Final feature list
final_features = [
    'apr_drg_code_oof_avg',
    'apr_severity_of_illness',
    'patient_disposition_Home or Self Care',
    'patient_disposition_Skilled Nursing Home',
    'apr_risk_of_mortality',
    'apr_mdc_code_oof_avg',
    'age_group',
    'Disease_Mental Health',
    'Black/African American',
    'is_rare_drg',
    'patient_disposition_Left Against Medical Advice',
    'disease_burden',
    'Admission_Emergency',
    'White',
    'apr_drg_code_enc',
    'gender',
    'Disease_Infectious Diseases',
    'patient_disposition_Inpatient Rehabilitation Facility',
    'patient_disposition_Home w/ Home Health Services',
    'Admission_Newborn',
    'Admission_Elective',
    'Admission_Urgent'
]

# X = data[final_features]
# y = data['length_of_stay']
```


 