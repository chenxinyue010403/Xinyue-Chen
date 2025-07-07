import pandas as pd
from sklearn.utils import resample

file_path = 'variables_unencoded_67_1_1.csv'
data = pd.read_csv(file_path)
target_column = 'Survival_months_graded'
majority_class = data[data[target_column] == data[target_column].value_counts().idxmax()]
minority_class = data[data[target_column] != data[target_column].value_counts().idxmax()]
majority_class_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
balanced_data = pd.concat([majority_class_downsampled, minority_class]).sample(frac=1, random_state=42).reset_index(drop=True)
balanced_file_path = 'variables_unencoded_67_1_2.csv'
balanced_data.to_csv(balanced_file_path, index=False)
print(f"Balanced dataset saved to {balanced_file_path}")
print(balanced_data.shape)
