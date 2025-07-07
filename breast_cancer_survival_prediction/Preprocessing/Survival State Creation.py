import pandas as pd

file_path = 'Breast cancer data_2.2.csv'
df = pd.read_csv(file_path)

category_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

category_df = df[category_cols]
numeric_df = df[numeric_cols]

category_output_file_path = 'Breast cancer data_2.2_categorical_variables.csv'
numeric_output_file_path = 'Breast cancer data_2.2_numeric_variables.csv'

category_df.to_csv(category_output_file_path, index=False)
numeric_df.to_csv(numeric_output_file_path, index=False)

print(f"Categorical variables saved to {category_output_file_path}")
print(f"Numeric variables saved to {numeric_output_file_path}")

file_path = 'Breast cancer data_2.2_categorical_variables.csv'
df = pd.read_csv(file_path)

category_counts = {}
for column in df.columns:
    category_counts[column] = df[column].value_counts()

for column, counts in category_counts.items():
    print(f"Variable: {column}")
    print(counts)
    print("\n")

input_file_path = 'categorical_variables.csv'
output_file_path = 'categorical_variables_1.csv'
df = pd.read_csv(input_file_path)

column_name = 'Survival months'
bins = [-float('inf'), 60, float('inf')]
labels = [0, 1]

df[f'{column_name}_graded'] = pd.cut(df[column_name], bins=bins, labels=labels)
df.to_csv(output_file_path, index=False)

print(f"New CSV file saved as: {output_file_path}")

file_path = 'categorical_variables_1.csv'
df = pd.read_csv(file_path)

column_to_drop = 'Survival months'
df = df.drop(columns=[column_to_drop])

output_file_path = 'categorical_variables_2.csv'
df.to_csv(output_file_path, index=False)

print(f"Column '{column_to_drop}' has been dropped and the updated file saved to {output_file_path}")
