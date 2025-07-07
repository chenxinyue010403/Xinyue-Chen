import pandas as pd

input_file_path = 'variable(67+1)-6.csv'
df = pd.read_csv(input_file_path)
duplicate_rows = df[df.duplicated()]
print("Duplicates:")
print(duplicate_rows)
df_cleaned = df.drop_duplicates()
output_file_path = 'variable(67+1)-7.csv'
df_cleaned.to_csv(output_file_path, index=False)
print(f"Duplicates removed and saved to {output_file_path}")
print(df_cleaned.shape)
