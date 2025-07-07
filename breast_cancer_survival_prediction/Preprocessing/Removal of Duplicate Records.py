import pandas as pd
from scipy import stats

file_path = 'variable(67+1)-5.csv'
df = pd.read_csv(file_path)
z_scores = stats.zscore(df.select_dtypes(include=['number']))
z_scores_df = pd.DataFrame(z_scores, columns=df.select_dtypes(include=['number']).columns)
outliers = (abs(z_scores_df) > 3).any(axis=1)
df_cleaned = df[~outliers]
output_file_path = 'variable(67+1)-6.csv'
df_cleaned.to_csv(output_file_path, index=False)
print(f"Outliers removed using Z scores and saved to {output_file_path}")
print(df_cleaned.shape)
