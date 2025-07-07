import pandas as pd
from scipy.stats import pearsonr, f_oneway

file = 'variables(67+1)-7.csv'
df = pd.read_csv(file)

target_variable = 'Survival_months_graded'
alpha = 0.05
p_values = {}

for column in df.columns:
    if column == target_variable:
        continue

    if pd.api.types.is_numeric_dtype(df[column]):
        non_na_data = df[[column, target_variable]].dropna()
        if not non_na_data.empty:
            _, p_val = pearsonr(non_na_data[column], non_na_data[target_variable])
            p_values[column] = p_val
    elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
        groups = [df[df[column] == level][target_variable].dropna() for level in df[column].dropna().unique()]
        if len(groups) > 1 and all(not group.empty for group in groups):
            _, p_val = f_oneway(*groups)
            p_values[column] = p_val

sorted_p_values = sorted(p_values.items(), key=lambda item: item[1])

print("Variables and their P-values:")
for variable, p_val in sorted_p_values:
    print(f"{variable}: {p_val}")

significant_variables = [variable for variable, p_val in sorted_p_values if p_val < alpha]
print("\nSignificant variables:", significant_variables)
