# Variable Removal
file_path = 'Breast cancer data.csv'
df = pd.read_csv(file_path)
df = df.loc[:, df.nunique() > 1]
output_file_path = 'Breast cancer data_2.1.csv'
df.to_csv(output_file_path, index=False)
print(f"Columns with only one unique value have been removed and saved to {output_file_path}")

# Replace "Blank(s)", "NA" with empty values
file_path = 'Breast cancer data_2.1.csv'
df = pd.read_csv(file_path)
df.replace(["Blank(s)", "NA", "Unknown PRCDA", "Unknown", "Unknown/unstaged", "Unknown/unstaged/unspecified/DCO", "Unstaged", "Not applicable", "Not applicable (cases that do not have an AJCC staging scheme)"], "", inplace=True)
output_file_path = 'Breast cancer data_2.2.csv'
df.to_csv(output_file_path, index=False)
print(f"Cells with 'Blank(s)' etc. have been replaced with empty values and saved to {output_file_path}")

# Process categorical variables with more than 10 categories
file_path = 'categorical_variables-2.csv'
df = pd.read_csv(file_path)
def process_column(col):
    value_counts = col.value_counts()
    if len(value_counts) > 10:
        top_10_values = value_counts.index[:10]
        col = col.apply(lambda x: x if x in top_10_values else 'other')
    return col
df = df.apply(lambda col: process_column(col) if col.dtype == 'object' or col.dtype.name == 'category' else col)
output_file_path = 'categorical_variables-3.csv'
df.to_csv(output_file_path, index=False)
print(f"Processed data has been saved to {output_file_path}")

# Merge variables
input_file = 'Breast cancer data_2.2_numeric_variables(新增6个).csv'
output_file = 'merged_intermediate_variable - EOD 10 - extent (1988-2003)×10.csv'
df = pd.read_csv(input_file)
column = 'EOD 10 - extent (1988-2003)'
new_column = f'EOD 10 - extent (1988-2003)×10'
df[new_column] = df[column].apply(lambda x: x * 10 if pd.notnull(x) else x)
df.to_csv(output_file, index=False)
output_file = 'merged_variable - EOD Primary Tumor.csv'
df['EOD Primary Tumor'] = df['CS extension (2004-2015)'].combine_first(df['EOD Primary Tumor (2018+)'].combine_first(df['EOD 10 - extent (1988-2003)×10']))
df.to_csv(output_file, index=False)
file_path = output_file
df = pd.read_csv(file_path)
missing_values_count = df.isnull().sum()
print("Number of missing values in each column:")
print(missing_values_count)

input_file = 'Breast cancer data_2.2_numeric_variables(新增6个).csv'
output_file = 'merged_variable - EOD Regional Nodes .csv'
df = pd.read_csv(input_file)
df['EOD Regional Nodes'] = df['CS lymph nodes (2004-2015)'].combine_first(df['EOD Regional Nodes (2018+)'])
df.to_csv(output_file, index=False)
file_path = output_file
df = pd.read_csv(file_path)
missing_values_count = df.isnull().sum()
print("Number of missing values in each column:")
print(missing_values_count)

input_file = 'Breast cancer data_2.2_numeric_variables(新增6个).csv'
output_file = 'merged_variable - EOD Mets .csv'
df = pd.read_csv(input_file)
df['EOD Mets'] = df['CS mets at dx (2004-2015)'].combine_first(df['EOD Mets (2018+)'])
df.to_csv(output_file, index=False)
file_path = output_file
df = pd.read_csv(file_path)
missing_values_count = df.isnull().sum()
print("Number of missing values in each column:")
print(missing_values_count)

input_file = 'Breast cancer data_2.2_numeric_variables(新增6个).csv'
output_file = 'merged_variable - Reg Node Eval .csv'
df = pd.read_csv(input_file)
df['Reg Node Eval'] = df['EOD 10 - nodes (1988-2003)'].combine_first(df['CS Reg Node Eval (2004-2015)'])
df.to_csv(output_file, index=False)
file_path = output_file
df = pd.read_csv(file_path)
missing_values_count = df.isnull().sum()
print("Number of missing values in each column:")
print(missing_values_count)

input_file = 'Breast cancer data_2.2_numeric_variables(新增6个).csv'
output_file = 'merged_variable — Tumor Size Recode (1988+).csv'
df = pd.read_csv(input_file)
df['Tumor Size Recode (1988+)'] = df['Tumor Size Over Time Recode (1988+)'].combine_first(df['CS tumor size (2004-2015)']).combine_first(df['Tumor Size Summary (2016+)']).combine_first(df['EOD 10 - size (1988-2003)'])
df.to_csv(output_file, index=False)
file_path = output_file
df = pd.read_csv(file_path)
missing_values_count = df.isnull().sum()
print("Number of missing values in each column:")
print(missing_values_count)

# Remove columns with missing value ratio greater than 50%
data = pd.read_csv('variables_unencoded(88+1).csv')
missing_ratio = data.isnull().sum() / len(data)
threshold = 0.5
data_cleaned = data.loc[:, missing_ratio <= threshold]
data_cleaned.to_csv('variables_unencoded(67+1).csv', index=False)
print("Processing complete, saved to: variables_unencoded(67+1).csv")
print(data_cleaned.shape)

# Remove rows with missing values in the target column
input_file = 'variables_unencoded(67+1).csv'
output_file = 'variables_unencoded(67+1)-1.csv'
df = pd.read_csv(input_file)
df_cleaned = df.dropna(subset=['Survival months_graded'])
df_cleaned.to_csv(output_file, index=False)
print(f"Processing complete, saved to: {output_file}")

# Impute missing values using Missing Forest algorithm
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import pandas as pd
from missingpy import MissForest

df = pd.read_csv('variables(67+1)-4-Label encoding.csv')
categorical_columns = [
    'Year of diagnosis', 'CS Tumor Size/Ext Eval (2004-2015)', 'CS Mets Eval (2004-2015)',
    'CS version input current (2004-2015)', 'Record number recode', 'Year of follow-up recode',
    'SS seq # - mal+ins (most detail)', 'SS seq # 1975+ - mal (most detail)', 'Year of death recode',
    'Reg Node Eval', 'Sex', 'PRCDA 2020', 'Race recode (W, B, AI, API)', 'Origin recode NHIA (Hispanic, Non-Hisp)',
    'TNM 7/CS v0204+ Schema recode', 'SEER Brain and CNS Recode', 'Behavior recode for analysis',
    'Laterality', 'Diagnostic Confirmation', 'ICCC site recode extended 3rd edition/IARC 2017',
    'Combined Summary Stage (1998-2017)', 'Breast - Adjusted AJCC 6th T (1988-2015)', 'RX Summ--Scope Reg LN Sur (2003+)',
    'RX Summ--Surg Oth Reg/Dis (2003+)', 'RX Summ--Surg/Rad Seq', 'Reason no cancer-directed surgery',
    'Radiation recode', 'Chemotherapy recode (yes, no/unk)', 'RX Summ--Systemic/Sur Seq (2007+)',
    'Breast Subtype (2010+)', 'ER Status Recode Breast Cancer (1990+)', 'PR Status Recode Breast Cancer (1990+)',
    'Derived HER2 Recode (2010+)', 'SEER cause-specific death classification', 'SEER other cause of death classification',
    'Survival months flag', 'Vital status recode (study cutoff used)', 'First malignant primary indicator',
    'Primary by international rules', 'IHS Link', 'Site - mal+ins (most detail)', 'Type of Reporting Source',
    'Marital status at diagnosis', 'Rural-Urban Continuum Code', 'AYA site recode 2020 Revision',
    'Site recode ICD-O-3 2023 Revision Expanded', 'Primary Site - labeled', 'ICD-O-3 Hist/behav, malignant',
    'Site recode - rare tumors', 'COD to site recode ICD-O-3 2023 Revision Expanded (1999+)', 'Sequence number',
    'Race/ethnicity', 'Survival months_graded'
]
numeric_columns = [
    'Primary Site', 'Regional nodes examined (1988+)', 'Regional nodes positive (1988+)',
    'RX Summ--Surg Prim Site (1998+)', 'Adjusted CS site-specific factor 7 (2004-2017 varying by schema)',
    'CS site-specific factor 25 (2004-2017 varying by schema)', 'Total number of in situ/malignant tumors for patient',
    'Total number of benign/borderline tumors for patient', 'Time from diagnosis to treatment in days recode',
    'Age recode with single ages and 90', 'Median household income inflation adj to 222($)', 'EOD Primary Tumor',
    'EOD Regional Nodes', 'EOD Mets', 'Tumor Size Recode (1988+)'
]
missing_ratios = df[categorical_columns + numeric_columns].isnull().mean()
columns_to_impute = missing_ratios[(missing_ratios > 0) & (missing_ratios < 1)].index
categorical_to_impute = [col for col in columns_to_impute if col in categorical_columns]
numeric_to_impute = [col for col in columns_to_impute if col in numeric_columns]
df_categorical_to_impute = df[categorical_to_impute]
df_numeric_to_impute = df[numeric_to_impute]
imputer = MissForest()
df_categorical_imputed = df_categorical_to_impute.copy()
df_categorical_imputed[:] = imputer.fit_transform(df_categorical_to_impute)
df_numeric_imputed = df_numeric_to_impute.copy()
df_numeric_imputed[:] = imputer.fit_transform(df_numeric_to_impute)
df[categorical_to_impute] = df_categorical_imputed
df[numeric_to_impute] = df_numeric_imputed
df[categorical_to_impute] = df[categorical_to_impute].round().astype(int)
df[numeric_to_impute] = df[numeric_to_impute].round().astype(int)
df.to_csv('variables(67+1)-4-factor encoding-MF.csv', index=False)
