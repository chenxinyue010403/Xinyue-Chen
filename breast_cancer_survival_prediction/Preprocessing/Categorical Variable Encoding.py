# Ordinal encoding
variable_names_string = "Year of diagnosis,CS Tumor Size/Ext Eval (2004-2015),CS Mets Eval (2004-2015),CS version input current (2004-2015),Record number recode,Year of follow-up recode,SS seq # - mal+ins (most detail),SS seq # 1975+ - mal (most detail),Year of death recode,Reg Node Eval,SEER Brain and CNS Recode,Rural-Urban Continuum Code,Sequence number"
variable_list = [var.strip() for var in variable_names_string.split(',')]
print(variable_list)

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import pickle

df = pd.read_csv('variables_unencoded(67+1)-4.csv')
ordinal_columns = ['Year of diagnosis', 'CS Tumor Size/Ext Eval (2004-2015)', 'CS Mets Eval (2004-2015)',
                   'CS version input current (2004-2015)', 'Record number recode', 'Year of follow-up recode',
                   'SS seq # - mal+ins (most detail)', 'SS seq # 1975+ - mal (most detail)', 'Year of death recode',
                   'Reg Node Eval', 'SEER Brain and CNS Recode', 'Rural-Urban Continuum Code', 'Sequence number']

ordinal_encoder = OrdinalEncoder()
ordinal_mappings = {}
for column in ordinal_columns:
    non_null_data = df[[column]].dropna()
    encoded_values = ordinal_encoder.fit_transform(non_null_data)
    df.loc[non_null_data.index, column] = encoded_values
    ordinal_mappings[column] = ordinal_encoder.categories_[0]
with open('ordinal_mapping.pkl', 'wb') as f:
    pickle.dump(ordinal_mappings, f)
df.to_csv('variables(67+1)-4-ordinal_encoding.csv', index=False)
print("Ordinal encoded columns and their mappings:")
for column, mapping in ordinal_mappings.items():
    print(f"{column}: {list(mapping)}")
print("Encoding complete and saved to file.")

# Label encoding
variable_names_string = "Sex,PRCDA 2020,Origin recode NHIA (Hispanic, Non-Hisp),Chemotherapy recode (yes, no/unk),Vital status recode (study cutoff used),First malignant primary indicator,Primary by international rules,IHS Link,Survival months_graded,Race recode (W, B, AI, API),TNM 7/CS v0204+ Schema recode,Behavior recode for analysis,Laterality,Diagnostic Confirmation,ICCC site recode extended 3rd edition/IARC 2017,Combined Summary Stage (1998-2017),Breast - Adjusted AJCC 6th T (1988-2015),RX Summ--Scope Reg LN Sur (2003+),RX Summ--Surg Oth Reg/Dis (2003+),RX Summ--Surg/Rad Seq,Reason no cancer-directed surgery,Radiation recode,RX Summ--Systemic/Sur Seq (2007+),Breast Subtype (2010+),ER Status Recode Breast Cancer (1990+),PR Status Recode Breast Cancer (1990+),Derived HER2 Recode (2010+),SEER cause-specific death classification,SEER other cause of death classification,Survival months flag,Site - mal+ins (most detail),Type of Reporting Source,Marital status at diagnosis,AYA site recode 2020 Revision,Site recode ICD-O-3 2023 Revision Expanded,Primary Site - labeled,ICD-O-3 Hist/behav, malignant,Site recode - rare tumors,COD to site recode ICD-O-3 2023 Revision Expanded (1999+),Race/ethnicity"
variable_list = [var.strip() for var in variable_names_string.split(',')]
print(variable_list)

from sklearn.preprocessing import LabelEncoder

# Read CSV file
df = pd.read_csv('variables(67+1)-4-ordinal_encoding.csv')

# Specify columns for label encoding
label_columns = ['Sex', 'PRCDA 2020', 'Origin recode NHIA (Hispanic, Non-Hisp)', 'Chemotherapy recode (yes, no/unk)',
                 'Vital status recode (study cutoff used)', 'First malignant primary indicator',
                 'Primary by international rules', 'IHS Link', 'Survival months_graded', 'Race recode (W, B, AI, API)',
                 'TNM 7/CS v0204+ Schema recode', 'Behavior recode for analysis', 'Laterality',
                 'Diagnostic Confirmation', 'ICCC site recode extended 3rd edition/IARC 2017',
                 'Combined Summary Stage (1998-2017)', 'Breast - Adjusted AJCC 6th T (1988-2015)',
                 'RX Summ--Scope Reg LN Sur (2003+)', 'RX Summ--Surg Oth Reg/Dis (2003+)', 'RX Summ--Surg/Rad Seq',
                 'Reason no cancer-directed surgery', 'Radiation recode', 'RX Summ--Systemic/Sur Seq (2007+)',
                 'Breast Subtype (2010+)', 'ER Status Recode Breast Cancer (1990+)',
                 'PR Status Recode Breast Cancer (1990+)', 'Derived HER2 Recode (2010+)',
                 'SEER cause-specific death classification', 'SEER other cause of death classification',
                 'Survival months flag', 'Site - mal+ins (most detail)', 'Type of Reporting Source',
                 'Marital status at diagnosis', 'AYA site recode 2020 Revision',
                 'Site recode ICD-O-3 2023 Revision Expanded', 'Primary Site - labeled',
                 'ICD-O-3 Hist/behav, malignant', 'Site recode - rare tumors',
                 'COD to site recode ICD-O-3 2023 Revision Expanded (1999+)', 'Race/ethnicity']

label_mappings = {}
for column in label_columns:
    non_null_data = df[column].dropna()
    label_encoder = LabelEncoder()
    encoded_values = label_encoder.fit_transform(non_null_data)
    df.loc[non_null_data.index, column] = encoded_values
    label_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
with open('label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mappings, f)
df.to_csv('variables(67+1)-4-label_encoding.csv', index=False)
print("Label encoded columns and their mappings:")
for column, mapping in label_mappings.items():
    print(f"{column}: {mapping}")
print("Encoding complete and saved to file.")
