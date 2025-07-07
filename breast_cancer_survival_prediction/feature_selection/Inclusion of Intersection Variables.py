import pandas as pd

df = pd.read_csv('variables_68.csv')

def replace_numbers_with_words(text):
    number_to_word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    text = str(text)
    for digit, word in number_to_word.items():
        text = text.replace(digit, word)
    return text

columns_to_process = [
    'Year_of_diagnosis', 'CS_Tumor_Size_Ext_Eval_2004_2015', 'CS_Mets_Eval_2004_2015',
    'CS_version_input_current_2004_2015', 'Record_number_recode', 'Year_of_follow_up_recode',
    'SS_seq_num_mal_ins_most_detail', 'SS_seq_num_1975_plus_mal_most_detail', 'Year_of_death_recode',
    'Reg_Node_Eval', 'Sex', 'PRCDA_2020', 'Race_recode_W_B_AI_API', 'Origin_recode_NHIA_Hispanic_Non_Hisp',
    'TNM_7_CS_v0204_plus_Schema_recode', 'SEER_Brain_and_CNS_Recode', 'Behavior_recode_for_analysis', 'Laterality',
    'Diagnostic_Confirmation', 'ICCC_site_recode_extended_3rd_edition_IARC_2017', 'Combined_Summary_Stage_1998_2017',
    'Breast_Adjusted_AJCC_6th_T_1988_2015', 'RX_Summ_Surg_Oth_Reg_Dis_2003_plus', 'RX_Summ_Surg_Rad_Seq', 'Reason_no_cancer_directed_surgery', 'Radiation_recode',
    'RX_Summ_Systemic_Sur_Seq_2007_plus', 'Breast_Subtype_2010_plus', 'ER_Status_Recode_Breast_Cancer_1990_plus', 'PR_Status_Recode_Breast_Cancer_1990_plus', 'Derived_HER2_Recode_2010_plus', 'SEER_cause_specific_death_classification',
    'SEER_other_cause_of_death_classification', 'Survival_months_flag', 'Vital_status_recode_study_cutoff_used',
    'First_malignant_primary_indicator', 'Primary_by_international_rules', 'Site_mal_ins_most_detail',
    'Type_of_Reporting_Source', 'Marital_status_at_diagnosis', 'Rural_Urban_Continuum_Code',
    'AYA_site_recode_2020_Revision', 'Site_recode_ICD_O_3_2023_Revision_Expanded', 'Primary_Site_labeled',
    'ICD_O_3_Hist_behav_malignant', 'Site_recode_rare_tumors', 'COD_to_site_recode_ICD_O_3_2023_Revision_Expanded_1999_plus',
    'Sequence_number', 'Race_ethnicity', 'Survival_months_graded'
]

for column in columns_to_process:
    if column in df.columns:
        df[column] = df[column].apply(replace_numbers_with_words)

df.to_csv('variables_68_encoded_1.csv', index=False)

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pickle

df = pd.read_csv('variables_68_encoded_1.csv')

ordinal_columns = [
    'Year_of_diagnosis', 'CS_Tumor_Size_Ext_Eval_2004_2015', 'CS_Mets_Eval_2004_2015',
    'CS_version_input_current_2004_2015', 'Record_number_recode', 'Year_of_follow_up_recode',
    'SS_seq_num_mal_ins_most_detail', 'SS_seq_num_1975_plus_mal_most_detail', 'Year_of_death_recode',
    'Reg_Node_Eval', 'SEER_Brain_and_CNS_Recode', 'Rural_Urban_Continuum_Code', 'Sequence_number'
]

ordinal_encoder = OrdinalEncoder()
ordinal_mappings = {}

for column in ordinal_columns:
    non_null_data = df[[column]].dropna()
    encoded_values = ordinal_encoder.fit_transform(non_null_data)
    df.loc[non_null_data.index, column] = encoded_values
    ordinal_mappings[column] = ordinal_encoder.categories_[0]

with open('ordinal_mapping.pk2', 'wb') as f:
    pickle.dump(ordinal_mappings, f)

df.to_csv('variables_68_encoded_1_ordinal.csv', index=False)

print("Ordinal encoded columns and their mappings:")
for column, mapping in ordinal_mappings.items():
    print(f"{column}: {list(mapping)}")

print("Encoding complete and saved to specified file.")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('variables_68_encoded_1_ordinal.csv')

label_columns = [
    'Sex', 'PRCDA_2020', 'Origin_recode_NHIA_Hispanic_Non_Hisp',
    'Vital_status_recode_study_cutoff_used', 'First_malignant_primary_indicator',
    'Primary_by_international_rules', 'Survival_months_graded', 'Race_recode_W_B_AI_API',
    'TNM_7_CS_v0204_plus_Schema_recode', 'Behavior_recode_for_analysis', 'Laterality',
    'Diagnostic_Confirmation', 'ICCC_site_recode_extended_3rd_edition_IARC_2017',
    'Combined_Summary_Stage_1998_2017', 'Breast_Adjusted_AJCC_6th_T_1988_2015',
    'RX_Summ_Surg_Oth_Reg_Dis_2003_plus', 'RX_Summ_Surg_Rad_Seq', 'Reason_no_cancer_directed_surgery',
    'Radiation_recode', 'RX_Summ_Systemic_Sur_Seq_2007_plus', 'Breast_Subtype_2010_plus',
    'ER_Status_Recode_Breast_Cancer_1990_plus', 'PR_Status_Recode_Breast_Cancer_1990_plus',
    'Derived_HER2_Recode_2010_plus', 'SEER_cause_specific_death_classification',
    'SEER_other_cause_of_death_classification', 'Survival_months_flag', 'Site_mal_ins_most_detail',
    'Type_of_Reporting_Source', 'Marital_status_at_diagnosis', 'AYA_site_recode_2020_Revision',
    'Site_recode_ICD_O_3_2023_Revision_Expanded', 'Primary_Site_labeled',
    'ICD_O_3_Hist_behav_malignant', 'Site_recode_rare_tumors',
    'COD_to_site_recode_ICD_O_3_2023_Revision_Expanded_1999_plus', 'Race_ethnicity'
]

label_mappings = {}

for column in label_columns:
    non_null_data = df[column].dropna()
    label_encoder = LabelEncoder()
    encoded_values = label_encoder.fit_transform(non_null_data)
    df.loc[non_null_data.index, column] = encoded_values
    label_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

with open('label_mapping.pk2', 'wb') as f:
    pickle.dump(label_mappings, f)

df.to_csv('variables_68_encoded_1_label.csv', index=False)

print("Label encoded columns and their mappings:")
for column, mapping in label_mappings.items():
    print(f"{column}: {mapping}")

print("Encoding complete and saved to specified file.")

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

data = pd.read_csv('variables_68_encoded_1_label.csv')
X = data.drop(columns=['Survival_months_graded'])
y = data['Survival_months_graded']

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
boruta_selector = BorutaPy(rf_model, n_estimators='auto', perc=20, random_state=42, verbose=2)
boruta_selector.fit(X.values, y.values)

all_features_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': boruta_selector.ranking_,
    'Selected': boruta_selector.support_
}).sort_values(by='Ranking', ascending=True)

print(all_features_ranking)
selected_features = X.columns[boruta_selector.support_].tolist()
print(f"Selected features: {selected_features}")

top_n_features = 35
important_features = all_features_ranking.head(top_n_features)['Feature'].tolist()
print(f"Top {top_n_features} features: {important_features}")

from sklearn.feature_selection import SelectKBest, mutual_info_classif

data = pd.read_csv('variables_68_encoded_1_label.csv')
X = data.drop(columns=['Survival_months_graded'])
y = data['Survival_months_graded']

selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X, y)

scores = pd.DataFrame({
    'Feature': X.columns,
    'Mutual_Information_Score': selector.scores_
})

print(scores.sort_values(by='Mutual_Information_Score', ascending=False))
k = 35
selected_features = scores.nlargest(k, 'Mutual_Information_Score')['Feature']
print(f"Selected top {k} features: {selected_features.tolist()}")

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

data = pd.read_csv('variables_68_encoded_1_label.csv')
X = data.drop('Survival_months_graded', axis=1)
y = data['Survival_months_graded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_model, n_features_to_select=35)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]
print("Selected features: ", selected_features)

boruta_algorithm_features = [
    'Primary_Site', 'ICCC_site_recode_extended_3rd_edition_IARC_2017', 'Combined_Summary_Stage_1998_2017', 'Breast_Adjusted_AJCC_6th_T_1988_2015', 'RX_Summ_Surg_Oth_Reg_Dis_2003_plus', 'RX_Summ_Surg_Rad_Seq', 'Reason_no_cancer_directed_surgery', 'Radiation_recode', 'RX_Summ_Systemic_Sur_Seq_2007_plus', 'Breast_Subtype_2010_plus', 'ER_Status_Recode_Breast_Cancer_1990_plus', 'PR_Status_Recode_Breast_Cancer_1990_plus', 'Derived_HER2_Recode_2010_plus', 'SEER_cause_specific_death_classification', 'Diagnostic_Confirmation', 'SEER_other_cause_of_death_classification', 'Vital_status_recode_study_cutoff_used', 'First_malignant_primary_indicator', 'Primary_by_international_rules', 'Site_mal_ins_most_detail', 'Type_of_Reporting_Source', 'Marital_status_at_diagnosis', 'Rural_Urban_Continuum_Code', 'AYA_site_recode_2020_Revision', 'Site_recode_ICD_O_3_2023_Revision_Expanded', 'Primary_Site_labeled', 'ICD_O_3_Hist_behav_malignant', 'Site_recode_rare_tumors', 'COD_to_site_recode_ICD_O_3_2023_Revision_Expanded_1999_plus', 'Survival_months_flag', 'Laterality', 'Behavior_recode_for_analysis', 'Year_of_diagnosis', 'Regional_nodes_examined_1988_plus', 'Regional_nodes_positive_1988_plus'
]

information_gain_method_features = [
    'Year_of_diagnosis', 'EOD_Primary_Tumor', 'EOD_Regional_Nodes', 'EOD_Mets', 'Year_of_death_recode', 'Year_of_follow_up_recode', 'Tumor_Size_Recode_1988_plus', 'Breast_Adjusted_AJCC_6th_T_1988_2015', 'RX_Summ_Surg_Prim_Site_1998_plus', 'Reg_Node_Eval', 'Regional_nodes_positive_1988_plus', 'Breast_Subtype_2010_plus', 'ICD_O_3_Hist_behav_malignant', 'CS_Tumor_Size_Ext_Eval_2004_2015', 'Site_recode_rare_tumors', 'CS_Mets_Eval_2004_2015', 'Regional_nodes_examined_1988_plus', 'Derived_HER2_Recode_2010_plus', 'COD_to_site_recode_ICD_O_3_2023_Revision_Expanded_1999_plus', 'AYA_site_recode_2020_Revision', 'Age_recode_with_single_ages_and_90', 'ICCC_site_recode_extended_3rd_edition_IARC_2017', 'Vital_status_recode_study_cutoff_used', 'ER_Status_Recode_Breast_Cancer_1990_plus', 'Site_recode_ICD_O_3_2023_Revision_Expanded', 'PR_Status_Recode_Breast_Cancer_1990_plus', 'Adjusted_CS_site_specific_factor_7_2004_2017_varying_by_schema', 'Primary_by_international_rules', 'RX_Summ_Surg_Oth_Reg_Dis_2003_plus', 'Reason_no_cancer_directed_surgery', 'Diagnostic_Confirmation', 'RX_Summ_Surg_Rad_Seq', 'Race_ethnicity', 'First_malignant_primary_indicator', 'SEER_cause_specific_death_classification'
]

rfe_method_features = [
    'Regional_nodes_examined_1988_plus', 'Regional_nodes_positive_1988_plus', 'RX_Summ_Surg_Prim_Site_1998_plus', 'Time_from_diagnosis_to_treatment_in_days_recode', 'Age_recode_with_single_ages_and_90', 'Median_household_income_inflation_adj_to_222_dollar', 'EOD_Primary_Tumor', 'EOD_Regional_Nodes', 'EOD_Mets', 'Tumor_Size_Recode_1988_plus', 'Year_of_diagnosis', 'CS_Tumor_Size_Ext_Eval_2004_2015', 'CS_Mets_Eval_2004_2015', 'CS_version_input_current_2004_2015', 'Year_of_follow_up_recode', 'Year_of_death_recode', 'Reg_Node_Eval', 'ICCC_site_recode_extended_3rd_edition_IARC_2017', 'Combined_Summary_Stage_1998_2017', 'Breast_Adjusted_AJCC_6th_T_1988_2015', 'Reason_no_cancer_directed_surgery', 'RX_Summ_Systemic_Sur_Seq_2007_plus', 'Breast_Subtype_2010_plus', 'ER_Status_Recode_Breast_Cancer_1990_plus', 'PR_Status_Recode_Breast_Cancer_1990_plus', 'Derived_HER2_Recode_2010_plus', 'SEER_cause_specific_death_classification', 'Vital_status_recode_study_cutoff_used', 'Marital_status_at_diagnosis', 'Rural_Urban_Continuum_Code', 'AYA_site_recode_2020_Revision', 'Primary_Site_labeled', 'ICD_O_3_Hist_behav_malignant', 'Site_recode_rare_tumors', 'COD_to_site_recode_ICD_O_3_2023_Revision_Expanded_1999_plus'
]

set1 = set(boruta_algorithm_features)
set2 = set(information_gain_method_features)
set3 = set(rfe_method_features)

common_features = set1 & set2 & set3

print("Common features across all methods:")
print(common_features)

input_file = 'variables_68_encoded_1_label.csv'
df = pd.read_csv(input_file)
df_common_features = df[list(common_features)]
output_file = 'intersection_features_encoded.csv'
df_common_features.to_csv(output_file, index=False)

print(f"New feature set with common features saved to: {output_file}")
