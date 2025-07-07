# Bar Plot
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def create_abbreviation_map(feature_names, abbreviations):
    return dict(zip(feature_names, abbreviations))

simplified_feature_names = ['YR_DIAG', 'VITALSTATUS', 'ADJ_AJCC_6TH_T', 'NO_SURG', 'DEATH_SPC', 'ERSTATUS',
                            'CODTOSITE', 'ICCC_SITE', 'POS_RED_NODES', 'PRSTATUS', 'BEHAV_MAL', 'RARETM_SITE',
                            'EXAM_RED_NODES', 'HER2', 'AYA_SITE', 'SUBTYPE']

abbreviation_map = create_abbreviation_map(X_test.columns, simplified_feature_names)
abbreviated_feature_names = [abbreviation_map[name] for name in X_test.columns]
with tqdm(total=1, desc="Plotting feature importance") as pbar:
    shap.summary_plot(shap_values, X_test, plot_type='bar', feature_names=abbreviated_feature_names)
    pbar.update(1)

# Summary Plot
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def create_abbreviation_map(feature_names, abbreviations):
    return dict(zip(feature_names, abbreviations))

simplified_feature_names = ['YR_DIAG', 'VITALSTATUS', 'ADJ_AJCC_6TH_T', 'NO_SURG', 'DEATH_SPC', 'ERSTATUS',
                            'CODTOSITE', 'ICCC_SITE', 'POS_RED_NODES', 'PRSTATUS', 'BEHAV_MAL', 'RARETM_SITE',
                            'EXAM_RED_NODES', 'HER2', 'AYA_SITE', 'SUBTYPE']

abbreviation_map = create_abbreviation_map(X_test.columns, simplified_feature_names)
abbreviated_feature_names = [abbreviation_map[name] for name in X_test.columns]
with tqdm(total=1, desc="Plotting SHAP value distribution") as pbar:
    ax = plt.gca()

    ax.set_xlim(-2, 4.25)
    ax.set_xticks(np.arange(-2, 4.25, 0.5))
    plt.setp(ax.get_xticklabels(), fontsize=1)
    plt.setp(ax.get_yticklabels(), fontsize=1)
    ax.set_xlabel('SHAP Value', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('SHAP Values Summary Plot', fontsize=14)
    shap.summary_plot(shap_values, X_test, feature_names=abbreviated_feature_names)
    pbar.update(1)
plt.show()

# Scatter Plot
import shap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def create_abbreviation_map(feature_names, abbreviations):
    return dict(zip(feature_names, abbreviations))

simplified_feature_names = ['YR_DIAG', 'VITALSTATUS', 'ADJ_AJCC_6TH_T', 'NO_SURG', 'DEATH_SPC', 'ERSTATUS',
                            'CODTOSITE', 'ICCC_SITE', 'POS_RED_NODES', 'PRSTATUS', 'BEHAV_MAL', 'RARETM_SITE',
                            'EXAM_RED_NODES', 'HER2', 'AYA_SITE', 'SUBTYPE']

abbreviation_map = create_abbreviation_map(X_test.columns, simplified_feature_names)
feature_names = X_test.columns
for feature_name in tqdm(feature_names, desc="Plotting SHAP scatter plot for each feature"):
    simplified_feature_name = abbreviation_map[feature_name]
    feature_index = X_test.columns.get_loc(feature_name)
    feature_values = X_test[feature_name]
    shap_values_for_feature = shap_values[:, feature_index]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(feature_values, shap_values_for_feature, c=feature_values, cmap='coolwarm', alpha=0.5,
                          edgecolors='w', s=40)
    plt.colorbar(scatter, label=feature_name)  # Add color bar to show the mapping of values
    plt.xlabel(simplified_feature_name, fontsize=12)
    plt.ylabel('SHAP Value', fontsize=12)
    plt.title(f'SHAP Scatter Plot for {simplified_feature_name}', fontsize=14)
    plt.grid(True)
    plt.show()

# Force Plot
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def create_abbreviation_map(feature_names, abbreviations):
    return dict(zip(feature_names, abbreviations))

simplified_feature_names = ['YR_DIAG', 'VITALSTATUS', 'ADJ_AJCC_6TH_T', 'NO_SURG', 'DEATH_SPC', 'ERSTATUS',
                            'CODTOSITE', 'ICCC_SITE', 'POS_RED_NODES', 'PRSTATUS', 'BEHAV_MAL', 'RARETM_SITE',
                            'EXAM_RED_NODES', 'HER2', 'AYA_SITE', 'SUBTYPE']

abbreviation_map = create_abbreviation_map(X_test.columns, simplified_feature_names)
with tqdm(total=1, desc="Plotting force plot") as pbar:
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], feature_names=abbreviated_feature_names, matplotlib=True)
    pbar.update(1)

# Waterfall Plot
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def create_abbreviation_map(feature_names, abbreviations):
    return dict(zip(feature_names, abbreviations))

simplified_feature_names = ['YR_DIAG', 'VITALSTATUS', 'ADJ_AJCC_6TH_T', 'NO_SURG', 'DEATH_SPC', 'ERSTATUS',
                            'CODTOSITE', 'ICCC_SITE', 'POS_RED_NODES', 'PRSTATUS', 'BEHAV_MAL', 'RARETM_SITE',
                            'EXAM_RED_NODES', 'HER2', 'AYA_SITE', 'SUBTYPE']

abbreviation_map = create_abbreviation_map(X_test.columns, simplified_feature_names)
abbreviated_feature_names = [abbreviation_map[name] for name in X_test.columns]
sample_index = 0
with tqdm(total=1, desc="Plotting waterfall plot") as pbar:
    shap.waterfall_plot(shap.Explanation(values=shap_values[sample_index],
                                         base_values=explainer.expected_value,
                                         data=X_test.iloc[sample_index, :],
                                         feature_names=abbreviated_feature_names),
                        max_display=16)
    pbar.update(1)
plt.show()
