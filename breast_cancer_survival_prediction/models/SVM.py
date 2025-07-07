import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

data = pd.read_csv('intersection_features_and_target_variable_encoded.csv')

X = data.drop('Survival_months_graded', axis=1).values  # Features
y = data['Survival_months_graded'].values  # Target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

best_params = {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
best_model = SVC(**best_params, probability=True)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []

for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    best_model.fit(X_train_fold, y_train_fold)
    y_val_pred = best_model.predict(X_val_fold)
    y_val_prob = best_model.predict_proba(X_val_fold)[:, 1]

    accuracies.append(accuracy_score(y_val_fold, y_val_pred))
    precisions.append(precision_score(y_val_fold, y_val_pred))
    recalls.append(recall_score(y_val_fold, y_val_pred))
    f1_scores.append(f1_score(y_val_fold, y_val_pred))
    roc_aucs.append(roc_auc_score(y_val_fold, y_val_prob))

metrics = {
    'Accuracy': (np.mean(accuracies), np.max(accuracies), np.min(accuracies)),
    'Precision': (np.mean(precisions), np.max(precisions), np.min(precisions)),
    'Recall': (np.mean(recalls), np.max(recalls), np.min(recalls)),
    'F1 Score': (np.mean(f1_scores), np.max(f1_scores), np.min(f1_scores)),
    'ROC AUC': (np.mean(roc_aucs), np.max(roc_aucs), np.min(roc_aucs))
}

print("Metrics Summary:")
for metric, (mean, max_val, min_val) in metrics.items():
    print(f"{metric} - Mean: {mean:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}")

plt.figure()
for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    best_model.fit(X_train_fold, y_train_fold)
    y_val_prob = best_model.predict_proba(X_val_fold)[:, 1]
    fpr, tpr, _ = roc_curve(y_val_fold, y_val_prob)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC curve (area = %0.2f)' % roc_auc_score(y_val_fold, y_val_prob))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
