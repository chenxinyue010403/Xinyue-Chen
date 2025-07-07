import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_curve, auc, precision_score
)
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('intersection_features_target_encoded.csv')
X = df.drop(columns=['Survival months_graded'])
y = df['Survival months_graded']

# Define Random Forest classifier
rf = RandomForestClassifier(
    bootstrap=True, max_depth=30, max_features='auto',
    min_samples_leaf=4, min_samples_split=10, n_estimators=200,
    random_state=42
)

# Define cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store metrics
train_accuracies, val_accuracies = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
train_precisions, val_precisions = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Cross-validation loop
for train_index, val_index in cv.split(X, y):
    # Split data into train and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on training and validation sets
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)

    # Compute metrics for training set
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    train_recalls.append(recall_score(y_train, y_train_pred))
    train_f1s.append(f1_score(y_train, y_train_pred))
    train_precisions.append(precision_score(y_train, y_train_pred))

    # Compute metrics for validation set
    val_accuracies.append(accuracy_score(y_val, y_val_pred))
    val_recalls.append(recall_score(y_val, y_val_pred))
    val_f1s.append(f1_score(y_val, y_val_pred))
    val_precisions.append(precision_score(y_val, y_val_pred))

    # Compute ROC curve and AUC for validation set
    probas_ = rf.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, probas_)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

# Compute mean, max, and min values for metrics
def print_stats(name, values):
    print(f"{name}")
    print(f"  Mean: {np.mean(values):.4f}")
    print(f"  Max: {np.max(values):.4f}")
    print(f"  Min: {np.min(values):.4f}\n")

# Print statistics
print_stats("Training Accuracy", train_accuracies)
print_stats("Validation Accuracy", val_accuracies)
print_stats("Training Recall", train_recalls)
print_stats("Validation Recall", val_recalls)
print_stats("Training F1 Score", train_f1s)
print_stats("Validation F1 Score", val_f1s)
print_stats("Training Precision", train_precisions)
print_stats("Validation Precision", val_precisions)

# Plot ROC curve
plt.figure(figsize=(10, 7))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)

# Compute mean TPR
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)

# Plot individual folds' ROC curves
for i, tpr in enumerate(tprs):
    plt.plot(mean_fpr, tpr, alpha=0.3, label=f'Fold {i+1} ROC (AUC = {aucs[i]:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
