import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, log_loss

df = pd.read_csv('intersection_features_target_encoded.csv')
X = df.drop(columns=['Survival_months_graded'])
y = df['Survival_months_graded']
best_params = {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 0.01, 'subsample': 0.8}
model = xgb.XGBClassifier(**best_params)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

train_accuracies, train_precisions, train_recalls, train_f1_scores = [], [], [], []
val_accuracies, val_precisions, val_recalls, val_f1_scores = [], [], [], []
train_losses, val_losses = [], []

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]

    train_loss = log_loss(y_train, y_train_prob)
    val_loss = log_loss(y_val, y_val_prob)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    train_precisions.append(precision_score(y_train, y_train_pred))
    train_recalls.append(recall_score(y_train, y_train_pred))
    train_f1_scores.append(f1_score(y_train, y_train_pred))

    val_accuracies.append(accuracy_score(y_val, y_val_pred))
    val_precisions.append(precision_score(y_val, y_val_pred))
    val_recalls.append(recall_score(y_val, y_val_pred))
    val_f1_scores.append(f1_score(y_val, y_val_pred))

    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

metrics_summary = {
    'Train Accuracy': (np.mean(train_accuracies), np.max(train_accuracies), np.min(train_accuracies)),
    'Validation Accuracy': (np.mean(val_accuracies), np.max(val_accuracies), np.min(val_accuracies)),
    'Train Precision': (np.mean(train_precisions), np.max(train_precisions), np.min(train_precisions)),
    'Validation Precision': (np.mean(val_precisions), np.max(val_precisions), np.min(val_precisions)),
    'Train Recall': (np.mean(train_recalls), np.max(train_recalls), np.min(train_recalls)),
    'Validation Recall': (np.mean(val_recalls), np.max(val_recalls), np.min(val_recalls)),
    'Train F1 Score': (np.mean(train_f1_scores), np.max(train_f1_scores), np.min(train_f1_scores)),
    'Validation F1 Score': (np.mean(val_f1_scores), np.max(val_f1_scores), np.min(val_f1_scores)) }

print("Metrics Summary (Mean, Max, Min):")
for metric, values in metrics_summary.items():
    print(f"{metric}: Mean = {values[0]:.4f}, Max = {values[1]:.4f}, Min = {values[2]:.4f}")

plt.figure(figsize=(10, 7))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

for i, tpr in enumerate(tprs):
    plt.plot(mean_fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i + 1} ROC (AUC = {aucs[i]:.2f})')

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_auc, lw=2, alpha=.8)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
