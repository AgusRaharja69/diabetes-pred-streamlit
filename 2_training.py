# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed dataset
data_path = '../dataset/diabetes_with_new_features.csv'
df = pd.read_csv(data_path)

# Separate features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Apply SMOTEENN for balancing the dataset
# smoteenn = SMOTEENN()
# X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)

# Apply SMOTE only to the training set
smote = SMOTE(sampling_strategy='minority', random_state=0)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Gradient Boosting Classifier with Grid Search
classifier = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_classifier = grid_search.best_estimator_

y_train_pred = best_classifier.predict(X_train)
y_test_pred = best_classifier.predict(X_test)

# Load the original dataset
data_original_path = '../dataset/diabetes.csv'
df_original = pd.read_csv(data_original_path)

# Adding new features to the original dataset
df_original['BMI_vs_SkinThickness'] = df_original['BMI'] * df_original['SkinThickness']
df_original['Pregnancies_vs_Age'] = df_original['Pregnancies'] / df_original['Age']
df_original['Age_vs_DiabetesPedigreeFunction'] = df_original['Age'] * df_original['DiabetesPedigreeFunction']
df_original['Age_vs_Insulin'] = df_original['Age'] / df_original['Insulin']

# Handle potential division by zero or NaN values
df_original.replace([np.inf, -np.inf], np.nan, inplace=True)
df_original.fillna(0, inplace=True)

# Separate features and target variable
X_original_test = df_original.drop(columns=['Outcome'])
y_original_test = df_original['Outcome']

# Reorder columns to match the training dataset
X_original_test = X_original_test[X.columns]

# Scale the original test dataset using the scaler fitted during training
X_original_test_scaled = scaler.transform(X_original_test)

# Evaluate the trained model on the original dataset
y_original_test_pred = best_classifier.predict(X_original_test_scaled)
y_original_test_proba = best_classifier.predict_proba(X_original_test_scaled)[:, 1]

# Metrics on the original dataset
print("Test Accuracy (Original Dataset):", roc_auc_score(y_original_test, y_original_test_proba))
# Accuracy and Classification Reports
print("Train Accuracy:", roc_auc_score(y_train, best_classifier.predict_proba(X_train)[:, 1]))
print("Test Accuracy:", roc_auc_score(y_test, best_classifier.predict_proba(X_test)[:, 1]))

print("\nClassification Report (Original Dataset):")
print(classification_report(y_original_test, y_original_test_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix for the original dataset
cm_original = confusion_matrix(y_original_test, y_original_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix (Original Dataset)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve for the original dataset
fpr_original, tpr_original, _ = roc_curve(y_original_test, y_original_test_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr_original, tpr_original, label=f'AUC = {roc_auc_score(y_original_test, y_original_test_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve (Original Dataset)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()



# print("\nClassification Report (Test Data):")
# print(classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive']))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_test_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # ROC Curve
# fpr, tpr, _ = roc_curve(y_test, best_classifier.predict_proba(X_test)[:, 1])
# plt.figure(figsize=(6, 6))
# plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, best_classifier.predict_proba(X_test)[:, 1]):.2f}')
# plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
# plt.title('ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()
