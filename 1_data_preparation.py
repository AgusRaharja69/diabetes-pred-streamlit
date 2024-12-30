# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import sklearn and IPython tools
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('../dataset/diabetes.csv')

# Data Info
print(df.info())
print(df.describe(include='all').T)

# Copy original data
df_copy = df.copy()

# Missing Value Count Function
def show_missing():
    missing = df_copy.columns[df_copy.isnull().any()].tolist()
    return missing

diabetes_data_copy = df_copy.copy(deep=True)
diabetes_data_copy[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
] = diabetes_data_copy[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
].replace(0, np.NaN)

# Showing the count of NaNs
print(diabetes_data_copy.isnull().sum())

# Filling missing values
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)

# Adding new features
diabetes_data_copy['BMI_vs_SkinThickness'] = diabetes_data_copy['BMI'] * diabetes_data_copy['SkinThickness']
diabetes_data_copy['Pregnancies_vs_Age'] = diabetes_data_copy['Pregnancies'] / diabetes_data_copy['Age']
diabetes_data_copy['Age_vs_DiabetesPedigreeFunction'] = diabetes_data_copy['Age'] * diabetes_data_copy['DiabetesPedigreeFunction']
diabetes_data_copy['Age_vs_Insulin'] = diabetes_data_copy['Age'] / diabetes_data_copy['Insulin']

# Handling potential division by zero issues
diabetes_data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
diabetes_data_copy.fillna(0, inplace=True)  # Fill any remaining NaNs with 0

# Plotting the histograms of the new data
diabetes_data_copy.hist(figsize=(20, 20))
plt.suptitle('Histograms of All Features', fontsize=16)
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = diabetes_data_copy.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# Boxplots for All Features
plt.figure(figsize=(20, 10))
sns.boxplot(data=diabetes_data_copy, orient="h", palette="Set2")
plt.title("Boxplot of All Features", fontsize=16)
plt.show()

# Saving to CSV and XLSX formats
csv_path = '../dataset/diabetes_with_new_features.csv'
xlsx_path = '../dataset/diabetes_with_new_features.xlsx'

diabetes_data_copy.to_csv(csv_path, index=False)
diabetes_data_copy.to_excel(xlsx_path, index=False)

print(f"Data saved to {csv_path} and {xlsx_path}")

# Display updated data
print(diabetes_data_copy.head())
