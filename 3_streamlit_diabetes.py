# Streamlit App for Diabetes Prediction
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings globally
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='st.cache is deprecated')
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Custom CSS for footer and image styling
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #6c757d;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 1000;
    }
    .footer img {
        height: 50px;
        margin-right: 10px;
    }
    .footer a {
        color: #6c757d;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Footer HTML
st.markdown("""
    <div class="footer">
        <span>
            <img src="https://teknik.warmadewa.ac.id/storage/uploads/teknik-komputer-logo.jpg" alt="Warmadewa Computer Engineering">
            <a href="https://www.teknik.warmadewa.ac.id/teknik-komputer">© Warmadewa Computer Engineering Team</a>.
        </span>
    </div>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('../dataset/diabetes_with_new_features.csv')

df = load_data()
st.title("Diabetes Prediction Training")
st.write("Train a machine learning model to predict diabetes based on selected features.")

# Data Preprocessing
st.subheader("Data Preprocessing")

# Feature selection
df_copy = df.copy()
features = df_copy.drop(columns=['Outcome']).columns.tolist()
default_features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
target = 'Outcome'

selected_features = st.multiselect("Select Features for Training", features, default=default_features)

if not selected_features:
    st.error("Please select at least one feature.")
else:
    X = df_copy[selected_features]
    y = df_copy[target]
    
    # SMOTE Oversampling
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if st.checkbox("Show Correlation Heatmap"):
        correlation = df_copy.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title("Feature Correlation Matrix")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(plt)

    # Feature Distribution
    st.subheader("Feature Distribution")
    if st.checkbox("Show Feature Distributions"):
        col1, col2, col3 = st.columns(3)
        for i, feature in enumerate(selected_features):
            plt.figure(figsize=(8, 4))
            sns.histplot(df_copy[feature], kde=True, color="blue", bins=30)
            plt.title(f"Distribution of {feature}")
            if i % 3 == 0:
                with col1:
                    st.pyplot(plt)
            elif i % 3 == 1:
                with col2:
                    st.pyplot(plt)
            else:
                with col3:
                    st.pyplot(plt)

    # Feature vs Target Analysis
    st.subheader("Feature vs Target Analysis")
    if st.checkbox("Show Feature vs Target Analysis"):
        col1, col2, col3 = st.columns(3)
        for i, feature in enumerate(selected_features):
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df_copy['Outcome'], y=df_copy[feature], palette="Set2")
            plt.title(f"{feature} vs Diabetes")
            plt.xlabel("Diabetes (0: No, 1: Yes)")
            plt.ylabel(feature)
            if i % 3 == 0:
                with col1:
                    st.pyplot(plt)
            elif i % 3 == 1:
                with col2:
                    st.pyplot(plt)
            else:
                with col3:
                    st.pyplot(plt)

    # Training Button
    if st.button("Train Model"):
        # Model Training
        st.subheader("Model Training")
        classifier = RandomForestClassifier(random_state=42)
        grid_values = {'n_estimators': [50, 65, 80, 95, 120], 'max_depth': [3, 5, 7, 9, 12]}
        GSclassifier = GridSearchCV(classifier, param_grid=grid_values, scoring='roc_auc', cv=5)

        # Fit the model
        st.write("Training the model...")
        GSclassifier.fit(X_train, y_train)
        best_params = GSclassifier.best_params_

        # Final Model Training
        classifier = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
        classifier.fit(X_train, y_train)

        # Evaluation
        st.subheader("Model Evaluation")
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        st.write(f"Training Accuracy: {train_accuracy:.2f}")
        st.write(f"Test Accuracy: {test_accuracy:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(plt)

        # ROC Curve
        st.subheader("ROC Curve")
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        classifier.fit(X_train_sm, y_train_sm)

        # ROC Evaluation
        y_test_proba = classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = roc_auc_score(y_test, y_test_proba)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(plt)

        # Feature Importance
        st.subheader("Feature Importance")
        importances = classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.write(feature_importance_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title("Feature Importance")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(plt)

        # Model Performance Comparison
        st.subheader("Model Performance Comparison")
        feature_combinations = [
            selected_features[:i] for i in range(1, len(selected_features) + 1)
        ]
        performances = []

        for combination in feature_combinations:
            X_temp = X[combination]
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_temp_sm, y_sm = smote.fit_resample(X_temp, y)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                X_temp_sm, y_sm, test_size=0.2, random_state=42
            )
            classifier.fit(X_train_temp, y_train_temp)
            y_test_pred_temp = classifier.predict(X_test_temp)
            acc = accuracy_score(y_test_temp, y_test_pred_temp)
            performances.append((combination, acc))

        performance_df = pd.DataFrame(performances, columns=["Feature Combination", "Accuracy"])
        st.write(performance_df)

# Close the div
st.markdown('</div>', unsafe_allow_html=True)
