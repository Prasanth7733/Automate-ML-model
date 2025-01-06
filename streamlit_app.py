import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.title("Simplified Machine Learning App")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Upload Data", "Data Visualization", "Train Model", "Make Predictions"])

if section == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())
        if st.checkbox("Show Dataset Summary"):
            st.write(data.describe())
            st.write("### Null Values:", data.isnull().sum())
        if st.checkbox("Clean Data"):
            # Fill missing numeric columns with their mean values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
            st.write("Data cleaned!")
            st.write(data.head())
        st.session_state['data'] = data

if section == "Data Visualization":
    if 'data' in st.session_state:
        st.header("Data Visualization")
        data = st.session_state['data']
        plot_type = st.selectbox("Choose Plot Type", ["Scatter Plot", "Histogram", "Correlation Heatmap", "Pairplot","Boxplot"])
        
        if plot_type == "Scatter Plot":
            x_col = st.selectbox("X-Axis", data.columns)
            y_col = st.selectbox("Y-Axis", data.columns)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=x_col, y=y_col)
            st.pyplot(plt)
            
        elif plot_type == "Histogram":
            col = st.selectbox("Select Column", data.columns)
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], kde=True)
            st.pyplot(plt)
        
        elif plot_type == "Correlation Heatmap":
            plt.figure(figsize=(10, 6))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)
            
        elif plot_type == "Pairplot":
            plt.figure(figsize=(10, 6))
            sns.pairplot(data)
            st.pyplot(plt)
        elif plot_type == "Box Plot":
            col = st.selectbox("Select Column", data.columns)
            plt.figure(figsize=(10, 6))
            sns.boxplot(data[col])
            st.pyplot(plt)
            
    else:
        st.warning("Upload data first!")

if section == "Train Model":
    if 'data' in st.session_state:
        st.header("Train a Model")
        data = st.session_state['data']
        target_col = st.selectbox("Select Target Column", data.columns)
        X = data.drop(columns=[target_col])
        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_type = st.selectbox("Choose Model Type", ["Linear Regression", "Logistic Regression", "Ridge", "Lasso", 
                                                        "Decision Tree", "Random Forest", "SVM", "KNN"])
        
        if st.button("Train Model"):
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "Ridge":
                model = Ridge()
            elif model_type == "Lasso":
                model = Lasso()
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_type == "Random Forest":
                model = RandomForestClassifier()
            elif model_type == "SVM":
                model = SVC()
            elif model_type == "KNN":
                model = KNeighborsClassifier()
            
            model.fit(X_train, y_train)

            # Save the feature names (columns)
            feature_names = X_train.columns

            if model_type in ["Linear Regression", "Ridge", "Lasso"]:
                y_pred = model.predict(X_test)
                st.write("### R² Score:", r2_score(y_test, y_pred))
            else:
                y_pred = model.predict(X_test)
                st.write("### Accuracy:", accuracy_score(y_test, y_pred))
                st.write("### Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))

            # Save the trained model and feature names
            joblib.dump(model, "trained_model.pkl")
            joblib.dump(feature_names, "feature_names.pkl")  # Save feature names as a separate file
            st.success("Model trained and saved!")

if section == "Make Predictions":
    st.header("Make Predictions")
    prediction_file = st.file_uploader("Upload new data for prediction", type=["csv", "xlsx"])
    if prediction_file:
        if prediction_file.name.endswith('.csv'):
            new_data = pd.read_csv(prediction_file)
        else:
            new_data = pd.read_excel(prediction_file)
        
        st.write("### New Data Preview")
        st.write(new_data.head())
        
        if st.button("Predict"):
            # Load the trained model and feature names
            model = joblib.load("trained_model.pkl")
            feature_names = joblib.load("feature_names.pkl")  # Load saved feature names

            # Preprocess the new data the same way as training data
            # Convert date columns to datetime and extract features
            date_columns = new_data.select_dtypes(include=['object']).columns
            for col in date_columns:
                try:
                    new_data[col] = pd.to_datetime(new_data[col], errors='raise')
                except ValueError:
                    pass  # Skip non-date columns

            for col in date_columns:
                if pd.api.types.is_datetime64_any_dtype(new_data[col]):
                    new_data[f'{col}_year'] = new_data[col].dt.year
                    new_data[f'{col}_month'] = new_data[col].dt.month
                    new_data[f'{col}_day'] = new_data[col].dt.day
                    new_data[f'{col}_dayofweek'] = new_data[col].dt.dayofweek
                    new_data[f'{col}_elapsed'] = (new_data[col] - new_data[col].min()).dt.days
                    new_data.drop(columns=[col], inplace=True)

            # One-hot encode categorical columns
            categorical_columns = new_data.select_dtypes(include=['object']).columns
            new_data = pd.get_dummies(new_data, columns=categorical_columns, drop_first=True)
            
            # Fill missing values in numeric columns (same as training data)
            numeric_columns = new_data.select_dtypes(include=[np.number]).columns
            new_data[numeric_columns] = new_data[numeric_columns].fillna(new_data[numeric_columns].mean())

            # Ensure that the new data has the same columns as the training data
            missing_cols = set(feature_names) - set(new_data.columns)
            for col in missing_cols:
                new_data[col] = 0  # Add missing columns with default value 0
            new_data = new_data[feature_names]  # Reorder columns to match the training data

            try:
                predictions = model.predict(new_data)
                st.write("### Predictions:")
                st.write(predictions)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

st.write('Hello world!')







st.write('Hello world!')
