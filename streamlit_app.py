import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

st.title("Enhanced Machine Learning App")

# Sidebar Navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Upload Data", "Data Visualization", "Train Model", "Evaluate Model", "Make Predictions"])

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

        # Data Cleaning
        if st.checkbox("Clean Data"):
            if st.checkbox("Fill Missing Values with Mean"):
                data.fillna(data.select_dtypes(include=np.number).mean(), inplace=True)
            if st.checkbox("Drop Rows with Missing Values"):
                data.dropna(inplace=True)

            st.write("Data cleaned!")
            st.write(data.head())

        # Save data in session state
        st.session_state['data'] = data

if section == "Data Visualization":
    if 'data' in st.session_state:
        st.header("Data Visualization")
        data = st.session_state['data']

        plot_type = st.selectbox("Choose Plot Type", ["Scatter Plot", "Histogram", "Correlation Heatmap", "Box Plot"])
        
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

        elif plot_type == "Box Plot":
            col = st.selectbox("Select Column", data.columns)
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data[col])
            st.pyplot(plt)
    else:
        st.warning("Upload data first!")
if section == "Train Model":
    if 'data' in st.session_state:
        st.header("Train a Model")
        data = st.session_state['data']

        # Target Selection
        target_col = st.selectbox("Select Target Column", data.columns)
        features = st.multiselect("Select Features (Default: All)", data.columns, default=list(data.columns.drop(target_col)))

        X = data[features]
        y = data[target_col]

        # Preprocessing
        if st.checkbox("Standardize Features"):
            numeric_cols = X.select_dtypes(include=np.number).columns
            non_numeric_cols = X.select_dtypes(exclude=np.number).columns

            if not numeric_cols.empty:
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            if not non_numeric_cols.empty:
                st.warning(f"Non-numeric columns excluded: {list(non_numeric_cols)}")
                X = X[numeric_cols]

        if y.dtype == 'object':
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Debug Logs
        st.write("Sample of X_train:")
        st.write(X_train.head())
        st.write("Sample of y_train:", y_train[:5])

        # Model Selection
        model_type = st.selectbox("Choose Model Type", ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

        if st.button("Train Model"):
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_type == "Random Forest":
                model = RandomForestClassifier()
            elif model_type == "Gradient Boosting":
                model = GradientBoostingClassifier()

            model.fit(X_train, y_train)
            joblib.dump(model, "trained_model.pkl")
            st.success(f"{model_type} trained successfully!")
    else:
        st.warning("Upload data first!")


if section == "Evaluate Model":
    st.header("Evaluate Model")

    if 'data' in st.session_state:
        data = st.session_state['data']

        if 'trained_model.pkl' not in globals():
            model = joblib.load("trained_model.pkl")
        else:
            st.error("Train a model first!")

        y_pred = model.predict(X_test)

        if model_type in ["Linear Regression", "Ridge", "Lasso"]:
            st.write("### R² Score:", r2_score(y_test, y_pred))
        else:
            st.write("### Accuracy:", accuracy_score(y_test, y_pred))
            st.write("### Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("### Classification Report:")
            st.text(classification_report(y_test, y_pred))




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
            model = joblib.load("trained_model.pkl")
            predictions = model.predict(new_data)
            st.write("### Predictions:")
            st.write(predictions)



st.write('Hello world!')
