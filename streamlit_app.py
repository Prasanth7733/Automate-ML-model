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
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])  # Ensure this line is present

    if uploaded_file is not None:  # Check if a file has been uploaded
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
            if st.checkbox("Fill Missing Values with Mean"):
                data.fillna(data.mean(), inplace=True)
            if st.checkbox("Drop Rows with Missing Values"):
                data.dropna(inplace=True)
            st.write("Data cleaned!")
            st.write(data.head())

        # Save to session state
        st.session_state['data'] = data
    else:
        st.warning("Please upload a file to proceed!")

if section == "Data Visualization":
    if 'data' in st.session_state:  # Check if data is available
        st.header("Data Visualization")
        data = st.session_state['data']  # Access the stored data

        plot_type = st.selectbox("Choose Plot Type", [
            "Scatter Plot", "Histogram", "Correlation Heatmap", "Box Plot",
            "Pair Plot", "Bar Plot", "Line Plot", "Violin Plot", "Distribution Plot"
        ])

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

        elif plot_type == "Pair Plot":
            st.write("### Pair Plot")
            plt.figure(figsize=(10, 6))
            sns.pairplot(data.select_dtypes(include=[np.number]))
            st.pyplot(plt)

        elif plot_type == "Bar Plot":
            x_col = st.selectbox("Category Column (X)", data.select_dtypes(include=['object', 'category']).columns)
            y_col = st.selectbox("Value Column (Y)", data.select_dtypes(include=[np.number]).columns)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=data, x=x_col, y=y_col)
            st.pyplot(plt)

        elif plot_type == "Line Plot":
            x_col = st.selectbox("X-Axis (Time/Sequence)", data.columns)
            y_col = st.selectbox("Y-Axis (Value)", data.select_dtypes(include=[np.number]).columns)
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x=x_col, y=y_col)
            st.pyplot(plt)

        elif plot_type == "Violin Plot":
            col = st.selectbox("Select Column for Violin Plot", data.select_dtypes(include=[np.number]).columns)
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=data, y=col)
            st.pyplot(plt)

        elif plot_type == "Distribution Plot":
            col = st.selectbox("Select Column for Distribution Plot", data.select_dtypes(include=[np.number]).columns)
            plt.figure(figsize=(10, 6))
            sns.displot(data[col], kde=True, height=6, aspect=1.5)
            st.pyplot(plt)

    else:
        st.warning("Please upload and clean the data in the 'Upload Data' section first!")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

if section == "Train Model":
    if 'data' in st.session_state:
        st.header("Train a Model")
        data = st.session_state['data']
        target_col = st.selectbox("Select Target Column", data.columns)

        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        # Define preprocessing for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))  # Replace missing values with mean
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent value
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
        ])

        # Combine preprocessors in a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        # Apply preprocessing
        X = preprocessor.fit_transform(X)

        # Encode target column if it's categorical
        if y.dtype == 'object' or isinstance(y, pd.CategoricalDtype):
            y = LabelEncoder().fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose Model
        model_type = st.selectbox("Choose Model Type", ["Linear Regression", "Logistic Regression", "Ridge", "Lasso",
                                                        "Decision Tree", "Random Forest"])
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

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            if model_type in ["Linear Regression", "Ridge", "Lasso"]:
                y_pred = model.predict(X_test)
                st.write("### R² Score:", r2_score(y_test, y_pred))
            else:
                y_pred = model.predict(X_test)
                st.write("### Accuracy:", accuracy_score(y_test, y_pred))
                st.write("### Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))

            # Save the model
            joblib.dump(model, "trained_model.pkl")
            st.success("Model trained and saved!")
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

st.title('Machine Learning App')

st.write('Hello world!')
