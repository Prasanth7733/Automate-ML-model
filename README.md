

# Streamlit Machine Learning App

A simple Streamlit app for uploading datasets, training machine learning models, visualizing data, and making predictions. The app supports different machine learning models and offers data cleaning and preprocessing functionality, along with various types of visualizations.

## Features

- **Upload Data**: Users can upload CSV or Excel files and view a preview of the dataset.
- **Data Visualization**: Includes options for scatter plots, histograms, and correlation heatmaps.
- **Train Model**: Choose from various machine learning models (Linear Regression, Logistic Regression, Decision Trees, Random Forests, and more) to train on the dataset.
- **Make Predictions**: Upload new data to make predictions using the trained model.
- **Custom Styling**: The app includes a custom background and styles defined via an external CSS file for a personalized look.

## Installation

1. Clone the repository to your local machine or create a new directory for the app.
   ```bash
   git clone <repository_link>
   ```

2. Navigate to the project directory.
   ```bash
   cd streamlit_ml_app
   ```

3. Install the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes the necessary Python packages:
   - streamlit
   - pandas
   - numpy
   - scikit-learn
   - seaborn
   - matplotlib
   - joblib

## Running the App

1. Make sure you have all dependencies installed as per the instructions above.
2. To run the app, use the following command:
   ```bash
   streamlit run app.py
   ```
3. The app will open in your default browser. If not, you can access it at `http://localhost:8501/`.

## Project Structure

The project directory contains the following files:
- `app.py`: Main Python script containing the logic for the Streamlit app.
- `styles.css`: External CSS file containing custom styles for the app, including background color and sidebar customization.
- `requirements.txt`: List of Python dependencies for the project.
- `trained_model.pkl`: Saved trained machine learning model (generated during training).
- `feature_names.pkl`: Saved feature names used during training.

## Customization

- **Background Style**: Modify the `styles.css` file to change the background color, font styles, or sidebar appearance.
- **Model Configuration**: You can easily add more machine learning models or modify the existing ones in the `Train Model` section of the app.

## Example Use Case

1. Upload a dataset in CSV or Excel format.
2. Visualize the data using scatter plots, histograms, or correlation heatmaps.
3. Train a machine learning model on the dataset (e.g., Random Forest, Linear Regression).
4. After training, save the model and use it to make predictions on new data.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for making it easy to build interactive apps.
- [Scikit-learn](https://scikit-learn.org/) for providing a wide variety of machine learning models.
- [Seaborn](https://seaborn.pydata.org/) for beautiful and informative statistical graphics.
  
---
## Demo App

[[Streamlit App](https://automate-ml-model.streamlit.app/)

