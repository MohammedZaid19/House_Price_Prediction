🏠 House Price Prediction System

A Machine Learning project that predicts house prices using Multiple Linear Regression and Random Forest Regressor, with an interactive Streamlit Web Application.

📖 Project Overview

The House Price Prediction System is an end-to-end Machine Learning project that estimates the selling price of a house based on various property features.

The project demonstrates the complete Machine Learning workflow, beginning with data preprocessing and ending with deployment as a Streamlit web application.

To improve understanding of machine learning algorithms, the project implements and compares two popular regression models:

📈 Multiple Linear Regression
🌲 Random Forest Regressor

Their performances are evaluated using standard regression metrics to determine the better-performing model.

✨ Features
📂 Load Housing Dataset
🧹 Data Preprocessing
🔄 One-Hot Encoding
📊 Train-Test Split
📈 Multiple Linear Regression
🌲 Random Forest Regressor
📊 Model Comparison
📉 Actual vs Predicted Graph
📏 Performance Evaluation
💾 Model Saving using Joblib
🌐 Streamlit Web Application
🏠 Real-Time House Price Prediction
📋 User-Friendly Interface
🤖 Machine Learning Algorithms
📈 Multiple Linear Regression

Multiple Linear Regression predicts house prices by modeling a linear relationship between multiple input features and the target variable.

Advantages
Simple
Fast Training
Easy to Interpret
Displays Coefficients
Suitable for Linear Relationships
🌲 Random Forest Regressor

Random Forest Regressor is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.

Advantages
Handles Non-linear Data
Higher Prediction Accuracy
Reduces Overfitting
More Robust than Single Decision Trees
Works well on Complex Datasets
📂 Dataset

Dataset Used

House_Prices.csv

Input Features (X)
Feature	Description
SqFt	House Size
Bedrooms	Number of Bedrooms
Bathrooms	Number of Bathrooms
Offers	Number of Offers
Brick	Brick House (Yes / No)
Neighborhood	House Location
Target Variable (Y)
Target
Price
⚙ Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Joblib
Streamlit
Git
GitHub
🧠 Machine Learning Workflow
           House_Prices.csv
                   │
                   ▼
        Data Preprocessing
                   │
                   ▼
        One-Hot Encoding
                   │
                   ▼
        Feature Selection
                   │
                   ▼
        Train-Test Split
                   │
          ┌────────┴────────┐
          ▼                 ▼
 Multiple Linear      Random Forest
    Regression          Regressor
          │                 │
          └────────┬────────┘
                   ▼
          Model Evaluation
                   │
                   ▼
          Model Comparison
                   │
                   ▼
         Save Best Model (.pkl)
                   │
                   ▼
        Streamlit Web Application
                   │
                   ▼
        House Price Prediction
📊 Model Evaluation

The models are evaluated using:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score

These metrics are used to compare the prediction performance of both algorithms.

📉 Data Visualization

The project generates:

Actual vs Predicted House Price Graph
Regression Performance Visualization

These visualizations help analyze prediction quality.

💾 Model Serialization

The trained model is saved using Joblib.

joblib.dump(model, "house_price_model.pkl")

The saved model is then loaded directly into the Streamlit application.

🌐 Streamlit Web Application

The application allows users to:

Enter House Size
Enter Bedrooms
Enter Bathrooms
Enter Offers
Select Brick Type
Select Neighborhood
Predict House Price Instantly
📁 Project Structure
House_Price_Prediction/
│
├── app.py
├── main.py
├── House_Prices.csv
├── house_price_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
🚀 Installation
Install Dependencies
pip install -r requirements.txt
Train the Model
python main.py
Launch Streamlit
streamlit run app.py
📊 Model Comparison
Metric	Multiple Linear Regression	Random Forest Regressor
Mean Squared Error	Your Result	Your Result
Mean Absolute Error	Your Result	Your Result
RMSE	Your Result	Your Result
R² Score	Your Result	Your Result

🏡 Sample Prediction
Feature	Example
SqFt	2200
Bedrooms	4
Bathrooms	3
Offers	2
Brick	Yes
Neighborhood	West

Through this project, I learned:

Data Preprocessing
Feature Engineering
One-Hot Encoding
Multiple Linear Regression
Random Forest Regression
Model Evaluation
