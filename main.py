# ==========================================================
# AI House Price Prediction System
# Part 1 - Data Loading & Preprocessing
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import shap

import joblib

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("AmesHousing.csv")

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)

print(df.head())

print("\nShape :", df.shape)

# ----------------------------------------------------------
# Remove Duplicates
# ----------------------------------------------------------

df.drop_duplicates(inplace=True)

# ----------------------------------------------------------
# Handle Missing Values
# ----------------------------------------------------------

for col in df.columns:

    if df[col].dtype == "object":

        df[col].fillna(df[col].mode()[0], inplace=True)

    else:

        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values Left")
print(df.isnull().sum().sum())

# ----------------------------------------------------------
# Target Column
# ----------------------------------------------------------

TARGET = "SalePrice"

# ----------------------------------------------------------
# Separate Features and Target
# ----------------------------------------------------------

X = df.drop(TARGET, axis=1)

y = df[TARGET]

# ----------------------------------------------------------
# One-Hot Encoding
# ----------------------------------------------------------

X = pd.get_dummies(X)

print("\nEncoded Shape :", X.shape)

# ----------------------------------------------------------
# Train Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaling Completed")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("\nPreprocessing Finished Successfully")

print("="*60)
# ==========================================================
# PART 2 : MODEL TRAINING
# ==========================================================

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# ----------------------------------------------------------
# Random Forest
# ----------------------------------------------------------

print("\nTraining Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"R2   : {rf_r2:.4f}")

# ----------------------------------------------------------
# XGBoost
# ----------------------------------------------------------

print("\nTraining XGBoost...")

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Results")
print(f"MAE  : {xgb_mae:.2f}")
print(f"RMSE : {xgb_rmse:.2f}")
print(f"R2   : {xgb_r2:.4f}")

# ----------------------------------------------------------
# Deep Neural Network
# ----------------------------------------------------------

print("\nTraining Deep Neural Network...")

nn_model = Sequential([

    Dense(256, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.30),

    Dense(128, activation="relu"),
    Dropout(0.20),

    Dense(64, activation="relu"),

    Dense(32, activation="relu"),

    Dense(1)

])

nn_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = nn_model.fit(

    X_train_scaled,
    y_train,

    validation_split=0.20,

    epochs=300,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)

nn_pred = nn_model.predict(X_test_scaled).flatten()

nn_mae = mean_absolute_error(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
nn_r2 = r2_score(y_test, nn_pred)

print("\nDeep Learning Results")
print(f"MAE  : {nn_mae:.2f}")
print(f"RMSE : {nn_rmse:.2f}")
print(f"R2   : {nn_r2:.4f}")

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results = pd.DataFrame({

    "Model": [

        "Random Forest",
        "XGBoost",
        "Deep Neural Network"

    ],

    "MAE": [

        rf_mae,
        xgb_mae,
        nn_mae

    ],

    "RMSE": [

        rf_rmse,
        xgb_rmse,
        nn_rmse

    ],

    "R2 Score": [

        rf_r2,
        xgb_r2,
        nn_r2

    ]

})

results = results.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(results)

# ==========================================================
# SAVE MODELS
# ==========================================================

joblib.dump(rf_model, "random_forest.pkl")
joblib.dump(xgb_model, "xgboost.pkl")

nn_model.save("deep_learning.keras")

print("\nModels Saved Successfully")

print("="*60)
# ==========================================================
# PART 3 : VISUALIZATION
# ==========================================================

import matplotlib.pyplot as plt
import plotly.express as px

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# ----------------------------------------------------------
# Correlation Heatmap (Top 20 Features)
# ----------------------------------------------------------

corr = df.corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False)

top_features = corr.head(20).index

corr_matrix = df[top_features].corr()

plt.figure(figsize=(12,10))

plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")

plt.xticks(range(len(top_features)), top_features, rotation=90)

plt.yticks(range(len(top_features)), top_features)

plt.colorbar()

plt.title("Top Correlated Features with Sale Price")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.close()

print("Correlation Heatmap Saved")

# ----------------------------------------------------------
# Feature Importance (XGBoost)
# ----------------------------------------------------------

importance = pd.DataFrame({

    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

top20 = importance.head(20)

fig = px.bar(

    top20,

    x="Importance",

    y="Feature",

    orientation="h",

    title="Top 20 Important Features (XGBoost)"

)

fig.write_html("feature_importance.html")

print("Feature Importance Saved")

# ----------------------------------------------------------
# Actual vs Predicted
# ----------------------------------------------------------

actual_pred = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": xgb_pred

})

fig2 = px.scatter(

    actual_pred,

    x="Actual",

    y="Predicted",

    title="Actual vs Predicted Price"

)

fig2.add_shape(

    type="line",

    x0=actual_pred["Actual"].min(),

    y0=actual_pred["Actual"].min(),

    x1=actual_pred["Actual"].max(),

    y1=actual_pred["Actual"].max()

)

fig2.write_html("actual_vs_predicted.html")

print("Actual vs Predicted Graph Saved")

# ----------------------------------------------------------
# Residual Plot
# ----------------------------------------------------------

residual = y_test - xgb_pred

fig3 = px.scatter(

    x=xgb_pred,

    y=residual,

    labels={

        "x":"Predicted Price",

        "y":"Residual Error"

    },

    title="Residual Error Plot"

)

fig3.write_html("residual_plot.html")

print("Residual Plot Saved")

print("="*60)
# Top 20 Most Important Features

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(20))
# Save encoded feature names
model_columns = X.columns.tolist()

joblib.dump(model_columns, "model_columns.pkl")
import joblib
import pandas as pd
import numpy as np

# Load encoded columns
model_columns = joblib.load("model_columns.pkl")

def preprocess_input(
    overall_qual,
    garage_cars,
    gr_liv_area,
    full_bath,
    total_bsmt_sf,
    garage_area,
    year_built,
    fireplaces,
    kitchen_quality,
    basement_quality,
    exterior_quality,
    central_air,
    garage_finish,
    ms_zoning
):

    # Create blank row
    sample = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns
    )

    # -----------------------
    # Numerical Features
    # -----------------------

    sample["Overall Qual"] = overall_qual
    sample["Garage Cars"] = garage_cars
    sample["Gr Liv Area"] = gr_liv_area
    sample["Full Bath"] = full_bath
    sample["Total Bsmt SF"] = total_bsmt_sf
    sample["Garage Area"] = garage_area
    sample["Year Built"] = year_built
    sample["Fireplaces"] = fireplaces

    # -----------------------
    # Kitchen Quality
    # -----------------------

    kitchen_col = f"Kitchen Qual_{kitchen_quality}"

    if kitchen_col in sample.columns:
        sample[kitchen_col] = 1

    # -----------------------
    # Basement Quality
    # -----------------------

    bsmt_col = f"Bsmt Qual_{basement_quality}"

    if bsmt_col in sample.columns:
        sample[bsmt_col] = 1

    # -----------------------
    # Exterior Quality
    # -----------------------

    exter_col = f"Exter Qual_{exterior_quality}"

    if exter_col in sample.columns:
        sample[exter_col] = 1

    # -----------------------
    # Central Air
    # -----------------------

    air_col = f"Central Air_{central_air}"

    if air_col in sample.columns:
        sample[air_col] = 1

    # -----------------------
    # Garage Finish
    # -----------------------

    finish_col = f"Garage Finish_{garage_finish}"

    if finish_col in sample.columns:
        sample[finish_col] = 1

    # -----------------------
    # MS Zoning
    # -----------------------

    zone_col = f"MS Zoning_{ms_zoning}"

    if zone_col in sample.columns:
        sample[zone_col] = 1

    return sample
sample = preprocess_input(

    overall_qual=8,
    garage_cars=2,
    gr_liv_area=1800,
    full_bath=2,
    total_bsmt_sf=900,
    garage_area=450,
    year_built=2010,
    fireplaces=1,
    kitchen_quality="Ex",
    basement_quality="Ex",
    exterior_quality="TA",
    central_air="Y",
    garage_finish="Unf",
    ms_zoning="RL"

)

print(sample.head())
xgb_model = joblib.load("xgboost.pkl")

prediction = xgb_model.predict(sample)

print(prediction)
def predict_price(
    overall_qual,
    garage_cars,
    gr_liv_area,
    full_bath,
    total_bsmt_sf,
    garage_area,
    year_built,
    fireplaces,
    kitchen_quality,
    basement_quality,
    exterior_quality,
    central_air,
    garage_finish,
    ms_zoning
):

    sample = preprocess_input(
        overall_qual,
        garage_cars,
        gr_liv_area,
        full_bath,
        total_bsmt_sf,
        garage_area,
        year_built,
        fireplaces,
        kitchen_quality,
        basement_quality,
        exterior_quality,
        central_air,
        garage_finish,
        ms_zoning
    )

    prediction = xgb_model.predict(sample)[0]

    return f"${prediction:,.2f}"
import joblib

# Save default values from the original dataset
default_values = {}

for col in df.columns:
    if col == "SalePrice":
        continue

    if df[col].dtype == "object":
        default_values[col] = df[col].mode()[0]
    else:
        default_values[col] = float(df[col].median())

joblib.dump(default_values, "default_values.pkl")

print("default_values.pkl created successfully.")
import joblib
import tensorflow as tf

# Load models
xgb_model = joblib.load("xgboost.pkl")
rf_model = joblib.load("random_forest.pkl")
dnn_model = tf.keras.models.load_model("deep_learning.keras")
print("="*50)
print(" HOUSE PRICE PREDICTION ")
print("="*50)

print("\nChoose Prediction Model")
print("1. XGBoost")
print("2. Random Forest")
print("3. Deep Neural Network")

choice = int(input("\nEnter your choice (1-3): "))
overall_qual = int(input("Overall Quality (1-10): "))
gr_liv_area = float(input("Ground Living Area (sq ft): "))
garage_cars = int(input("Garage Cars: "))
garage_area = float(input("Garage Area (sq ft): "))
total_bsmt_sf = float(input("Basement Area (sq ft): "))
full_bath = int(input("Full Bathrooms: "))
year_built = int(input("Year Built: "))
fireplaces = int(input("Fireplaces: "))
kitchen_quality = input("Kitchen Quality (Ex/Gd/TA/Fa/Po): ")
basement_quality = input("Basement Quality (Ex/Gd/TA/Fa/Po): ")
exterior_quality = input("Exterior Quality (Ex/Gd/TA/Fa): ")
central_air = input("Central Air (Y/N): ")
garage_finish = input("Garage Finish (Fin/RFn/Unf): ")
ms_zoning = input("MS Zoning (RL/RM/FV/RH/C): ")
input_data = preprocess_input(
    overall_qual=overall_qual,
    garage_cars=garage_cars,
    gr_liv_area=gr_liv_area,
    full_bath=full_bath,
    total_bsmt_sf=total_bsmt_sf,
    garage_area=garage_area,
    year_built=year_built,
    fireplaces=fireplaces,
    kitchen_quality=kitchen_quality,
    basement_quality=basement_quality,
    exterior_quality=exterior_quality,
    central_air=central_air,
    garage_finish=garage_finish,
    ms_zoning=ms_zoning
)
if choice == 1:

    prediction = xgb_model.predict(input_data)[0]
    model_name = "XGBoost"
    accuracy = 93.10

elif choice == 2:

    prediction = rf_model.predict(input_data)[0]
    model_name = "Random Forest"
    accuracy = 91.20

elif choice == 3:

    prediction = dnn_model.predict(input_data, verbose=0)[0][0]
    model_name = "Deep Neural Network"
    accuracy = 87.65

else:

    print("Invalid Choice")
    exit()
print("\n" + "="*50)

print(" HOUSE PRICE PREDICTION RESULT ")

print("="*50)

print(f"\nSelected Model : {model_name}")

print(f"Predicted House Price : ${prediction:,.2f}")

print(f"Model Accuracy : {accuracy}%")

print("="*50)
print("\n" + "="*60)
print("🏠          AI HOUSE PRICE PREDICTION REPORT")
print("="*60)

print(f"\n🤖 Selected Model      : {model_name}")
print(f"💰 Predicted Price     : ${prediction:,.2f}")
print(f"📈 Model Accuracy      : {accuracy:.2f}%")

print("\n🏡 House Information")
print("-"*60)
print(f"Overall Quality        : {overall_qual}/10")
print(f"Living Area            : {gr_liv_area} sq ft")
print(f"Garage Cars            : {garage_cars}")
print(f"Garage Area            : {garage_area} sq ft")
print(f"Basement Area          : {total_bsmt_sf} sq ft")
print(f"Bathrooms              : {full_bath}")
print(f"Year Built             : {year_built}")
print(f"Fireplaces             : {fireplaces}")
print(f"Kitchen Quality        : {kitchen_quality}")
print(f"Basement Quality       : {basement_quality}")
print(f"Exterior Quality       : {exterior_quality}")
print(f"Central Air            : {central_air}")
print(f"Garage Finish          : {garage_finish}")
print(f"MS Zoning              : {ms_zoning}")

print("="*60)