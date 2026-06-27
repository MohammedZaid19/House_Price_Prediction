import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('D:/Downloads/House_Prices.csv')
print("Dataset : ")
print(df)

# Perform one-hot encoding for 'Brick' and 'Neighborhood' categorical features
df = pd.get_dummies(df, columns=['Brick', 'Neighborhood'], drop_first=True)

# Now select features, including the new one-hot encoded columns
X = df[["SqFt","Bedrooms","Bathrooms","Offers", 'Brick_Yes', 'Neighborhood_North', 'Neighborhood_West']]
y = df["Price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Data Size : ",len(X_train))
print("\nTesting Data Size : ",len(X_test))

# Linear Regression Model
model = LinearRegression()
model.fit(X_train,y_train)
print("\nModel Trained Successfully")

y_pred = model.predict(X_test)
print("\nActual Price : ")
print(y_test.values)
print("\nPredicted Price : ")
print(y_pred)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("\nModel Evaluation : ")
print("\nMean Squared Error : ",mse)
print("\nR2 Score : ",r2)

# Step 9: Take input from the user for Linear Regression
print("\n--- Taking the input for prediction ---")
SqFt_lr = float(input("\nEnter the House Size in terms of SqFt: "))
Bedrooms_lr = int(input("Enter the Number of Bedrooms: "))
Bathrooms_lr = int(input("Enter the Number of Bathrooms: "))
Offers_lr = int(input("Enter the Number of Offers: "))
Brick_lr = input("Is the house made of Brick? (Yes/No): ").lower() == 'yes'
Neighborhood_lr = input("Enter the Neighborhood (East, North, West): ").lower()

# Step 10: Create DataFrame with the same feature names and one-hot encode
new_data_lr = pd.DataFrame({
    "SqFt": [SqFt_lr],
    "Bedrooms": [Bedrooms_lr],
    "Bathrooms": [Bathrooms_lr],
    "Offers": [Offers_lr],
    "Brick_Yes": [Brick_lr],
    "Neighborhood_North": [1 if Neighborhood_lr == 'north' else 0],
    "Neighborhood_West": [1 if Neighborhood_lr == 'west' else 0]
})

# Step 11: Predict the house price using Linear Regression
predicted_price_lr = model.predict(new_data_lr)

# Step 12: Print the result for Linear Regression
print("\n===== House Details  =====")
print(new_data_lr)
print(f"\nPredicted House Price (Linear Regression): {predicted_price_lr[0]:.2f}")
