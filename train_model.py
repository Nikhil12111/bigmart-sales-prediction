import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle

# Load training data
df = pd.read_csv("Train.csv")

# Target and features
y = df["Item_Outlet_Sales"]
X = df.drop("Item_Outlet_Sales", axis=1)

# Columns
numeric_features = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]
categorical_features = [
    "Item_Identifier",
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type"
]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# XGBoost model
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

# Full pipeline
regressor = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
regressor.fit(X_train, y_train)

# Evaluate
# Evaluate
y_pred = regressor.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5

r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.3f}")


# Save pipeline + model together
with open("model.pkl", "wb") as f:
    pickle.dump(regressor, f)

print("Saved model to model.pkl")
