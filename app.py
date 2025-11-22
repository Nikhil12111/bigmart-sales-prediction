import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline + model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("BigMart Outlet Sales Predictor")
st.write("Predict Item_Outlet_Sales for a given product and store configuration.")

st.header("🧾 Item Information")

item_identifier = st.text_input("Item Identifier", "FDA15")
item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

item_type = st.selectbox(
    "Item Type",
    [
        "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy",
        "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
        "Health and Hygiene", "Household", "Meat", "Others",
        "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
    ]
)

item_mrp = st.number_input("Item MRP", min_value=0.0, max_value=500.0, value=150.0, step=1.0)

st.header("🏪 Outlet Information")

outlet_identifier = st.selectbox(
    "Outlet Identifier",
    ["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"]
)

outlet_establishment_year = st.number_input(
    "Outlet Establishment Year", min_value=1980, max_value=2025, value=2000, step=1
)

outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])

outlet_location_type = st.selectbox(
    "Outlet Location Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

outlet_type = st.selectbox(
    "Outlet Type",
    ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
)

# Build input row
input_data = pd.DataFrame([{
    "Item_Identifier": item_identifier,
    "Item_Weight": item_weight,
    "Item_Fat_Content": item_fat_content,
    "Item_Visibility": item_visibility,
    "Item_Type": item_type,
    "Item_MRP": item_mrp,
    "Outlet_Identifier": outlet_identifier,
    "Outlet_Establishment_Year": outlet_establishment_year,
    "Outlet_Size": outlet_size,
    "Outlet_Location_Type": outlet_location_type,
    "Outlet_Type": outlet_type
}])

if st.button("Predict Item Outlet Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Item Outlet Sales: ₹ {prediction:,.2f}")
