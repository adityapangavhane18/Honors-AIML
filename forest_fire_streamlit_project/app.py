
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from train_model import train_and_save_model
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Forest Fire Prediction", layout="centered")

st.title("🔥 Forest Fire Burned Area Prediction")
st.write("Predict the burned area of forest fires in the northeast region of Portugal.")

model, month_encoder, day_encoder = train_and_save_model()

st.header("Enter Input Values")

x_coord = st.number_input("X Coordinate", min_value=1, max_value=9, value=7)
y_coord = st.number_input("Y Coordinate", min_value=2, max_value=9, value=5)

month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
day = st.selectbox("Day", ["mon","tue","wed","thu","fri","sat","sun"])

ffmc = st.number_input("FFMC", value=90.2)
dmc = st.number_input("DMC", value=120.3)
dc = st.number_input("DC", value=650.1)
isi = st.number_input("ISI", value=8.2)
temp = st.number_input("Temperature", value=25.4)
rh = st.number_input("Relative Humidity", value=35)
wind = st.number_input("Wind", value=4.5)
rain = st.number_input("Rain", value=0.0)

if st.button("Predict Burned Area"):
    input_data = pd.DataFrame([{
        "X": x_coord,
        "Y": y_coord,
        "month": month_encoder.transform([month])[0],
        "day": day_encoder.transform([day])[0],
        "FFMC": ffmc,
        "DMC": dmc,
        "DC": dc,
        "ISI": isi,
        "temp": temp,
        "RH": rh,
        "wind": wind,
        "rain": rain
    }])

    prediction_log = model.predict(input_data)[0]
    prediction_area = np.expm1(prediction_log)

    st.success(f"Predicted Burned Area: {prediction_area:.2f} hectares")
