import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Muat Model dan Scaler ---
# File-file ini harus ada di direktori yang sama dengan app.py
try:
    model = pickle.load(open('model_diabetes.pkl', 'rb'))
    scaler = pickle.load(open('scaler_diabetes.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model atau Scaler belum ditemukan. Pastikan 'model_diabetes.pkl' dan 'scaler_diabetes.pkl' ada.")
    st.stop()


# --- Fungsi Prediksi ---
def predict_diabetes(input_data):
    # Urutan fitur HARUS SAMA dengan saat model dilatih:
    # 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'

    # Ubah input_data menjadi DataFrame
    input_df = pd.DataFrame([input_data], columns=input_data.keys())

    # Lakukan Scaling
    scaled_data = scaler.transform(input_df)

    # Lakukan Prediksi
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    return prediction[0], prediction_proba[0]


# --- Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")

st.title("Aplikasi Prediksi Risiko Diabetes 🩺")
st.markdown("Masukkan data pasien di bawah ini untuk memprediksi risiko diabetes.")

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=17, value=1)
    glucose = st.number_input("Glukosa Plasma (Glucose)", min_value=40, max_value=200, value=120)
    blood_pressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=40, max_value=122, value=70)

with col2:
    skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness)", min_value=0, max_value=99, value=20)
    insulin = st.number_input("Insulin Serum (Insulin)", min_value=0, max_value=846, value=79)
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=15.0, max_value=67.1, value=32.0, format="%.1f")

with col3:
    dpf = st.number_input("Fungsi Silsilah Diabetes (DPF)", min_value=0.078, max_value=2.42, value=0.47, format="%.3f")
    age = st.number_input("Usia (Age)", min_value=21, max_value=100, value=30)


# Tombol Prediksi
if st.button("Prediksi Risiko Diabetes"):
    # Kumpulkan Input
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    # Lakukan Prediksi
    result, proba = predict_diabetes(input_data)

    st.subheader("Hasil Prediksi:")

    if result == 1:
        st.error(f"Pasien **TERKENA** Diabetes. (Probabilitas: {proba[1]*100:.2f}%)")
        st.balloons()
    else:
        st.success(f"Pasien **TIDAK TERKENA** Diabetes. (Probabilitas: {proba[0]*100:.2f}%)")

    st.markdown("---")
    st.info(f"Probabilitas Terkena Diabetes: **{proba[1]*100:.2f}%** | Probabilitas Tidak Terkena: **{proba[0]*100:.2f}%**")
