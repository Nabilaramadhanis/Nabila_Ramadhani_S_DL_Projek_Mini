import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺")

# Judul Aplikasi
st.title("🩺 Prediksi Diabetes Menggunakan Machine Learning")

# Load Model
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("❌ Model tidak ditemukan. Pastikan 'model.pkl' dan 'scaler.pkl' ada di repository GitHub kamu.")
    st.stop()

# Upload file CSV
uploaded_file = st.file_uploader("📂 Upload file CSV untuk prediksi", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Data yang diunggah:")
        st.dataframe(data)

        # Pastikan kolom sesuai dataset diabetes
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            st.warning(f"⚠️ Kolom berikut hilang di file CSV kamu: {missing}")
        else:
            data_scaled = scaler.transform(data[required_columns])
            pred = model.predict(data_scaled)

            st.subheader("📌 Hasil Prediksi")
            result = ["Tidak Diabetes" if x == 0 else "Diabetes" for x in pred]

            data["Prediksi"] = result
            st.success("🎉 Prediksi Berhasil!")

            st.dataframe(data)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")

else:
    st.info("Silakan upload file CSV untuk memulai prediksi.")
