# app.py
import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import urllib.parse
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🫁 Lung Cancer CDSS",
    layout="centered"
)

# ---------------- SETTINGS ----------------
IMG_SIZE = 224
MODEL_PATH = "lung_cancer_model.keras"

CLASS_NAMES = [
    "lung_adenocarcinomas",
    "lung_normal",
    "lung_squamous_cell_carcinomas"
]

# ---------------- MEDICAL GUIDANCE ----------------
CANCER_GUIDANCE = {
    "lung_adenocarcinomas": {
        "stage": "Early to Mid Stage (Stage I – III)",
        "doctor": "Pulmonologist / Medical Oncologist"
    },
    "lung_squamous_cell_carcinomas": {
        "stage": "Mid to Advanced Stage (Stage II – IV)",
        "doctor": "Oncologist / Thoracic Surgeon"
    },
    "lung_normal": {
        "stage": "No cancer detected",
        "doctor": "General Physician / Pulmonologist"
    }
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(path):
    return keras.models.load_model(path)

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- PREDICTION ----------------
def predict(model, img):
    probs = model.predict(img)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    return idx, confidence, probs

# ---------------- CLINICAL SCORE ----------------
def calculate_clinical_score(age, smoker, family_history, symptoms):
    score = 0
    if age >= 50:
        score += 2
    elif age >= 35:
        score += 1
    if smoker == "Yes":
        score += 3
    if family_history == "Yes":
        score += 2
    score += len(symptoms)
    return score

# ---------------- FINAL RISK ----------------
def risk_level(score):
    if score >= 8:
        return "🔴 HIGH RISK"
    elif score >= 4:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# ---------------- PINCODE → LAT/LON ----------------
def get_lat_lon_from_pincode(pincode):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={pincode},India"
    headers = {"User-Agent": "LungCancer-CDSS"}
    res = requests.get(url, headers=headers)
    data = res.json()
    if len(data) == 0:
        return None, None
    return data[0]["lat"], data[0]["lon"]

# ---------------- MAIN APP ----------------
def main():
    st.title("🫁 Lung Cancer Clinical Decision Support System")
    st.write("AI-based lung cancer screening + clinical risk analysis")

    model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader(
        "📤 Upload Lung Histopathology Image",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        processed = preprocess_image(image)
        class_idx, confidence, probs = predict(model, processed)

        predicted_class = CLASS_NAMES[class_idx]
        guide = CANCER_GUIDANCE[predicted_class]

        image_score = int(confidence * 10)

        st.subheader("🧪 AI Image Analysis")
        st.success(f"**Prediction:** {predicted_class.replace('_',' ').title()}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")
        st.write(f"**Possible Stage:** {guide['stage']}")

        # ---------------- PATIENT FORM ----------------
        st.markdown("## 🧾 Patient Medical History")

        with st.form("patient_form"):
            age = st.number_input("Age", min_value=1, max_value=120)
            smoker = st.selectbox("Smoking History", ["No", "Yes"])
            family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
            symptoms = st.multiselect(
                "Symptoms",
                ["Persistent Cough", "Breathlessness", "Weight Loss", "Chest Pain"]
            )
            pincode = st.text_input("City or Pincode")
            submit = st.form_submit_button("Assess Risk")

        if submit:
            clinical_score = calculate_clinical_score(
                age, smoker, family_history, symptoms
            )

            final_score = image_score + clinical_score
            risk = risk_level(final_score)

            st.subheader("📊 Final Risk Assessment")
            st.write(f"🧠 Image Risk Score: {image_score}")
            st.write(f"🩺 Clinical Risk Score: {clinical_score}")
            st.success(f"**Overall Risk Level:** {risk}")

            # ---------------- DOCTOR FINDER ----------------
            st.markdown("## 🏥 Doctor Finder")

            lat, lon = get_lat_lon_from_pincode(pincode)
            doctor_type = guide["doctor"]

            if lat and lon:
                maps_url = (
                    f"https://www.google.com/maps/search/"
                    f"{urllib.parse.quote(doctor_type)}/@{lat},{lon},12z"
                )
                st.write(f"📍 Location detected for **{pincode}**")
                st.write(f"👨‍⚕️ Recommended Specialist: **{doctor_type}**")
                st.markdown(f"[🗺️ Open Doctors Near You]({maps_url})")
            else:
                st.error("❌ Invalid pincode / location")

            # ---------------- CARE PLAN ----------------
            st.markdown("## 🩺 Suggested Care Plan")

            if predicted_class != "lung_normal":
                st.write(
                    "- Monthly hospital visits\n"
                    "- CT Scan / Biopsy if advised\n"
                    "- Stop smoking immediately\n"
                    "- Wear mask in polluted areas\n"
                    "- Nutritious diet & breathing exercises"
                )
            else:
                st.write(
                    "- Routine checkup every 6 months\n"
                    "- Avoid smoking & pollution\n"
                    "- Maintain healthy lifestyle"
                )

            # ---------------- DISCLAIMER ----------------
            st.warning(
                "⚠️ **Disclaimer:** This system is for academic demonstration only. "
                "It does NOT replace professional medical diagnosis."
            )

    st.markdown("---")
    st.caption("AI-powered Lung Cancer CDSS | College Project")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
