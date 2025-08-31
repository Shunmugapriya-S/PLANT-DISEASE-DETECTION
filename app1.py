import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyttsx3
from twilio.rest import Client
from gtts import gTTS
import os
import playsound
from dotenv import load_dotenv   # ✅ dotenv for credentials

# ✅ Load environment variables
load_dotenv('.env')
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
recipient_number = os.getenv("RECIPIENT_PHONE_NUMBER")

# ✅ Streamlit Page Config
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# ✅ Load Model
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model.h5")

model = load_trained_model()

# ✅ Class Labels
class_names = [
    "Pepper_bell_Bacterial_spot", "Pepperbell__healthy",
    "Potato__Early_blight", "Potato_Late_blight", "Potato__healthy",
    "Tomato__Bacterial_spot", "Tomato_Early_blight", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato__healthy"
]

# ✅ Remedies - English
remedies = {
    "Bacterial": "➡ Avoid overhead watering.\n➡ Remove infected leaves.\n➡ Apply copper-based bactericides.",
    "Fungal": "➡ Ensure proper drainage.\n➡ Apply recommended fungicides.\n➡ Avoid crowding of plants.",
    "Viral": "➡ Remove and destroy infected plants.\n➡ Control insect vectors.\n➡ Avoid planting near infected crops.",
    "Healthy": "✅ Plant is healthy. Keep monitoring regularly."
}

# ✅ Remedies - Tamil
remedies_tamil = {
    "Pepper_bell_Bacterial_spot": "இந்தத் தாவரத்தில் பாக்டீரியா ஸ்பாட் நோய் உள்ளது...",
    "Tomato_Early_blight": "இந்தத் தாவரத்தில் ஆரம்ப நிலை பிளைட் நோய் உள்ளது...",
    "Tomato_Late_blight": "தாமரையில் தாமத பிளைட் நோய் உள்ளது...",
    "Tomato_Leaf_Mold": "இலைமூடி நோய் உள்ளது...",
    "Tomato_Septoria_leaf_spot": "செப்டோரியா இலை ஸ்பாட் உள்ளது...",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "இது இரண்டு புள்ளிகள் கொண்ட சிலந்திப் பூச்சி தொற்றாகும்...",
    "Tomato_Target_Spot": "டார்கெட் ஸ்பாட் நோய் உள்ளது...",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "இது வைரஸ் நோய்...",
    "Tomato_Tomato_mosaic_virus": "மொசைக் வைரஸ் உள்ளது...",
    "Tomato__healthy": "இந்தத் தாவரம் ஆரோக்கியமாக உள்ளது..."
}

# ✅ Disease type mapper
disease_type_map = {
    "Bacterial_spot": "Bacterial",
    "Leaf_Mold": "Fungal",
    "Early_blight": "Fungal",
    "Late_blight": "Fungal",
    "Septoria_leaf_spot": "Fungal",
    "Target_Spot": "Fungal",
    "Tomato_mosaic_virus": "Viral",
    "Tomato_Yellow_Leaf_Curl_Virus": "Viral",
    "healthy": "Healthy"
}

# ✅ Tamil voice support
def speak_tamil(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if "ta" in voice.id.lower() or "tamil" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.say(text)
        engine.runAndWait()
    except:
        try:
            tts = gTTS(text=text, lang='ta')
            tts.save("remedy_tamil.mp3")
            playsound.playsound("remedy_tamil.mp3")
            os.remove("remedy_tamil.mp3")
        except:
            st.warning("🔇 Could not play Tamil voice message.")

# ✅ UI
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a plant leaf image. The app will predict the disease and suggest remedies.")

uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("🔍 Predicting..."):
        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        predicted_label = class_names[predicted_idx]
        confidence = np.max(prediction) * 100

    disease_key = predicted_label.split("_")[-1]
    disease_type = disease_type_map.get(disease_key, "Unknown")

    # ✅ Remedy selection with fallback
    if disease_type in remedies:
        suggestion = remedies[disease_type]
    else:
        suggestion = remedies_tamil.get(predicted_label, "✅ தாவரம் ஆரோக்கியமாக இருக்கலாம். மேலும் கவனிக்கவும்.")

    # ✅ Display results
    st.success(f"🌱 Disease Detected: *{predicted_label}*")
    st.info(f"📊 Confidence: *{confidence:.2f}%*")
    st.warning(f"🧪 Type: *{disease_type}*")
    st.markdown(f"💡 *Suggested Remedy:*\n\n{suggestion}")

    # ✅ English voice
    try:
        engine = pyttsx3.init()
        engine.say(f"Disease detected is {predicted_label}. Confidence is {confidence:.2f} percent.")
        engine.runAndWait()
    except:
        st.warning("🔇 English voice alert failed.")

    # ✅ Tamil voice
    tamil_remedy = remedies_tamil.get(predicted_label, "தகவல் கிடைக்கவில்லை.")
    speak_tamil(tamil_remedy)

    # ✅ Tamil SMS via Twilio
    try:
        if account_sid and auth_token and twilio_number and recipient_number:
            client = Client(account_sid, auth_token)
            tamil_msg = f"""🌿 நோய் கண்டறியப்பட்டது: {predicted_label}
நம்பிக்கை: {confidence:.2f}%
விதிகள்:
{tamil_remedy}"""

            client.messages.create(
                body=tamil_msg,
                from_=twilio_number,
                to=recipient_number
            )
            st.success("✅ Message sent successfully")
        else:
            st.error("⚠️ Twilio credentials not found in environment variables.")
    except Exception as e:
        st.error(f"⚠️ Message not sent: {e}")
