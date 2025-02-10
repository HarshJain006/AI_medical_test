import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pdfplumber
from groq import Groq

# Set up Streamlit page
st.set_page_config(page_title="DNN Prediction & AI Chatbot", page_icon="🤖", layout="wide")
st.title("🧠 DNN Breast Cancer Prediction & AI Chatbot 🤖")

# Sidebar Enhancement
with st.sidebar:
    st.title("🔍 MedAI Features")
    st.info("""
    🤖 **Welcome to MedAI!**

    Capabilities:
    - 📊 Summarize Medical PDFs
    - 🩻 Detect Breast Cancer from Images
    - 💬 AI Chat for Medical Q&A
    """, icon="💡")

    st.markdown("---")
    st.caption("Developed with ❤️ by Harsh Jain")

# Load the trained model
MODEL_PATH = "model.h5"
try:
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error Loading Model: {e}")

# Groq API Setup
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Image Preprocessing
def preprocess_image(img):
    img = img.resize((50, 50))
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    return np.expand_dims(img_array, axis=0)

# 📷 Image Upload
uploaded_file = st.sidebar.file_uploader("📷 Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)

    result_text = "🔴 Breast Cancer Detected" if prediction[0][0] > 0.5 else "🟢 No Cancer Detected"
    st.sidebar.subheader("📊 Prediction Result")
    st.sidebar.markdown(f"**{result_text}**")
    st.sidebar.write(f"Confidence: {prediction[0][0]:.4f}")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(result_text, color="red" if prediction[0][0] > 0.5 else "green")
    ax.axis("off")
    st.pyplot(fig)

# 📄 PDF Upload
pdf_file = st.sidebar.file_uploader("📄 Upload Medical Report (PDF)", type=["pdf"])
if pdf_file:
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    if text:
        st.session_state["medical_report"] = text
        st.subheader("📝 Medical Report Summary")
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Summarize this medical report in simple words:\n{text}"}],
                max_tokens=1024
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ AI Error: {e}")
    else:
        st.warning("No text found in the PDF. It may be an image-based file.")

# 💬 AI Chatbot
st.subheader("💬 Chat with MedAI")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = "You are an AI healthcare assistant."
    if "medical_report" in st.session_state:
        context += f"\nBased on this report:\n{st.session_state['medical_report']}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"{context}\nUser: {prompt}"}],
            max_tokens=1024
        )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    except Exception as e:
        st.error(f"❌ Chat Error: {e}")
