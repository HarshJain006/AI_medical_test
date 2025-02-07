import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pdfplumber
import os
from groq import Groq

# Set up Streamlit page
st.set_page_config(page_title="DNN Prediction & AI Chatbot", page_icon="🤖", layout="wide")
st.title("🧠 DNN Breast Cancer Prediction & AI Chatbot 🤖")

# Load the trained model
MODEL_PATH = "model.h5"
try:
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error Loading Model: {e}")

# Set API key directly (if not using .env)
GROQ_API_KEY = "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"


# Groq API Client
client = Groq(api_key=GROQ_API_KEY)

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((50, 50))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize
    if img_array.ndim == 2:  # Convert grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# 📷 Image Upload for Breast Cancer Detection
st.sidebar.subheader("📷 Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose a PNG/JPG Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_img = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(processed_img)
    result_text = "🔴 Breast Cancer Detected" if prediction[0][0] > 0.5 else "🟢 No Cancer Detected"

    # Display result
    st.sidebar.subheader("📊 Prediction Result")
    st.sidebar.markdown(f"**{result_text}**")
    st.sidebar.write(f"Confidence Score: {prediction[0][0]:.4f}")

    # Show image with prediction label
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(result_text, color="red" if prediction[0][0] > 0.5 else "green")
    ax.axis("off")
    st.pyplot(fig)

# 📄 Medical Report Upload & Extraction
st.sidebar.subheader("📄 Upload Medical Report (PDF)")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if pdf_file:
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    if text:
        st.session_state["medical_report"] = text  # Store extracted text

        # Summarize using AI
        st.subheader("📝 Medical Report Summary")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": f"Summarize this medical report in simple words:\n{text}"}],
            model="llama-3.3-70b-versatile",
        )
        st.write(response.choices[0].message.content)
    else:
        st.warning("Could not extract text from the PDF. It may be an image-based PDF.")

# 💬 AI Chatbot Section
st.subheader("💬 Chat with AI (Mixtral-8x7b-32768)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for chatbot
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Include medical report in chatbot context if available
    context = "You are an AI healthcare assistant. Answer based on the following medical report:\n"
    if "medical_report" in st.session_state:
        context += st.session_state["medical_report"] + "\n\n"

    context += f"User question: {prompt}"

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": context}],
            max_tokens=32767
        )
        full_response = response.choices[0].message.content
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info("This app uses a trained DNN model to predict breast cancer from images and includes a chatbot powered by Mixtral-8x7b-32768.")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Made with ❤️ using Streamlit & Groq API**")
