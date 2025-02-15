# **MedAI: AI-Powered Medical Diagnosis & Chatbot**

## 📌 **Project Overview**

MedAI is an AI-powered healthcare assistant that leverages **Deep Learning** and **Natural Language Processing (NLP)** to:

- 🩻 **Detect breast cancer** from medical images.
- 📄 **Summarize medical reports** from PDFs.
- 💬 **Provide AI-driven medical insights** through an intelligent chatbot.

## 🔥 **Key Features**

- **Breast Cancer Prediction:** Upload an image, and the AI model detects signs of cancer.
- **Medical Report Summarization:** Extracts key information from PDF reports.
- **AI-Powered Chatbot:** Answers medical-related queries using **Groq API & Llama 3.3-70B**.

## 🛠 **Tech Stack**

- **Python**: Core language for AI model execution.
- **Streamlit**: Interactive UI for a seamless experience.
- **TensorFlow/Keras**: Deep learning models for cancer detection.
- **Ollama (LLM) & Groq API**: AI chatbot integration for medical Q&A.
- **PDFPlumber**: Extracts text from medical PDF reports.
- **Matplotlib & PIL**: Visualizes uploaded images.

## 🚀 **How It Works**

1. **Upload an Image (Mammogram/X-ray):** The AI model predicts the likelihood of cancer.
2. **Upload a Medical PDF Report:** AI extracts key insights and summarizes the findings.
3. **Chat with MedAI:** Users can ask medical questions, and the chatbot provides intelligent responses.

## 📂 **Folder Structure**

```
MedAI/
│── try.py                          # Main application integrating all modules
│── requirements.txt                # List of dependencies
│── README.md                       # Project documentation
│── model.h5                        # learned parameters
```

## 📥 **Installation & Setup**

### **Step 1: Clone the Repository**

```bash
https://github.com/HarshJain006/AI_medical_test.git
cd AI_medical_test
```

### **Step 2: Set Up Virtual Environment**

```bash
python -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate     # For Windows
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Run the Application**

```bash
streamlit run try.py
```

## 🎯 **Usage Guide**

1. **Navigate to the sidebar** to select a feature (Breast Cancer Detection, Report Summarization, Chatbot).
2. **Upload images or PDFs** for AI-based analysis.
3. **Receive instant results** with confidence scores and medical insights.

## 🔍 **Future Enhancements**

- 🏥 **Expand Disease Detection** to include **pneumonia, lung cancer, and diabetic retinopathy**.
- 🗣️ **Voice-Enabled Chatbot** for hands-free medical consultations.
- 🌍 **Multilingual Support** for a global user base.

## 🤝 **Contributors**

- **[Harsh Jain]** – AI Engineer & Developer
- **Community Contributions** – Open for collaboration!

## 📜 **License**

This project is **open-source** and available under the **MIT License**.

---

🚀 **MedAI is designed to make AI-powered medical diagnostics more accessible. Join us in making healthcare smarter!**

