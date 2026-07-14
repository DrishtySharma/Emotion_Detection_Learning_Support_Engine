# 🎓 Emotion Detection & Learning Support Engine

An AI-powered web application that detects a student's emotional state from text and provides personalized learning support using Machine Learning and Large Language Models.

---

## 📌 Project Overview

The **Emotion Detection & Learning Support Engine** is designed to understand students' emotions from their written responses and provide intelligent guidance to improve their learning experience.

The system combines emotion classification with AI-generated learning support, helping students stay motivated and receive personalized recommendations.

---

## ✨ Features

* 😊 Detects emotions from text input
* 🧠 Supports multiple emotion classes
* 📊 Displays confidence scores
* 🤖 AI-generated personalized learning suggestions
* 📈 Emotion analytics dashboard
* 💻 Interactive Streamlit interface
* 📝 Logs user interactions for analysis

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning & AI

* TensorFlow / Keras
* Hugging Face Transformers (BERT)
* Scikit-learn

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly

---

## 📂 Project Structure

```
Emotion_Detection_Learning_Support_Engine/
│
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── bert_model.py
│   ├── train.py
│   └── predict.py
│
├── data/        (Ignored in GitHub)
├── models/      (Ignored in GitHub)
└── .gitignore
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/DrishtySharma/Emotion_Detection_Learning_Support_Engine.git
```

Move into the project directory

```bash
cd Emotion_Detection_Learning_Support_Engine
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Dataset

This project uses publicly available emotion datasets for training and evaluation.

Examples include:

* GoEmotions
* ISEAR Dataset
* Emotion Text Dataset
* Empathetic Dialogues

---

## 📈 Workflow

1. User enters text.
2. Text is preprocessed.
3. Emotion is predicted using trained models.
4. Confidence score is generated.
5. AI provides personalized learning guidance.
6. Results are displayed on the Streamlit dashboard.

---

## 🎯 Future Improvements

* Voice emotion recognition
* Facial emotion detection
* Student performance prediction
* Personalized study planner
* Learning history dashboard

---

## 👩‍💻 Author

**Drishty Sharma**

B.Tech Computer Science Engineering

---

## ⭐ Acknowledgements

* SmartBridge
* Kaggle
* Hugging Face
* TensorFlow
* Streamlit

---

## 📄 License

This project is developed for educational purposes.
