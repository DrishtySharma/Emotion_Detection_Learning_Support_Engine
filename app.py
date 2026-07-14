import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from src.predict import analyze_emotion
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_gemini = genai.GenerativeModel("gemini-flash-latest")

def add_to_history(field, problem, emotion, confidence, ai_response, scores):
    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": emotion,
        "confidence": confidence,
        "response": ai_response,
        "scores": scores
    })

st.set_page_config(
    page_title="Emotion Detection Learning Support Engine",
    page_icon="🧠",
    layout="wide"
)
@st.cache_resource
def load_models():
    try:
        from src.bert_model import predict_emotion

        return predict_emotion, "✅ Models Loaded"

    except Exception as e:
      return None, f"❌ Error : {e}"
model, status = load_models()

st.sidebar.success(status)
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []
def save_to_csv(field, problem, emotion, confidence, ai_response):

    try:

        new_example = {
            "text": problem,
            "emotion": emotion.lower(),
            "confidence": confidence,
            "response": ai_response,
            "field": field,
            "timestamp": datetime.now().isoformat()
        }

        if os.path.exists("emotion_response_examples.csv"):

            df = pd.read_csv("emotion_response_examples.csv")

            df = pd.concat(
                [df, pd.DataFrame([new_example])],
                ignore_index=True
            )

        else:

            df = pd.DataFrame([new_example])

        df.to_csv(
            "emotion_response_examples.csv",
            index=False
        )

        if os.path.exists("emotion_response_mapping.csv"):

            mapping_df = pd.read_csv(
                "emotion_response_mapping.csv"
            )

            if emotion not in mapping_df["emotion"].values:

                new_mapping = pd.DataFrame([{
                    "emotion": emotion,
                    "response": ai_response
                }])

                mapping_df = pd.concat(
                    [mapping_df, new_mapping],
                    ignore_index=True
                )

                mapping_df.to_csv(
                    "emotion_response_mapping.csv",
                    index=False
                )

        return True

    except Exception as e:

        st.error(e)

        return False
def get_gemini_response(field, problem, emotion, confidence):
    prompt = f"""
You are a helpful learning assistant.

Student Field: {field}
Emotion: {emotion}
Confidence: {confidence:.2f}

Problem:
{problem}

Provide:
1. Acknowledge the student's emotion.
2. One field-specific learning tip.
3. One encouraging next step.

Keep response short.
"""

    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
      st.error(f"Gemini Error: {e}")
      return None
st.title("🧠 AI Learning Assistant")
st.subheader("Emotion Detection & Personalized Learning Support")
field = st.selectbox(
    "What field are you studying?",
    [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Engineering",
        "Business",
        "Literature",
        "History",
        "Psychology",
        "Other"
    ]
)
st.subheader("⚙️ Settings")

use_ai = st.checkbox(
    "Use AI Response (Gemini)",
    value=True
)

save_data = st.checkbox(
    "Save to CSV for learning",
    value=True
)

show_details = st.checkbox(
    "Show analysis details",
    value=False
)
use_csv_prediction = st.checkbox(
    "Use CSV-based Prediction",
    value=False
)
if use_csv_prediction:

    if os.path.exists("emotion_response_examples.csv"):

        examples = len(
            pd.read_csv("emotion_response_examples.csv")
        )

        st.info(
            f"📄 CSV prediction enabled. {examples} examples available."
        )

    else:

        st.warning(
            "⚠️ emotion_response_examples.csv not found."
        )
placeholders = {

    "Computer Science":
    "Example: I'm confused about recursion.",

    "Mathematics":
    "Example: I don't understand integration.",

    "Physics":
    "Example: I can't solve projectile motion.",

    "Chemistry":
    "Example: Organic reactions are confusing.",

    "Biology":
    "Example: Explain DNA replication.",

    "Engineering":
    "Example: I don't understand circuit design.",

    "Business":
    "Example: Marketing concepts are difficult.",

    "Literature":
    "Example: I can't understand Shakespeare.",

    "History":
    "Example: I'm confused about World War II.",

    "Psychology":
    "Example: Cognitive theories are difficult.",

    "Other":
    "Describe your learning problem."
}

text = st.text_area(

    "Enter your learning problem:",

    placeholder=placeholders[field],

    height=150
)

if st.button("Analyze Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        try:

            with st.spinner("🔍 Analyzing emotion..."):

                result = analyze_emotion(text)

            st.success("✅ Analysis Completed!")

        except Exception as e:

            st.error(f"❌ Error: {e}")

            st.stop()

        st.markdown("### Primary Emotion")
        st.write(result["primary"])

        st.markdown("### Confidence")
        st.write(f"{result['confidence']} %")

        st.markdown("### 🎭 Mixed Emotions")

        emotion_icons = {
            "Confused": "🤔",
            "Frustrated": "😣",
            "Confident": "🎉",
            "Curious": "🤓",
            "Bored": "😴"
        }

        for emotion in result["mixed_emotions"]:
            score = result["scores"][emotion] * 100

            st.write(
                f"{emotion_icons.get(emotion, '🙂')} {emotion} ({score:.2f}%)"
            )

        scores = result["scores"]
        
        if show_details:
            st.markdown("## 📊 Emotion Confidence Scores")

            sorted_scores = sorted(
            result["scores"].items(),
            key=lambda x: x[1],
            reverse=True
            )

            for emotion, score in sorted_scores:
                percentage = score * 100

                st.write(f"{emotion}: {percentage:.2f}%")

                st.progress(float(score))

        df = pd.DataFrame({
            "Emotion": list(scores.keys()),
            "Confidence": [v * 100 for v in scores.values()]
        })

        fig = px.bar(
            df,
            x="Emotion",
            y="Confidence",
            title="Emotion Confidence Scores"
        )

        st.plotly_chart(fig, use_container_width=True)
        if show_details:

         st.markdown("### Emotion Scores")

         st.write(result["scores"])
        # Personalized Suggestions

        st.markdown("## Learning Suggestion")

        suggestions = {
            "Bored":
                "Try a short interactive quiz or take a short break before continuing.",

            "Confident":
                "Great! Move on to more challenging concepts.",

            "Confused":
                "Revise the topic step-by-step and watch a short explanation video.",

            "Curious":
                "Explore additional articles or practice projects related to this topic.",

            "Frustrated":
                "Relax for a few minutes and solve easier questions first."
        }
        emotion = result["primary"]
        confidence = result["confidence"]

        st.info(suggestions[emotion])

        if use_ai:
            ai_response = get_gemini_response(
                field,
                text,
                emotion,
                confidence
            )

            if ai_response is None:
                ai_response = suggestions[emotion]

        else:
            ai_response = suggestions[emotion]

        st.markdown("## 🤖 AI Learning Support")
        st.success(ai_response)

        if save_data:
            save_to_csv(
                field=field,
                problem=text,
                emotion=emotion,
                confidence=confidence,
                ai_response=ai_response
            )

            st.success("💾 Interaction saved.")

        add_to_history(
            field,
            text,
            emotion,
            confidence,
            ai_response,
            result["scores"]
        )


def add_to_history(field, problem, emotion, confidence, ai_response, scores):

    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": emotion,
        "confidence": confidence,
        "response": ai_response,
        "scores": scores
    })


st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session History")

st.sidebar.write(
    f"Total Interactions: {len(st.session_state.emotion_history)}"
)

if st.sidebar.button("🗑 Clear History"):
    st.session_state.emotion_history = []
    st.rerun()

for item in st.session_state.emotion_history[-3:]:
    st.sidebar.write(
        f"**{item['field']}**"
    )

    st.sidebar.write(
        f"{item['emotion']} ({item['confidence']:.2f})"
    )

    st.sidebar.write("---")


if os.path.exists("emotion_response_examples.csv"):
    csv_count = len(
        pd.read_csv("emotion_response_examples.csv")
    )

    st.sidebar.write(f"CSV Examples: {csv_count}")


tab1, tab2 = st.tabs(["📊 Analytics", "📜 History"])

with tab1:

    st.subheader("Analytics")

    emotion_tab, field_tab, summary_tab = st.tabs(
        ["😊 Emotions", "📚 Fields", "📋 Summary"]
    )

    if len(st.session_state.emotion_history) > 0:

        total = len(st.session_state.emotion_history)

        emotions = [
            item["emotion"]
            for item in st.session_state.emotion_history
        ]

        confidence = [
            item["confidence"]
            for item in st.session_state.emotion_history
        ]

        most_common = max(
            set(emotions),
            key=emotions.count
        )

        avg_conf = sum(confidence) / len(confidence)

        df = pd.DataFrame(
            st.session_state.emotion_history
        )

        with summary_tab:

            st.metric("Total Queries", total)

            st.metric("Most Common Emotion", most_common)

            st.metric(
                "Average Confidence",
                f"{avg_conf:.2f}%"
            )

        with emotion_tab:

            st.subheader("🥧 Emotion Distribution")

            fig = px.pie(
                df,
                names="emotion",
                title="Emotion Distribution Across Sessions"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with field_tab:

            st.subheader("📚 Emotion by Study Field")

        field_df = pd.DataFrame(
            st.session_state.emotion_history
        )

        fig = px.bar(
            field_df,
            x="field",
            color="emotion",
            title="Emotion Distribution by Study Field",
            barmode="group"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    else:

        st.info("No analytics available yet.")
with tab2:

    st.subheader("Recent History")

    if st.session_state.emotion_history:

        for item in reversed(st.session_state.emotion_history):

            st.write(f"**Field:** {item['field']}")
            st.write(f"**Emotion:** {item['emotion']}")
            st.write(f"**Confidence:** {item['confidence']:.2f}%")
            st.write("---")

    else:
        st.info("No history yet.")
                