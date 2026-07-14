import re
import nltk

nltk.download('punkt')

class TextPreprocessor:

    def __init__(self):
        pass
def clean_text(self, text):

    text = str(text).lower()

    # Keep punctuation that may express emotion
    text = re.sub(r"[^a-zA-Z\s,!']", " ", text)

    tokens = nltk.word_tokenize(text)

    # Remove only basic articles
    skip_words = {"the", "a", "an"}

    tokens = [
        t for t in tokens
        if t not in skip_words and len(t) > 1
    ]

    return " ".join(tokens) if tokens else text
def get_emotion_keywords(self):

    return {

        "Frustrated": [
            "frustrated", "frustrating", "annoying", "angry",
            "hate", "difficult", "stuck", "wrong answer",
            "keep getting", "tried", "problem", "error"
        ],

        "Curious": [
            "why", "how", "what", "curious", "wonder",
            "interested", "learn", "know more",
            "want to know", "explore", "explain"
        ],

        "Confident": [
            "easy", "solved", "got it", "understood",
            "clear", "finally", "perfect",
            "excellent", "amazing", "done"
        ],

        "Bored": [
            "boring", "bored", "tired",
            "repetitive", "dull", "not interesting",
            "too basic", "sleepy"
        ],

        "Confused": [
            "confused", "lost", "unclear",
            "don't understand", "doesn't make sense",
            "missing", "incomplete", "unsure"
        ]
    }
def calculate_keyword_scores(self, text):

    text_lower = text.lower()

    emotion_keywords = self.get_emotion_keywords()

    emotion_scores = {}

    for emotion, keywords in emotion_keywords.items():

        score = 0

        for keyword in keywords:

            if keyword in text_lower:

                if keyword in [
                    "frustrated",
                    "curious",
                    "confident",
                    "bored",
                    "boring",
                    "confused"
                ]:
                    score += 10
                else:
                    score += 2

        emotion_scores[emotion] = score

    return emotion_scores
def boost_prediction_probabilities(self, probabilities, keyword_scores):

    boosted_probs = probabilities.copy()

    emotions = [
        "Bored",
        "Confident",
        "Confused",
        "Curious",
        "Frustrated"
    ]

    for i, emotion in enumerate(emotions):

        if emotion in keyword_scores:

            boosted_probs[i] += keyword_scores[emotion] * 0.05

    # Renormalize
    total = sum(boosted_probs)

    if total > 0:
        boosted_probs = [p / total for p in boosted_probs]

    return boosted_probs