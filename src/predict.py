from src.bert_model import predict_emotion

CONFIDENT_KEYWORDS = [
    "confident", "sure", "definitely", "can do",
    "i know", "certain", "yes"
]

CONFUSED_KEYWORDS = [
    "confused", "don't know", "dont know",
    "maybe", "unclear", "stuck", "help"
]


def analyze_emotion(text):

    result = predict_emotion(text)

    scores = result["scores"]

    text_lower = text.lower()

    # Keyword adjustment only
    if any(word in text_lower for word in CONFIDENT_KEYWORDS):
        scores["Confident"] *= 2.5

    if any(word in text_lower for word in CONFUSED_KEYWORDS):
        scores["Confused"] *= 2.0

    # Normalize
    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}

    primary = max(scores, key=scores.get)

    mixed = [
        emotion
        for emotion, score in scores.items()
        if score >= 0.15
    ]

    return {
        "primary": primary,
        "confidence": round(scores[primary] * 100, 2),
        "mixed_emotions": mixed,
        "scores": scores
    }
EMOTION_RESPONSES = {

    "Confused": {
        "emoji": "🤔",
        "response": "I see you might be confused. Let me break this down step by step.",
        "action": "Show detailed explanation"
    },

    "Frustrated": {
        "emoji": "😣",
        "response": "I understand this is frustrating. Let's try a simpler approach.",
        "action": "Suggest alternative learning path"
    },

    "Confident": {
        "emoji": "🎉",
        "response": "Great! You're making excellent progress. Ready for the next challenge?",
        "action": "Suggest advanced content"
    },

    "Bored": {
        "emoji": "😴",
        "response": "Let's make this more engaging. Here are some interactive exercises.",
        "action": "Show interactive content"
    },

    "Curious": {
        "emoji": "🤓",
        "response": "Excellent question! Here's more in-depth information.",
        "action": "Provide research papers & advanced materials"
    }

}