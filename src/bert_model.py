from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

MODEL_PATH = "models/bert_emotion_model_final"

emotion_labels = {
    0: "Bored",
    1: "Confident",
    2: "Confused",
    3: "Curious",
    4: "Frustrated"
}

# SmartBridge Class Weights
CLASS_WEIGHTS = np.array([1.2, 1.8, 0.6, 1.0, 1.4])

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict_emotion(text):

    cleaned_text = text.strip()

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].numpy()

    # Apply class weighting
    weighted_probs = probs * CLASS_WEIGHTS
    weighted_probs = weighted_probs / weighted_probs.sum()

    pred_idx = np.argmax(weighted_probs)

    return {
        "emotion": emotion_labels[pred_idx],
        "confidence": float(weighted_probs[pred_idx]),
        "scores": {
            emotion_labels[i]: float(weighted_probs[i])
            for i in range(len(weighted_probs))
        },
        "cleaned_text": cleaned_text
    }