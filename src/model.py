import json
import re
import nltk
import numpy as np
import tensorflow as tf

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

nltk.download("punkt")
nltk.download("stopwords")

MAX_SEQ_LEN = 80
class EmotionPredictor:

    def __init__(
        self,
        model_path="models/bltsm/bilstm_student_adaptive.keras",
        tokenizer_path="models/bltsm/tokenizer.json",
        classes_path="models/bltsm/label_classes.json"
    ):