import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import extract_landmarks

model = load_model("models/sign_model.h5")
labels = ["A", "B", "C", "D"]

def predict_sign(frame):
    data = extract_landmarks(frame)
    if data is None:
        return "No Hand"

    prediction = model.predict(np.array([data]))
    return labels[np.argmax(prediction)]
