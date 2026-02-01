import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands.Hands()

def extract_landmarks(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img)

    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks).flatten()

    return None
