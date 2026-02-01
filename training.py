import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=20)
model.save("models/sign_model.h5")
