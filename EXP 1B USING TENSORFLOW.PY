import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Input data
# -----------------------------
x = np.array([[1, 2, 3]], dtype=float)
y = np.array([[1]], dtype=float)

# -----------------------------
# Build Feed Forward Neural Network
# -----------------------------
model = Sequential([
    Dense(2, activation='sigmoid', input_shape=(3,)),
    Dense(1, activation='sigmoid')
])

# -----------------------------
# Compile the model
# -----------------------------
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# -----------------------------
# Train the model
# -----------------------------
model.fit(x, y, epochs=100, verbose=0)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(x)
print("Prediction:", prediction)

OUTPUT:
Prediction: [[0.5794355]]
