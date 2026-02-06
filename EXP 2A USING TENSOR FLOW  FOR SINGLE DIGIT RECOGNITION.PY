# -----------------------------
# Import Libraries
# -----------------------------
from google.colab import files
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Load MNIST Dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# -----------------------------
# Normalize the data
# -----------------------------
x_train = x_train / 255.0
x_test = x_test / 255.0

# -----------------------------
# Build the Model
# -----------------------------
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# -----------------------------
# Compile the Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train the Model
# -----------------------------
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# -----------------------------
# Evaluate the Model
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", test_acc * 100, "%")

# -----------------------------
# Upload Handwritten Digit Image
# -----------------------------
print("\nUpload your handwritten digit image (black digit on white background)")
uploaded = files.upload()

image_name = list(uploaded.keys())[0]

# -----------------------------
# Preprocess the Image
# -----------------------------
img = Image.open(image_name).convert("L")   # Grayscale
img = ImageOps.invert(img)                  # Invert colors
img = img.resize((28, 28))                  # Resize

img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28, 28)

# -----------------------------
# Predict the Digit
# -----------------------------
prediction = model.predict(img_array)
digit = np.argmax(prediction)

# -----------------------------
# Display the Result
# -----------------------------
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {digit}")
plt.axis('off')
plt.show()

print(f"Predicted digit is: {digit}")


