# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===============================
# 2. SET SEEDS
# ===============================
np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# 3. CLASS NAMES (FIXED ERROR)
# ===============================
class_names = ['Bird', 'Cat', 'Dog']

# ===============================
# 4. LOAD CIFAR-10 DATASET
# ===============================
(x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.cifar10.load_data()

# ===============================
# 5. FILTER BIRD, CAT, DOG
# ===============================
def filter_dataset(x, y):
    mask = np.isin(y.flatten(), [2, 3, 5])
    x = x[mask]
    y = y[mask]

    label_map = {2: 0, 3: 1, 5: 2}  # bird, cat, dog
    y = np.array([label_map[i] for i in y.flatten()])

    return x, y

x_train, y_train = filter_dataset(x_train_full, y_train_full)
x_test, y_test = filter_dataset(x_test_full, y_test_full)

# ===============================
# 6. NORMALIZE & ONE-HOT
# ===============================
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)

# ===============================
# 7. TRAIN / VALIDATION SPLIT
# ===============================
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_train, axis=1)
)

# ===============================
# 8. BUILD CNN MODEL
# ===============================
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Dropout(0.5),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(3, activation='softmax')
])

# ===============================
# 9. COMPILE MODEL
# ===============================
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 10. TRAIN MODEL
# ===============================
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)

# ===============================
# 11. EVALUATE MODEL
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ===============================
# 12. PLOT ACCURACY & LOSS
# ===============================
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.legend()

plt.show()

# ===============================
# 13. VISUALIZE PREDICTIONS
# ===============================
def visualize_predictions(x, y, model, n=12):
    idx = np.random.choice(len(x), n, replace=False)
    images = x[idx]
    true_labels = np.argmax(y[idx], axis=1)

    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(12,8))
    for i in range(n):
        plt.subplot(3,4,i+1)
        plt.imshow(images[i])
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f"T:{class_names[true_labels[i]]}\nP:{class_names[pred_labels[i]]}", color=color)
        plt.axis('off')
    plt.show()

visualize_predictions(x_test, y_test, model)

# ===============================
# 14. SINGLE IMAGE PREDICTION
# ===============================
def predict_single_image(model, image):
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image, verbose=0)
    return class_names[np.argmax(pred)], np.max(pred)

sample_image = x_test[0]
true_label = class_names[np.argmax(y_test[0])]
pred_label, conf = predict_single_image(model, sample_image)

print("\nSingle Image Prediction")
print("True Label:", true_label)
print(f"Predicted: {pred_label} ({conf:.2%})")
