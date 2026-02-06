import numpy as np

# Input layer
inputs = np.array([1, 2, 2])

# Weights and bias for first layer
weights1 = np.array([
    [3, 4],
    [5, 6],
    [7, 8]
])
bias1 = np.array([1, 1])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------- Forward Propagation (Layer 1) --------
z1 = np.dot(inputs, weights1) + bias1
activated_output = sigmoid(z1)

print("Activated output (Layer 1):", activated_output)

# Weights and bias for output layer
weights2 = np.array([
    [0.15],
    [0.25]
])
bias2 = np.array([0.8])

# -------- Forward Propagation (Output Layer) --------
z2 = np.dot(activated_output, weights2) + bias2
final_output = sigmoid(z2)

print("Final output:", final_output)

# -------- Loss Calculation --------
y = 1
loss = (y - final_output) ** 2
print("Loss:", loss)

OUTPUT:

Activated output (Layer 1): [1. 1.]
Final output: [0.76852478]
Loss: [0.05358078]
Updated weights2: [[0.15823563]
 [0.25823563]]
