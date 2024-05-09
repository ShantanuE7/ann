import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset for XOR
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

# Expected output for XOR
expected_output = np.array([[0],
                             [1],
                             [1],
                             [0]])

# Initialize weights and biases
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights and biases initialization
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

# Learning rate
learning_rate = 0.01
epochs = 2000000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    # Calculate error
    error = expected_output - predicted_output
    
    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Test the network
hidden_layer_activation = np.dot(inputs, hidden_weights) + hidden_bias
hidden_layer_output = sigmoid(hidden_layer_activation)

output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
predicted_output = sigmoid(output_layer_activation)

print("Predicted XOR output:")
print(predicted_output)

"""
OUTPUT:
Predicted XOR output:
[[0.01314365]
 [0.98873386]
 [0.9887304 ]
 [0.01162932]]

"""