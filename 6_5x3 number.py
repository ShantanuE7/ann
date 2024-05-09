import numpy as np
import tensorflow as tf

# Create a dataset (5x3 matrices representing digits)
data = np.array([
    [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],  # Digit 0
    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],  # Digit 1
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # Digit 2
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # Digit 3
    [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],  # Digit 4
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # Digit 5
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # Digit 6
    [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],  # Digit 7
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # Digit 8
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]]   # Digit 9
])

# Define labels (one-hot encoded)
labels = np.eye(10)

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(5, 3)),  # Flatten the 5x3 matrix
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (softmax activation)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=500)

# Test the model
test_data = np.array([
    [[1, 1, 1], 
     [0, 0, 1], 
     [1, 1, 1], 
     [0, 0, 1], 
     [1, 1, 1]],
         # Digit 3
    [[1, 1, 1], 
     [1, 0, 1], 
     [1, 0, 1], 
     [1, 0, 1], 
     [1, 1, 1]],
         # digit close to 6
    [[1, 1, 1], 
     [0, 0, 1], 
     [0, 0, 1], 
     [0, 0, 1], 
     [0, 0, 1]] ,
         # Digit 7
])


predictions = model.predict(test_data)
predicted_digits = [np.argmax(prediction) for prediction in predictions]

print(f"Predicted digits: {predicted_digits}")

"""
OUTPUT:

Epoch 500/500
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - accuracy: 0.9000 - loss: 0.3373
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
Predicted digits: [3, 6, 7]

"""
