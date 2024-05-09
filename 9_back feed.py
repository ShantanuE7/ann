import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, X):
        # Hidden layer
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)
        
        # Output layer
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.output_activation  # Linear activation for regression
        
    def backward_propagation(self, X, y):
        # Calculate error
        self.error = y - self.predicted_output
        
        # Compute gradients for output layer
        gradient_predicted_output = self.error
        
        # Compute gradients for hidden layer
        error_hidden_layer = gradient_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(gradient_predicted_output) * self.learning_rate
        self.bias_output += np.sum(gradient_predicted_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate
        
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward propagation
            self.forward_propagation(X)
            
            # Backward propagation
            self.backward_propagation(X, y)
            
            # Calculate loss (MSE)
            loss = np.mean(np.square(y - self.predicted_output))
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                
    def predict(self, X):
        self.forward_propagation(X)
        return self.predicted_output

# Example usage
if __name__ == "__main__":
    # Input data (two real-valued numbers)
    X = np.array([[0.1, 0.2],
                  [0.2, 0.3],
                  [0.4, 0.1],
                  [0.5, 0.4]])
    
    # Output data (sum of the two numbers)
    y = np.array([[0.3], [0.5], [0.5], [0.9]])
    
    # Input, Hidden, Output sizes
    input_size = X.shape[1]
    hidden_size = 3
    output_size = y.shape[1]
    
    # Learning rate
    learning_rate = 0.1
    
    # Create a neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    # Train the neural network
    nn.train(X, y, epochs=100000)
    
    # Test the neural network
    test_data = np.array([[0.2, 0.4], [0.6, 0.1]])
    predictions = nn.predict(test_data)
    
    print("Predictions:")
    print(predictions)

    """
    OUTPUT:
    Epoch 47000, Loss: 2.4153278103444627e-06
    Epoch 48000, Loss: 2.341237682969071e-06
    Epoch 49000, Loss: 2.2696352335949433e-06
    Predictions:
    [[0.60272782]
    [0.69942557]]
    
    """
