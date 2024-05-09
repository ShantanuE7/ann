import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        
    def train(self, patterns):
        # Initialize weights
        for pattern in patterns:
            pattern_matrix = np.reshape(pattern, (self.num_neurons, 1))
            self.weights += np.dot(pattern_matrix, pattern_matrix.T)
        
        # Set diagonal elements of weights to zero
        np.fill_diagonal(self.weights, 0)
        
    def predict(self, input_pattern, max_iterations=100):
        output_pattern = np.copy(input_pattern)
        
        for _ in range(max_iterations):
            for i in range(self.num_neurons):
                net_input = np.dot(self.weights[i, :], output_pattern)
                output_pattern[i] = 1 if net_input > 0 else -1
        
        return output_pattern

# Example usage
if __name__ == "__main__":
    # Define 4 binary vectors (patterns)
    patterns = [
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [-1, -1, 1, 1]
    ]
    
    # Initialize Hopfield Network
    num_neurons = len(patterns[0])
    hopfield_net = HopfieldNetwork(num_neurons)
    
    # Train the Hopfield Network
    hopfield_net.train(patterns)
    
    # Test the Hopfield Network
    test_patterns = [
        [1, 1, -1, -1],  # Should retrieve the same pattern
        [1, -1, -1, 1],  # Should converge to one of the stored patterns
        [-1, -1, -1, -1]  # Should not converge to any stored pattern
    ]
    
    for test_pattern in test_patterns:
        output_pattern = hopfield_net.predict(test_pattern)
        print(f"Input pattern: {test_pattern}, Output pattern: {output_pattern}")

"""
Input pattern: [1, 1, -1, -1], Output pattern: [ 1  1 -1 -1]
Input pattern: [1, -1, -1, 1], Output pattern: [-1  1 -1  1]
Input pattern: [-1, -1, -1, -1], Output pattern: [ 1  1 -1 -1]


"""