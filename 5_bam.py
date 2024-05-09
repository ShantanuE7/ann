import numpy as np
import matplotlib.pyplot as plt

class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.zeros((output_size, input_size))
        self.T = np.zeros((input_size, output_size))

    def train(self, X, Y):
        for i in range(len(X)):
            x = np.expand_dims(X[i], axis=0)
            y = np.expand_dims(Y[i], axis=0)
            self.W += np.dot(y.T, x)
            self.T += np.dot(x.T, y)

    def recall(self, x, max_iter=1000):
        x = np.expand_dims(x, axis=0)
        y_prev = np.zeros((1, self.output_size))
        for _ in range(max_iter):
            y = np.dot(x, self.W.T)
            y = np.where(y >= 0, 1, -1)  # Threshold to bipolar values
            x = np.dot(y, self.T.T)
            x = np.where(x >= 0, 1, -1)  # Threshold to bipolar values
            if np.array_equal(y, y_prev):
                break
            y_prev = y
        return y

# Example usage
# Define input patterns (binary images of shapes)
square = np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1]])

circle = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]])



input_patterns = np.array([square.flatten(), circle.flatten()])

# Define output patterns (labels)
output_patterns = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])  # 1 for square, 1 for circle, 1 for triangle

bam = BAM(input_size=input_patterns.shape[1], output_size=output_patterns.shape[1])
bam.train(input_patterns, output_patterns)


# Test recall with a new shape (e.g., diamond)
test = np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1]])

test_input = test.flatten()
retrieved_output = bam.recall(test_input)
print("Test Input :")
print(test)
print("Retrieved Output (Label):", retrieved_output.ravel())