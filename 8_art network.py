import numpy as np

class ARTNeuron:
    def __init__(self, dimension, vigilance=0.8):
        self.weights = np.random.rand(dimension)
        self.vigilance = vigilance

    def similarity(self, input):
        return np.dot(self.weights, input) / (np.linalg.norm(self.weights) * np.linalg.norm(input))

    def update_weights(self, input, learning_rate):
        self.weights = self.weights + learning_rate * (input - self.weights)

class ARTNetwork:
    def __init__(self, vigilance=0.8, learning_rate=0.1):
        self.neurons = []
        self.vigilance = vigilance
        self.learning_rate = learning_rate

    def learn(self, input):
        # Find winning neuron with highest similarity
        max_similarity = -1
        winning_neuron = None
        for neuron in self.neurons:
            similarity = neuron.similarity(input)
            if similarity > max_similarity:
                max_similarity = similarity
                winning_neuron = neuron

        # Check if similarity is high enough (vigilance criteria)
        if winning_neuron is not None and max_similarity >= self.vigilance:
            # Print details about the winning neuron
            print(f"Winning neuron index: {self.neurons.index(winning_neuron)}")
            print(f"Similarity with winning neuron: {max_similarity:.4f}")
            # Update weights of winning neuron
            winning_neuron.update_weights(input, self.learning_rate)
        else: 
            # Create a new neuron with the input as its weight
            new_neuron = ARTNeuron(len(input), vigilance=self.vigilance)
            self.neurons.append(new_neuron)
            print(f"Created new neuron with index: {self.neurons.index(new_neuron)}")

# Example usage
network = ARTNetwork(vigilance=0.6, learning_rate=0.1)
data = [np.random.rand(10) for _ in range(5)]  # Sample data points

for input_data in data:
    print(f"\nLearning with input: {input_data}")
    network.learn(input_data)
