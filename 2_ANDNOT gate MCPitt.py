def mcculloch_pitts_neuron(inputs, weights, threshold):
    # Compute the weighted sum
    weighted_sum = sum([x * w for x, w in zip(inputs, weights)])

    # Apply the threshold logic
    output = 1 if weighted_sum >= threshold else 0

    return output

# Take inputs from the user
input_1 = [0,1,0,1]
input_2 = [0,0,1,1]

# Define weights and threshold for ANDNOT function
weights = [1, -1]  
threshold = 1  

for i in range(0,4):
# Compute output of ANDNOT function
    output = mcculloch_pitts_neuron([input_1[i], input_2[i]], weights, threshold)

    # Print the result
    print(f"Input 1: {input_1[i]}, Input 2: {input_2[i]}, Output: {output}")

"""
OUTPUT:
Input 1: 0, Input 2: 0, Output: 0
Input 1: 1, Input 2: 0, Output: 1
Input 1: 0, Input 2: 1, Output: 0
Input 1: 1, Input 2: 1, Output: 0

"""