import matplotlib.pyplot as plt
import numpy as np

def main():
	print("Neural network")

	# Training data
	training_data_y = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 6, 5, 4, 2, 1, 4, 6, 4, 3, 3, 2]
	data_len = len(training_data_y)

	# Params
	p_weights = [[2, -3, 4], [1, 1, 1]]
	p_biases  = [[2, -3, 4], [0, 0, 0]]

	# Generate points between 0 and 30
	x_values = np.linspace(0, data_len-1, data_len)
	print((x_values))
	print(len(x_values))

	# Calculate corresponding y values and plot
	y_values = [calculate_y(x, p_weights, p_biases)[0] for x in x_values]
	plot(x_values, y_values, training_data_y)

	# Update params
	p_weights[0][0] = 4
	p_weights[0][1] = -1
	p_weights[0][2] = 40
	p_weights[1][0] = 4
	p_weights[1][1] = -1
	p_weights[1][2] = 0.4

	# Calculate corresponding y values and plot
	y_values = [calculate_y(x, p_weights, p_biases)[0] for x in x_values]
	plot(x_values, y_values, training_data_y)

# Network
def calculate_y(x, p_weights, p_biases):
	# Start with inputs
	hs = x

	# For each layer, calculate an array of that layer's node outputs
	for layer in range(len(p_weights)):
		hs = a(np.dot(p_weights[layer], hs) + p_biases[layer])

	# Output layer
	return hs

# Activation function
def a(x):
	# ReLU
	return np.maximum(0, x)

# Plot
def plot(x_values, y_values, y_real):
	plt.figure(figsize=(8, 6))
	plt.plot(x_values, y_values, label='Neural Network Output')
	plt.plot(x_values, y_real, label='Actual')
	plt.title("Output")
	plt.xlabel('Input (x)')
	plt.ylabel('Output (y)')
	plt.grid(True)
	plt.legend()
	plt.show()

# Run main	
if __name__ == "__main__":
	main()
