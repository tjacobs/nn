import matplotlib.pyplot as plt
import numpy as np

def main():
	print("Neural network")

	# Params
	p_output  = [0, 1, 1, 1]
	p_weights = [[2, -3, 4], [1, 1, 1]]
	p_biases  = [[2, -3, 4], [0, 0, 0]]

	# Generate 100 points between -5 and 5
	x_values = np.linspace(-5, 5, 100)

	# Calculate corresponding y values
	y_values = [calculate_y(x, p_weights, p_biases, p_output) for x in x_values]

	# Output
	print(f"Output: {y_values=}")

	# Plot the line
	plot(x_values, y_values)

# Network
def calculate_y(x, p_weights, p_biases, p_output):
	# Start with inputs
	hs = x

	# For each layer, calculate an array of that layer's node outputs, just one layer to start
	for layer in range(len(p_weights)):
		hs = a(np.dot(p_weights[layer], hs) + p_biases[layer])

	# Output layer
	return hs

# Activation function
def a(x):
	# ReLU
	return np.maximum(0, x)

# Plot
def plot(x_values, y_values):
	plt.figure(figsize=(8, 6))
	plt.plot(x_values, y_values, label='Neural Network Output')
	plt.title('Neural Network Output')
	plt.xlabel('Input (x)')
	plt.ylabel('Output (y)')
	plt.grid(True)
	plt.legend()
	plt.show()

# Run main	
if __name__ == "__main__":
	main()
