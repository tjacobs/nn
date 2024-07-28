# Neural network
# (c) Tom Jacobs
#
# 03: Plot
# Adds a graph plot of the values

import matplotlib.pyplot as plt
import numpy as np

def main():
	# Params
	p = [0, 1, 1, 1, 2, 2, -3, -3, 4, 4]

	# Generate 100 points between -5 and 5
	x_values = np.linspace(-5, 5, 100)

	# Calculate corresponding y values
	y_values = [calculate_y(x, p) for x in x_values]

	# Output
	print(f"Output: {y_values=}")

	# Plot the line
	plot(x_values, y_values)

# Network
def calculate_y(x, p):
	h1 = a(p[4] * x + p[5])
	h2 = a(p[6] * x + p[7])
	h3 = a(p[8] * x + p[9])
	y = p[0] + (p[1] * h1) + (p[2] * h2) + (p[3] * h3)
	return y

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
