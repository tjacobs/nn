


def main():
	print("Neural network")

	# Params
	p = [0, 1, 1, 1, 2, 2, -3, -3, 4, 4]

	# Generate 100 points between -5 and 5
	x_values = np.linspace(-5, 5, 100)

	# Calculate corresponding y values
	y_values = [calculate_y(x, p) for x in x_values]

	# Output
	print(f"Output: {y_values=}")

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
	if x < 0: x = 0
	return x

# Run main	
if __name__ == "__main__":
	main()
