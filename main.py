
def main():
	print("Neural network")

	# Params
	p = [0, 1, 1, 1, 2, 2, -3, -3, 4, 4]

	# Input
	x = 5;

	# Network
	h1 = a(p[4] * x + p[5])
	h2 = a(p[6] * x + p[7])
	h3 = a(p[8] * x + p[9])
	y = p[0] + (p[1] * h1) + (p[2] * h2) + (p[3] * h3)

	# Output
	print(f"Output: {y=}")

# Activation function
def a(x):
	# ReLU
	if x < 0: x = 0
	return x

# Run main	
if __name__ == "__main__":
	main()
