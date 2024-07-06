# Neural network


def main():
	print("Neural network")

	# Params
	p0 = 0
	p1 = 1
	p2 = 1
	p3 = 1
	p10 = 2
	p11 = 2
	p20 = -3
	p21 = -3
	p30 = 4
	p31 = 4

	# Input
	x = 5;

	# Network
	h1 = a(p10 * x + p11)
	h2 = a(p20 * x + p21)
	h3 = a(p30 * x + p31)
	y = p0 + (p1 * h1) + (p2 * h2) + (p3 * h3)

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
