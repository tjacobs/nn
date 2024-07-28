import matplotlib.pyplot as plt
import numpy as np

def main():
    # Training data
    training_data_y = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 6, 5, 4, 2, 1, 4, 6, 4, 3, 3, 2]
    data_len = len(training_data_y)

    # Params
    p_weights = [np.array([[2], [-3], [4]], dtype=np.float64), np.array([[1, 1, 1]], dtype=np.float64)]
    p_biases  = [np.array([2, -3, 4], dtype=np.float64), np.array([0], dtype=np.float64)]

    # Generate points between 0 and data_len
    x_values = np.linspace(0, data_len - 1, data_len)

    # Training settings
    learning_rate = 0.0000001
    epochs = 1000

    # Train
    for epoch in range(epochs + 1):

        if epoch <= 2: print(p_weights)

        # Get values, loss, and gradients
        y_values, total_loss, gradients_w, gradients_b = forward_and_loss(x_values, p_weights, p_biases, training_data_y)

        # Update weights and biases using gradients
        for layer in range(len(p_weights)):
            p_weights[layer] -= learning_rate * gradients_w[layer]
            p_biases[layer] -= learning_rate * gradients_b[layer]


        # Print
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {total_loss}")
            print(p_weights)

    # Plot results
    plot(x_values, y_values, training_data_y)

# Calculate forward values and loss and gradients
def forward_and_loss(x_values, p_weights, p_biases, training_data_y):
    # Init items to 0
    total_loss = 0
    y_values = np.zeros(len(x_values))
    gradients_w = [np.zeros_like(w, dtype=np.float64) for w in p_weights]
    gradients_b = [np.zeros_like(b, dtype=np.float64) for b in p_biases]

    # For each x value
    for i in range(len(x_values)):
        # Get x value
        x = x_values[i]

        # Run network forward
        y, layer_outputs = forward(x, p_weights, p_biases)
        y_values[i] = y

        # Calculate loss as difference squared (so it is always positive)
        loss = (y - training_data_y[i]) ** 2
        total_loss += loss

        # Backward pass: calculate gradients
        # Compute the gradient of the loss with respect to the network output y
        gradient_loss_y = 2 * (y - training_data_y[i]) # dL/dy, the 2 * is from the ^2 derived
        for layer in reversed(range(len(p_weights))):
        	# Calculate derivative of the ReLU function, 1 for positive values
            grad_h = gradient_loss_y * (layer_outputs[layer] > 0).astype(np.float64)  

            # Accumulate the bias gradient
            gradients_b[layer] += grad_h

            # Computes the gradient of the loss with respect to the weights of the first layer.
            # Accumulate these gradients over all x data points
            if layer == 0:       gradients_w[layer] += grad_h[:, np.newaxis] * x # Reshape grad_h to a column vector

            # Computes the outer product of grad_h and the outputs of the previous layer.
			# layer_outputs[layer-1] is the output from the previous layer (i.e., input to the current layer).
			# reshape(-1) flattens the previous layer's output to ensure proper shape for the outer product.
            else:                gradients_w[layer] += np.outer(grad_h, layer_outputs[layer-1].reshape(-1))

            # Compute the gradient of the loss with respect to the inputs of the current layer (to propagate it backward to the previous layer).
            gradient_loss_y = np.dot(p_weights[layer].T, grad_h)

    # Return
    return y_values, total_loss, gradients_w, gradients_b

# Network
def forward(x, p_weights, p_biases):
    # Start with inputs
    hs = np.array([x], dtype=np.float64)
    layer_outputs = []

    # For each layer, calculate an array of that layer's node outputs
    for layer in range(len(p_weights)):
        hs = a(np.dot(p_weights[layer], hs) + p_biases[layer])
        layer_outputs.append(hs)

    # Get output
    y = sum(hs)

    # Output value and layer outputs
    return y, layer_outputs

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
