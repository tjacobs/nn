# Neural network
# (c) Tom Jacobs
#
# 11: Batch
# Shows taking batches of data from the training data to train on, converges much faster, see the learning rate, for stochastic gradient descent

import matplotlib.pyplot as plt
import numpy as np

# Training settings
learning_rate = 0.001
batch_size = 3
epochs = 100000

# Training data
training_data_y = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 6]

# Batch index state
index_data = 0

# Generate points between 0 and data_len, and space to store y values to plot
data_len = len(training_data_y)
x_values = np.linspace(0, data_len - 1, data_len)
y_values = np.zeros(data_len)

def main():
    # Params
    np.random.seed(10)                  # For reproducibility
    n_neurons = 10
    p_weights = [
        np.random.randn(n_neurons, 1),  # First hidden layer weights
        np.random.randn(1, n_neurons)   # Output layer weights
    ]
    p_biases  = [
        np.random.randn(n_neurons),     # First hidden layer biases
        np.random.randn(1)              # Output layer biases
    ]

    # Plot initial
    total_loss, gradients_w, gradients_b = forward_and_loss(p_weights, p_biases, get_data, data_len)
    plot(x_values, y_values, training_data_y)

    # Train
    for epoch in range(epochs + 1):
        # Get values, loss, and gradients
        total_loss, gradients_w, gradients_b = forward_and_loss(p_weights, p_biases, get_data, batch_size)

        # Print
        if epoch % 100 == 0: print(f"Epoch {epoch}, loss: {total_loss}")

        # Update weights and biases using gradients
        for layer in range(len(p_weights)):
            p_weights[layer] -= learning_rate * gradients_w[layer]
            p_biases[layer] -= learning_rate * gradients_b[layer]

    # Plot results
    plot(x_values, y_values, training_data_y)

# Calculate forward values and loss and gradients
def forward_and_loss(p_weights, p_biases, get_data_function, batch_size_requested):
    global y_values

    # Init items to 0
    total_loss = 0
    gradients_w = [np.zeros_like(w, dtype=np.float64) for w in p_weights]
    gradients_b = [np.zeros_like(b, dtype=np.float64) for b in p_biases]

    # Get batch of data
    data_x, data_y = get_data_function(batch_size_requested)

    # For each x value in this batch
    for i in range(len(data_x)):
        # Get x value
        x = data_x[i]

        # Run network forward
        y, layer_outputs = forward(x, p_weights, p_biases)
        y_values[int(x)] = y

        # Calculate loss as difference squared (so it is always positive)
        loss = (y - data_y[i]) ** 2
        total_loss += loss

        # Backward pass: calculate gradients
        # Compute the gradient of the loss with respect to the network output y
        gradient_loss_y = 2 * (y - data_y[i]) # dL/dy, the 2 * is from the ^2 derived
        for layer in reversed(range(len(p_weights))):
        	# Calculate derivative of the ReLU function, 1 for positive values, 0 otherwise, times gradient of loss
            grad_h = gradient_loss_y * (layer_outputs[layer] > 0).astype(np.float64)  

            # Accumulate the bias gradient
            gradients_b[layer] += grad_h

            # Compute the gradient of the loss with respect to the weights of the first layer, accumulate these gradients over all x data points
            if layer == 0: gradients_w[layer] += grad_h[:, np.newaxis] * x # Reshape grad_h to a column vector

            # Compute the outer product of grad_h and the outputs of the previous layer
			# The layer_outputs[layer-1] is the output from the previous layer (i.e., input to the current layer)
			# and reshape(-1) flattens the previous layer's output to ensure proper shape for the outer product
            else:          gradients_w[layer] += np.outer(grad_h, layer_outputs[layer-1].reshape(-1))

            # Compute the gradient of the loss with respect to the inputs of the current layer (to propagate it backward to the previous layer)
            gradient_loss_y = np.dot(p_weights[layer].T, grad_h)

    # Return
    return total_loss, gradients_w, gradients_b

# Network
def forward(x, p_weights, p_biases):
    # Start with input
    layer_output = np.array([x], dtype=np.float64)

    # For each layer, calculate an array of that layer's node outputs
    layer_outputs = []
    for layer in range(len(p_weights)):
        layer_output = activation(np.dot(p_weights[layer], layer_output) + p_biases[layer])
        layer_outputs.append(layer_output)

    # Get output
    y = layer_output[0]

    # Output value and layer outputs
    return y, layer_outputs

# Activation function
def activation(x):
    # ReLU
    return np.maximum(0, x)

def get_data(length):
    global index_data
    xs, ys = x_values[index_data:index_data + length], training_data_y[index_data:index_data + length]
    index_data += length
    if index_data >= len(training_data_y): index_data = 0
    return xs, ys

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
