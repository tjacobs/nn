# Neural network
# (c) Tom Jacobs
#
# 08: Loss
# Calculates the loss, i.e. how different the model predicts to the training data

import matplotlib.pyplot as plt
import numpy as np

def main():
    # Training data
    training_data_y = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 6, 5, 4, 2, 1, 4, 6, 4, 3, 3, 2]
    data_len = len(training_data_y)

    # Params
    p_weights = [[2, -3, 4], [1, 1, 1]]
    p_biases  = [[2, -3, 4], [0, 0, 0]]

    # Generate points between 0 and 30
    x_values = np.linspace(0, data_len - 1, data_len)

    # Forward pass: calculate y values and total loss
    y_values, total_loss = print_forward_and_loss(x_values, p_weights, p_biases, training_data_y)

    # Update params
    p_weights[0][0] = 2
    p_weights[0][1] = -1
    p_weights[0][2] = -3

    # Forward pass: calculate y values and total loss
    print("After updating p_weights:")
    y_values, total_loss = print_forward_and_loss(x_values, p_weights, p_biases, training_data_y)

def print_forward_and_loss(x_values, p_weights, p_biases, training_data_y):
    total_loss = 0
    y_values = np.zeros(len(x_values))
    for i in range(len(x_values)):
        x = x_values[i]

        # Forward
        y = forward(x, p_weights, p_biases)
        y_values[i] = y

        # Calculate loss as difference squared (so always positive)
        loss = (y - training_data_y[i]) ** 2
        total_loss += loss

        # Print
        print(f"{i}:\tX: {x}, \tY: {y}, \tData: {training_data_y[i]}, \tLoss: {loss}")

    # Print
    print(f"Total loss: {total_loss}")

    # Plot
    plot(x_values, y_values, training_data_y)

    # Return
    return y_values, total_loss


# Network
def forward(x, p_weights, p_biases):
    # Start with inputs
    hs = x

    # For each layer, calculate an array of that layer's node outputs
    for layer in range(len(p_weights)):
        hs = a(np.dot(p_weights[layer], hs) + p_biases[layer])

    # Get output
    y = hs[0]

    # Output value
    return y

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
