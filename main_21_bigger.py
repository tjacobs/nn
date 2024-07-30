# Neural network
# (c) Tom Jacobs
#
# 21: Bigger
# Change to PyTorch implementation with bigger training data

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

# Training settings
learning_rate = 0.001
epochs = 10000

# Training data
data_len = 6
np.random.seed(0)
#training_data_y = torch.tensor([1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 2], dtype=torch.float32)
training_data_y = torch.tensor(((np.random.randn(data_len, 1) * 2) + 1), dtype=torch.float32)
batch_size = data_len // 4

# Generate points between 0 and data_len, and space to store y values to plot
data_len = len(training_data_y)
x_values = torch.linspace(0, data_len - 1, data_len)
y_values = torch.zeros(data_len)

# Network size
n_nodes = 12

# Define neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, n_nodes)
        self.hidden2 = nn.Linear(n_nodes, n_nodes)
        self.output = nn.Linear(n_nodes, 1)
        self.relu = nn.ReLU()

    # Define forward pass of model
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def main():
    # Initialize network
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create graph
    fig, ax = plt.subplots(figsize=(8, 6))
    line1, = ax.plot(x_values, y_values, label='Neural Network Output')
    line2, = ax.plot(x_values, training_data_y, label='Actual')
    ax.set_title("Output")
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('Output (y)')
    ax.grid(True)
    ax.legend()

    # Train
    for epoch in range(epochs + 1):
        # Init
        total_loss = 0
        
        # For each batch
        for index in range(0, data_len, batch_size):
            # Get data
            xs, ys = get_data(batch_size, index)
            
            # Forward pass
            outputs = model(xs)
            loss = criterion(outputs, ys)
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad() # Clear gradients, because they are accumulated
            loss.backward()       # Calculate gradients
            optimizer.step()      # Update the model paremeters
        
        # Update graph
        with torch.no_grad():
            y_values[:] = model(x_values.unsqueeze(1)).squeeze()
            line1.set_ydata(y_values)
            fig.canvas.draw()
            plt.pause(0.001)
        
        # Print every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {total_loss}")

# Define data loading function
def get_data(batch_size, index):
    #index = int(random.random() * 4) * 3
    xs = x_values[index:index + batch_size].unsqueeze(1) # Add a dimension
    ys = training_data_y[index:index + batch_size].unsqueeze(1)
    return xs, ys

# Run main    
if __name__ == "__main__":
    main()
