import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    # Detach tensors and move to CPU for plotting
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # Get the learned parameters from the model
    # model.parameters is a torch.Tensor directly, not an nn.Module method
    a, b, c = model.parameters.detach().cpu().numpy()
    c = c

    print(f"Learned parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")

    # Plot the data points
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm', edgecolor='k')
    plt.title('Classification Data with Linear Classification Boundary')

    # Define the x-range for plotting the decision boundary
    x_min, x_max = X_np[:, 0].min() - 0.1, X_np[:, 0].max() + 0.1
    x_values = np.linspace(x_min, x_max, 100)

    # Calculate y-values for the decision boundary line: ax + by + c = 0
    # Handle cases where 'b' might be zero or very close to zero to avoid division by zero
    if abs(b) < 1e-6:  # If b is approximately zero, the line is vertical: x = -c/a
        if abs(a) > 1e-6:
            x_vertical = -c / a
            plt.axvline(x=x_vertical, color='red', linestyle='--', label='Decision Boundary')
        else:
            # Both a and b are zero, which means no meaningful boundary line can be plotted
            print("Warning: Both 'a' and 'b' parameters are near zero. Cannot plot decision line.")
    else:  # Standard case: y = -(a*x + c) / b
        y_values = -(a * x_values + c) / b
        plt.plot(x_values, y_values, color='red', linestyle='--', label='Decision Boundary')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Call the plotting function with the trained model and data
#plot_decision_boundary(model, X, y)
