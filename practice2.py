import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('/content/Practice2_Chapter2.csv')


data = {
    'TV': df['TV'].tolist(),
    'Radio': df['Radio'].tolist(),
    'Newspaper': df['Newspaper'].tolist(),
    'Sales': df['Sales'].tolist()
}

X = np.array([data['TV'], data['Radio'], data['Newspaper']]).T  
y = np.array(data['Sales'])

means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
X_normalized = (X - means) / stds  

m, n = X_normalized.shape
weights = np.zeros(n)
bias = 0
learning_rate = 0.001
iterations = 20000
tolerance = 1e-9  

def compute_cost(X, y, weights, bias):
    m = len(y)
    predictions = X.dot(weights) + bias
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, weights, bias, learning_rate, iterations, tolerance):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(weights) + bias
        error = predictions - y

        weights_gradient = (1 / m) * X.T.dot(error)
        bias_gradient = (1 / m) * np.sum(error)

        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient

        cost = compute_cost(X, y, weights, bias)
        cost_history.append(cost)

        # Convergence check
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tolerance:
            print(f"Converged at iteration {i}")
            break

    return weights, bias, cost_history

weights_optimal, bias_optimal, cost_history = gradient_descent(X_normalized, y, weights, bias, learning_rate, iterations, tolerance)


#print("Optimal weights:", weights_optimal)
#print("Optimal bias:", bias_optimal)

new_data = np.array([151.5,41.3,58.5])
new_data_normalized = (new_data - means) / stds  
predicted_sales = new_data_normalized.dot(weights_optimal) + bias_optimal
print("Predicted sales:", predicted_sales)

plt.plot(cost_history)
plt.title("Cost Function over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()