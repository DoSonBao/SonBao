import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo dữ liệu
X1 = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364, 0.398, 0.4,
               0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.561, 0.569, 0.594, 0.638, 0.656,
               0.816, 0.853, 0.938, 1.036, 1.045])
X2 = X1 ** 2
Y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Khởi tạo ngẫu nhiên các tham số
theta0 = np.random.rand()
theta1 = np.random.rand()
theta2 = np.random.rand()

alpha = 1e-4  # Tốc độ học
iterations = 10000  # Số vòng lặp
lambda_reg = 1  # Hệ số điều chỉnh

# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm dự đoán
def predict(theta0, theta1, theta2, X1, X2):
    return sigmoid(theta0 + theta1 * X1 + theta2 * X2)

# Hàm tính chi phí với regularization
def compute_cost(Y, theta0, theta1, theta2):
    m = len(Y)
    Y_pred = predict(theta0, theta1, theta2, X1, X2)
    epsilon = 1e-15  # Để tránh log(0)
    
    # Tính toán chi phí
    cost = -(1/m) * np.sum(Y * np.log(Y_pred + epsilon) + (1 - Y) * np.log(1 - Y_pred + epsilon))
    regularization_term = (lambda_reg / (2 * m)) * (theta0**2 + theta1**2 + theta2**2)
    
    return cost + regularization_term

# Hàm Gradient Descent để tối ưu hóa tham số
def optimize_theta(Y, theta0, theta1, theta2, alpha, iterations):
    m = len(Y)
    
    for _ in range(iterations):
        Y_pred = predict(theta0, theta1, theta2, X1, X2)
        
        grad0 = (1/m) * np.sum(Y_pred - Y) + (lambda_reg / m) * theta0
        grad1 = (1/m) * np.sum((Y_pred - Y) * X1) + (lambda_reg / m) * theta1
        grad2 = (1/m) * np.sum((Y_pred - Y) * X2) + (lambda_reg / m) * theta2
        
        theta0 -= alpha * grad0
        theta1 -= alpha * grad1
        theta2 -= alpha * grad2
    
    return theta0, theta1, theta2

# Huấn luyện mô hình
theta0, theta1, theta2 = optimize_theta(Y, theta0, theta1, theta2, alpha, iterations)
print(f"theta0 = {theta0:.6f}, theta1 = {theta1:.6f}, theta2 = {theta2:.6f}")

# Vẽ biểu đồ dữ liệu và đường hồi quy
plt.scatter(X1, Y, color='red', label='Data Points')
X_values = np.linspace(min(X1), max(X1), 100)
X2_values = X_values ** 2
Y_pred = predict(theta0, theta1, theta2, X_values, X2_values)

plt.plot(X_values, Y_pred, color='blue', label='Regression Line')
plt.xlabel('Grain Size (X1)')
plt.ylabel('Probability')
plt.title('Logistic Regression Model')
plt.legend()
plt.show()