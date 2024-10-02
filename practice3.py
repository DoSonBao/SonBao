import numpy as np

# Logistic Function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Prediction Function
def predict(X, theta0, theta1):
    z = theta0 + theta1 * X
    gz = logistic_function(z)
    return gz

# Cost Function (Binary Cross-Entropy)
def cost_function(X, y_true, theta0, theta1):
    m = len(X)
    epsilon = 1e-15  # Để tránh log(0)
    
    y_pred = predict(X, theta0, theta1)
    
    # Cross-entropy loss
    cost = (-1 / m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return cost

# Gradient Descent
def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    m = len(X)
    
    for _ in range(iterations):
        # Tính đạo hàm của hàm mất mát (gradient)
        gradient0 = (1 / m) * np.sum(predict(X, theta0, theta1) - y)
        gradient1 = (1 / m) * np.sum((predict(X, theta0, theta1) - y) * X)
        
        # Cập nhật trọng số (theta0, theta1)
        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1
    
    return theta0, theta1

# Dữ liệu ví dụ
X = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.336, 0.356, 0.364, 0.394, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.565, 0.569, 0.589, 0.638, 0.709])
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 0: Absent, 1: Present

# Khởi tạo trọng số ban đầu
theta0 = 0
theta1 = 0

# Thông số huấn luyện
learning_rate = 0.01
iterations = 10000

# Huấn luyện mô hình logistic regression
theta0, theta1 = gradient_descent(X, y, theta0, theta1, learning_rate, iterations)

# In ra các trọng số sau khi huấn luyện
print(f'Theta 0: {theta0}')
print(f'Theta 1: {theta1}')

# Dự đoán sự hiện diện của nhện trên các giá trị mới
new_grain_size = np.array([0.250, 0.500, 0.600])  # Ví dụ kích thước hạt mới
predictions = predict(new_grain_size, theta0, theta1)

print(f'Dự đoán sự hiện diện của nhện: {predictions}')