import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
file_path = '/Users/panpan/Desktop/ML/result_100.csv'
data = pd.read_csv(file_path)

# 自变量和因变量
X = data.iloc[:, :-1]  # 前4列为自变量
y = data.iloc[:, -1]   # 最后一列为因变量

# 标准化自变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 创建神经网络模型（多层感知机回归）
nn_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)

# 6. 训练模型
nn_model.fit(X_train_scaled, y_train)

# 7. 测试集预测
y_pred_nn = nn_model.predict(X_test_scaled)

# 8. 评估模型性能
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

# 9. 输出评估结果
print(f"神经网络模型 - 均方误差 (MSE): {mse_nn}")
print(f"神经网络模型 - 决定系数 (R²): {r2_nn}")

# 10. 预测新数据（可选）
new_data = [[0.84, 0.7, 0.5, 0.7]]  # 示例输入
new_data_scaled = scaler.transform(new_data)  # 标准化新数据
predicted_value = nn_model.predict(new_data_scaled)
print(f"新数据的预测值: {predicted_value[0]}")