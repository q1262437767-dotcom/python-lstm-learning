"""
LSTM 时序预测入门示例
=====================
用简单的正弦波数据演示 LSTM 的基本用法

运行方式: python lstm_sine_demo.py
依赖安装: pip install numpy matplotlib tensorflow
"""

import numpy as np
import matplotlib.pyplot as plt

# 尝试导入 Keras（TensorFlow 后端）
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    print("请先安装 TensorFlow: pip install tensorflow")
    exit(1)


# ============================================================
# 1. 生成数据：正弦波
# ============================================================
print("=" * 50)
print("1. 生成正弦波数据...")
print("=" * 50)

# 生成 1000 个数据点
t = np.linspace(0, 100, 1000)
data = np.sin(t)

print(f"数据总长度: {len(data)}")
print(f"数据前5个值: {data[:5]}")


# ============================================================
# 2. 构造时序数据集
# ============================================================
print("\n" + "=" * 50)
print("2. 构造时序训练数据...")
print("=" * 50)

def create_dataset(data, time_steps):
    """
    将一维时间序列转为 LSTM 输入格式
    
    参数:
        data: 原始一维数据
        time_steps: 用多少个历史数据点来预测下一个值
    
    返回:
        X: 形状 (样本数, time_steps, 1) 的输入
        y: 形状 (样本数,) 的目标值
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 用前10个点预测第11个点
TIME_STEPS = 10
X, y = create_dataset(data, TIME_STEPS)

# LSTM 需要的三维输入: (样本数, 时间步, 特征数)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 划分训练集和测试集（80% / 20%）
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"时间步长 (time_steps): {TIME_STEPS}")
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print(f"输入形状: {X_train.shape}  (样本数, 时间步, 特征数)")
print(f"输出形状: {y_train.shape}  (样本数,)")


# ============================================================
# 3. 构建 LSTM 模型
# ============================================================
print("\n" + "=" * 50)
print("3. 构建 LSTM 模型...")
print("=" * 50)

model = Sequential([
    # LSTM 层: 50个隐藏单元
    LSTM(50, input_shape=(TIME_STEPS, 1)),
    # 输出层: 预测1个值
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.summary()


# ============================================================
# 4. 训练模型
# ============================================================
print("\n" + "=" * 50)
print("4. 开始训练...")
print("=" * 50)

history = model.fit(
    X_train, y_train,
    epochs=20,          # 训练20轮
    batch_size=32,      # 每批32个样本
    validation_split=0.1,  # 10% 作为验证集
    verbose=1           # 显示训练进度
)


# ============================================================
# 5. 预测 & 可视化
# ============================================================
print("\n" + "=" * 50)
print("5. 预测并可视化结果...")
print("=" * 50)

# 在测试集上预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = np.mean((y_test - y_pred.flatten()) ** 2)
print(f"测试集 MSE: {mse:.6f}")

# 绘图
plt.figure(figsize=(12, 5))

# 子图1: 训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('轮次 (Epoch)')
plt.ylabel('损失 (MSE)')
plt.title('模型训练过程')
plt.legend()

# 子图2: 预测结果对比
plt.subplot(1, 2, 2)
plt.plot(y_test, label='真实值', linewidth=1.5)
plt.plot(y_pred, label='预测值', linewidth=1.5, linestyle='--')
plt.xlabel('样本')
plt.ylabel('值')
plt.title(f'预测 vs 真实  (MSE={mse:.4f})')
plt.legend()

plt.tight_layout()
plt.savefig('lstm_result.png', dpi=150)
plt.show()
print(f"\n图片已保存为 lstm_result.png")


# ============================================================
# 6. 关键概念总结
# ============================================================
print("\n" + "=" * 50)
print("6. LSTM 关键概念回顾")
print("=" * 50)
print("""
LSTM (长短期记忆网络) 是一种特殊的 RNN，能记住长序列中的信息。

核心组件:
  - 遗忘门: 决定丢弃哪些旧信息
  - 输入门: 决定存入哪些新信息  
  - 输出门: 决定输出哪些信息

时序预测流程:
  1. 准备数据 → 2. 构造滑动窗口 → 3. 训练LSTM → 4. 预测未来值

下一步学习:
  - 换成真实数据（比如滑坡位移数据）试试
  - 调整 TIME_STEPS 和 LSTM 单元数，观察效果变化
  - 增加更多 LSTM 层，搭建更深的网络
""")
