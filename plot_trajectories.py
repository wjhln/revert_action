import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('trajectory_results.csv')

# 创建图形
plt.figure(figsize=(10, 8))

# 绘制原始轨迹
plt.plot(df['orig_x'], df['orig_y'], 'b-o', label='original', linewidth=2, markersize=4)

# 绘制优化轨迹
plt.plot(df['opt_x'], df['opt_y'], 'r-s', label='optimized', linewidth=2, markersize=4)

# 设置图形属性
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# 设置坐标轴相等比例
plt.axis('equal')

# 显示图形
plt.tight_layout()
plt.show() 