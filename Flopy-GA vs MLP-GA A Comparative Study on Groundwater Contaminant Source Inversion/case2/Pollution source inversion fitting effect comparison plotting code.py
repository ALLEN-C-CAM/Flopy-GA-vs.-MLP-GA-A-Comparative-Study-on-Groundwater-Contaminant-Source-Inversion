import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
stress_periods = np.arange(1, 9)  # 应力期1-8
s1_inverted = [87, 88, 67, 64, 85, 34, 38, 20]  # S1反演浓度
s1_real = [35, 35, 90, 90, 63, 63, 47, 47]      # S1真实浓度
s2_inverted = [51, 32, 54, 23, 36, 32, 51, 56]  # S2反演浓度
s2_real = [24, 24, 56, 56, 43, 43, 35, 35]      # S2真实浓度

# 井的位置信息
s1_row_inv, s1_col_inv = 4, 7  # S1反演行号列号
s1_row_real, s1_col_real = 4, 7  # S1真实行号列号
s2_row_inv, s2_col_inv = 4, 4  # S2反演行号列号
s2_row_real, s2_col_real = 4, 4  # S2真实行号列号

# 计算拟合指标
def calculate_metrics(real, inverted):
    r2 = r2_score(real, inverted)
    rmse = np.sqrt(mean_squared_error(real, inverted))
    mse = mean_squared_error(real, inverted)
    return r2, rmse, mse

s1_r2, s1_rmse, s1_mse = calculate_metrics(s1_real, s1_inverted)
s2_r2, s2_rmse, s2_mse = calculate_metrics(s2_real, s2_inverted)

# 创建图形
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[4, 1], hspace=0.4, wspace=0.3)

# S1井的图形 (左图)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(stress_periods, s1_real, 'bo-', label='真实值', markersize=8)
ax1.plot(stress_periods, s1_inverted, 'rs--', label='反演值', markersize=8)
ax1.set_xlabel('应力期', fontsize=12)
ax1.set_ylabel('释放浓度', fontsize=12)
ax1.set_title('S1井释放浓度拟合效果', fontsize=14)
ax1.legend(loc='upper left', bbox_to_anchor=(0.15, 0.95))
ax1.grid(True, linestyle='--', alpha=0.6)

# S1井位置小图 (调整到左下角)
ax1_inset = ax1.inset_axes([0.2, 0.15, 0.25, 0.25])
ax1_inset.scatter([s1_row_real], [s1_col_real], c='b', s=100, label='真实')
ax1_inset.scatter([s1_row_inv], [s1_col_inv], c='r', marker='s', s=100, label='反演')
ax1_inset.set_xlabel('行号', fontsize=8)
ax1_inset.set_ylabel('列号', fontsize=8)
ax1_inset.set_title('井位置', fontsize=10)
ax1_inset.legend(fontsize=8, loc='upper right')
ax1_inset.grid(True, linestyle=':', alpha=0.4)

# S2井的图形 (右图)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(stress_periods, s2_real, 'bo-', label='真实值', markersize=8)
ax2.plot(stress_periods, s2_inverted, 'rs--', label='反演值', markersize=8)
ax2.set_xlabel('应力期', fontsize=12)
ax2.set_ylabel('释放浓度', fontsize=12)
ax2.set_title('S2井释放浓度拟合效果', fontsize=14)
ax2.legend(loc='upper left', bbox_to_anchor=(0.15, 0.95))
ax2.grid(True, linestyle='--', alpha=0.6)

# S2井位置小图 (调整到左下角)
ax2_inset = ax2.inset_axes([0.2, 0.15, 0.25, 0.25])
ax2_inset.scatter([s2_row_real], [s2_col_real], c='b', s=100, label='真实')
ax2_inset.scatter([s2_row_inv], [s2_col_inv], c='r', marker='s', s=100, label='反演')
ax2_inset.set_xlabel('行号', fontsize=8)
ax2_inset.set_ylabel('列号', fontsize=8)
ax2_inset.set_title('井位置', fontsize=10)
ax2_inset.legend(fontsize=8, loc='upper right')
ax2_inset.grid(True, linestyle=':', alpha=0.4)

# 添加表格 (跨两列)
table_ax = fig.add_subplot(gs[1, :2])
table_ax.axis('off')

# 表格数据
metrics_data = [
    ['井号', 'R²', 'RMSE', 'MSE'],
    ['S1', f'{s1_r2:.4f}', f'{s1_rmse:.4f}', f'{s1_mse:.4f}'],
    ['S2', f'{s2_r2:.4f}', f'{s2_rmse:.4f}', f'{s2_mse:.4f}']
]

# 创建表格
table = table_ax.table(cellText=metrics_data,
                      loc='center',
                      cellLoc='center',
                      colWidths=[0.2, 0.3, 0.3, 0.3])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)  # 调整表格高度

# 设置标题行样式
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#dddddd')

# 添加右侧空白区域用于可能的注释
note_ax = fig.add_subplot(gs[:, 2])
note_ax.axis('off')
note_ax.text(0.1, 0.5, "拟合指标说明:\nR²: 决定系数\nRMSE: 均方根误差\nMSE: 均方误差",
             fontsize=12, va='center')

plt.tight_layout()
plt.savefig('污染源反演拟合效果对比.png', dpi=300, bbox_inches='tight')
plt.show()