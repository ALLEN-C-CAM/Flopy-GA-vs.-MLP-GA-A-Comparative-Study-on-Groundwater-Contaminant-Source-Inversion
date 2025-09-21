import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import pandas as pd

# 更健壮的专业绘图设置
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # 通用字体
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.25,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'figure.figsize': (14, 6),
    'figure.dpi': 110
})

# 读取data文件夹中的预测结果.xlsx文件
data = pd.read_excel('data/预测结果.xlsx')
#列名为真实值的是y_true，列名为预测值的是y_pred
y_true = data['真实值']
y_pred = data['预测值']
relative_errors = (y_pred - y_true) / y_true   *10000# 相对误差百分比

# 计算统计指标
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mre = np.mean(np.abs(relative_errors))  # 平均绝对相对误差
print(f"mre = {mre:.3f}")

# 创建图形
plt.figure(figsize=(8, 7), dpi=100)

# 绘制散点图（颜色表示相对误差大小）
scatter = plt.scatter(y_true, y_pred,
                     c=relative_errors,
                     cmap='coolwarm',
                     vmin=-40, vmax=40,  # 对称的颜色范围
                     alpha=0.8,
                     s=60,
                     edgecolor='w',
                     linewidth=0.7)

# 添加1:1参考线
max_val = max(y_true.max(), y_pred.max()) * 1.05
min_val = min(y_true.min(), y_pred.min()) * 0.95
plt.plot([min_val, max_val], [min_val, max_val],
         'k--', lw=2, alpha=0.8, label='1:1 Line')

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Relative Error (%)', fontsize=11)

# 添加统计指标文本框
stats_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMRE = {mre/100:.5f}%"
plt.text(0.05, 0.95, stats_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# 设置坐标轴
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('Actual Values', fontsize=12, labelpad=8)
plt.ylabel('Predicted Values', fontsize=12, labelpad=8)
plt.title('Actual vs Predicted Values with Relative Error',
          fontsize=14, pad=15, weight='bold')

# 添加图例
plt.legend(loc='lower right', facecolor='white')

# 确保坐标轴比例一致
plt.gca().set_aspect('equal')

# 调整布局并显示
plt.tight_layout()
plt.show()

import seaborn as sns
from scipy import stats

# 计算额外统计指标
mdre = np.median(np.abs(relative_errors))  # 中位数绝对相对误差
mean_error = np.mean(relative_errors)  # 平均相对误差（带符号）
std_error = np.std(relative_errors)  # 相对误差标准差

# 创建图形
plt.figure(figsize=(8, 5), dpi=100)

# 绘制相对误差直方图和核密度估计
ax = sns.histplot(relative_errors, kde=True,
                 color='steelblue', alpha=0.7,
                 bins=20, stat='density',
                 edgecolor='w', linewidth=1)

# 添加正态分布拟合曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = stats.norm.pdf(x, mean_error, std_error)
plt.plot(x, p, 'k--', linewidth=1.8, alpha=0.8, label='Normal Distribution')

# 添加关键统计线
plt.axvline(mean_error, color='darkred', linestyle='-', lw=2,
            label=f'Mean: {mean_error:.1f}%')
plt.axvline(mre, color='darkred', linestyle='--', lw=2,
            label=f'MAE: {mre:.1f}%')
plt.axvline(mdre, color='darkgreen', linestyle='-.', lw=2,
            label=f'Median: {mdre:.1f}%')

# 添加零误差参考线
plt.axvline(0, color='gray', linestyle='-', lw=1.5, alpha=0.7)

# 添加统计指标文本框
stats_text = (f"Mean = {mean_error:.1f}%\n"
             f"Std Dev = {std_error:.1f}%\n"
             f"MAE = {mre:.1f}%\n"
             f"Median AE = {mdre:.1f}%")
plt.text(0.95, 0.95, stats_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# 设置坐标轴和标题
plt.xlabel('Relative Error (%)', fontsize=12, labelpad=8)
plt.ylabel('Density', fontsize=12, labelpad=8)
plt.title('Relative Error Distribution', fontsize=14, pad=15, weight='bold')

# 添加图例
plt.legend(loc='upper left', facecolor='white')

# 调整布局并显示
plt.tight_layout()
plt.show()