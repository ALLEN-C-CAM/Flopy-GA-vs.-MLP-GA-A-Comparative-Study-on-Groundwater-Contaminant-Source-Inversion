import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from torch.serialization import add_safe_globals
add_safe_globals([
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.Float64DType
])
from nn import NN_MonitorModel  # 从 nn.py 导入模型类

def predict_monitor_points(model, new_input, scaler_X_mean, scaler_X_scale, scaler_Y_mean=None, scaler_Y_scale=None):
    """
    使用训练好的模型预测监测点数据

    参数:
        model: 训练好的 PyTorch 模型
        new_input: 输入数据 (numpy 数组)
        scaler_X_mean, scaler_X_scale: 训练数据的标准化参数
        scaler_Y_mean, scaler_Y_scale: 可选的输出反标准化参数

    返回:
        prediction: 预测结果 (numpy 数组)
    """
    # 确保输入是正确的维度
    if new_input.ndim == 1:
        new_input = new_input.reshape(1, -1)

    # 应用与训练时相同的标准化
    new_input_scaled = (new_input - scaler_X_mean) / scaler_X_scale

    # 转换为 PyTorch 张量并进行预测
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)
        prediction = model(input_tensor).numpy()

    # 如果提供了输出标准化参数，则进行反标准化
    if scaler_Y_mean is not None and scaler_Y_scale is not None:
        prediction = prediction * scaler_Y_scale + scaler_Y_mean

    return prediction


def load_saved_model(model_path='nn_model_best_mre.pth'):
    """加载保存的模型和标准化参数"""
    try:
        # 加载保存的检查点
        checkpoint = torch.load(model_path, weights_only=False)

        # 初始化模型并加载权重
        model = NN_MonitorModel()
        model.load_state_dict(checkpoint['model_state_dict'])

        # 提取标准化参数
        scaler_X_mean = checkpoint['scaler_X_mean']
        scaler_X_scale = checkpoint['scaler_X_scale']
        scaler_Y_mean = checkpoint['scaler_Y_mean']
        scaler_Y_scale = checkpoint['scaler_Y_scale']

        print(f"✅ 成功从 {model_path} 加载模型")
        return model, scaler_X_mean, scaler_X_scale, scaler_Y_mean, scaler_Y_scale

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("请确保模型文件存在并且是通过 nn.py 脚本正确保存的")
        raise


class ModelEvaluator:
    """模型评估器，包含多种评估指标的计算方法"""

    def evaluate_model(self, y_true, y_pred):
        """
        评估模型性能的方法，包含 RMSE、R² 和 NSE 的计算组件。

        参数:
            y_true: 真实值
            y_pred: 预测值

        返回:
            rmse: 均方根误差
            r2: 决定系数 R²
            nse: Nash-Sutcliffe效率系数
            residuals: 残差
            mre:平均相对误差
        """
        # 计算均方根误差 (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 将真实值和预测值转换为数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 计算真实值和预测值的均值
        mean_y_true = np.mean(y_true)
        mean_y_pred = np.mean(y_pred)

        # 计算真实值与其均值之差 (TM) 和预测值与其均值之差 (TP)
        TM = y_true - mean_y_true
        TP = y_pred - mean_y_pred

        # 计算实际值与预测值之间的乘积和 (SSSP)
        SSSP = np.sum(TM * TP)

        # 计算平方和 (SSP)
        SSP = SSSP ** 2

        # 计算预测值的总平方和 (SSR)
        SSR = np.sum((y_pred - mean_y_pred) ** 2)

        # 计算真实值的总平方和 (SST)
        SST = np.sum((y_true - mean_y_true) ** 2)

        # 计算决定系数 R²
        r2 = SSP / (SSR * SST)

        # 重新计算总平方和误差 (SSR) - 用于 NSE 计算
        SSR = np.sum((y_true - y_pred) ** 2)

        # 重新计算总平方和 (SST) - 用于 NSE 计算
        SST = np.sum((y_true - mean_y_true) ** 2)

        # 计算效率系数 (NSE)
        nse = 1 - (SSR / SST)

        # 计算残差
        residuals = np.array(y_true) - np.array(y_pred)

        # 计算平均相对误差
        mre = np.mean(np.abs(residuals / y_true))

        #相对误差
        relative_errors = np.abs(residuals / y_true)

        # 返回 RMSE、R²、 NSE、mre 和 relative_errors
        return rmse, r2, nse, residuals, mre, relative_errors


def evaluate_model():
    # 加载验证数据
    try:
        data_X_val_np = pd.read_excel('data/模型输入数据_验证2.xlsx', header=None).values
        data_Y_val_np = pd.read_excel('data/模型输出数据_验证2.xlsx', header=None).values
        print(f"✅ 成功加载验证数据: X.shape={data_X_val_np.shape}, Y.shape={data_Y_val_np.shape}")
    except Exception as e:
        print(f"❌ 验证数据加载失败: {e}")
        return

    # 加载模型和标准化参数
    model, scaler_X_mean, scaler_X_scale, scaler_Y_mean, scaler_Y_scale = load_saved_model()

    # 进行预测
    predictions = predict_monitor_points(model, data_X_val_np, scaler_X_mean, scaler_X_scale, scaler_Y_mean,
                                         scaler_Y_scale)

    # 创建评估器实例
    evaluator = ModelEvaluator()

    # 使用自定义评估方法计算指标
    rmse, r2, nse, residuals, mre, relative_errors = evaluator.evaluate_model(data_Y_val_np, predictions)

    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"决定系数 (R²): {r2:.6f}")
    print(f"Nash-Sutcliffe效率系数 (NSE): {nse:.6f}")
    print(f"平均相对误差 (MRE): {mre:.6f}")

    # 保存预测结果到 Excel 文件，包含残差
    results_df = pd.DataFrame({
        '真实值': data_Y_val_np.flatten(),
        '预测值': predictions.flatten(),
        '残差': residuals.flatten(),
        '相对误差':relative_errors.flatten(),
    })
    results_df.to_excel('data/预测结果.xlsx', index=False)
    print(f"✅ 预测结果已保存至: 预测结果.xlsx")

    # 绘制预测值与真实值的散点图
    plt.figure(figsize=(10, 8))

    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei"]

    plt.scatter(data_Y_val_np.flatten(), predictions.flatten(), alpha=0.5, label='预测值 vs 真实值')

    # # 添加 y = x 参考线
    # min_val = min(data_Y_val_np.min(), predictions.min())
    # max_val = max(data_Y_val_np.max(), predictions.max())
    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    # 添加拟合直线
    z = np.polyfit(data_Y_val_np.flatten(), predictions.flatten(), 1)
    p = np.poly1d(z)
    plt.plot(data_Y_val_np.flatten(), p(data_Y_val_np.flatten()), 'g-', label=f'拟合直线: y = {z[0]:.2f}x + {z[1]:.2f}')

    # 将 textstr 放在图像左上角（坐标轴的相对位置）
    textstr = f'RMSE = {rmse:.4f}\nR^2 = {r2:.4f}\nNSE = {nse:.4f}\nMRE = {mre:.4f}'
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=14, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('真实值（mg/L）')
    plt.ylabel('替代模型预测值（mg/L）')
    plt.title('预测值与真实值的散点图')
    plt.legend(loc='lower right')  # 图例放在右下角
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像前忽略字体警告
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.savefig('png/模型评估指标.png')

    plt.close()

    print(f"✅ 评估完成，散点图保存至: 预测值与真实值的散点图.png")


if __name__ == "__main__":
    evaluate_model()