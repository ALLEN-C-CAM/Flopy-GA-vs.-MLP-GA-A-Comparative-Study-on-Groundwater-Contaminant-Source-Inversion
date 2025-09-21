import torch
import numpy as np
from torch.serialization import add_safe_globals
from nn import NN_MonitorModel  # 假设模型定义在nn.py中

# 添加安全的全局对象
add_safe_globals([
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.Float64DType  # 新增允许的全局对象
])

def predict_monitor_points(model, new_input, scaler_X_mean, scaler_X_scale, scaler_Y_mean=None, scaler_Y_scale=None):
    """
    使用训练好的模型预测监测点数据

    参数:
        model: 训练好的PyTorch模型
        new_input: 输入数据 (numpy数组)
        scaler_X_mean, scaler_X_scale: 训练数据的标准化参数
        scaler_Y_mean, scaler_Y_scale: 可选的输出反标准化参数

    返回:
        prediction: 预测结果 (numpy数组)
    """
    # 确保输入是正确的维度
    if new_input.ndim == 1:
        new_input = new_input.reshape(1, -1)

    # 应用与训练时相同的标准化
    new_input_scaled = (new_input - scaler_X_mean) / scaler_X_scale

    # 转换为PyTorch张量并进行预测
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)
        prediction = model(input_tensor).numpy()

    # 如果提供了输出标准化参数，则进行反标准化
    if scaler_Y_mean is not None and scaler_Y_scale is not None:
        prediction = prediction * scaler_Y_scale + scaler_Y_mean

    # 重塑输出 (假设输出格式为 [ c1, c2] 每个10个时间步)
    return prediction.reshape(-1, 7, 10)


def load_saved_model(model_path='nn_model.pth'):
    """加载保存的模型和标准化参数"""
    try:
        # 加载保存的检查点
        checkpoint = torch.load(model_path, weights_only=True)

        # 初始化模型并加载权重
        model = NN_MonitorModel()
        model.load_state_dict(checkpoint['model_state_dict'])

        # 提取标准化参数
        scaler_X_mean = checkpoint['scaler_X_mean']
        scaler_X_scale = checkpoint['scaler_X_scale']

        # 检查是否存在输出标准化参数
        scaler_Y_mean = checkpoint.get('scaler_Y_mean', None)
        scaler_Y_scale = checkpoint.get('scaler_Y_scale', None)

        print(f"✅ 成功从 {model_path} 加载模型")
        return model, scaler_X_mean, scaler_X_scale, scaler_Y_mean, scaler_Y_scale

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("请确保模型文件存在并且是通过nn.py脚本正确保存的")
        raise


if __name__ == "__main__":
    # 示例输入数据 (20维特征)
    new_input_14d = np.array([4,7,4,4,35,35,90,90,63,63,47,47,24,24,56,56,43,43,35,35])

    # 进行预测
    predictions = predict_monitor_points(load_saved_model()[0], new_input_14d, *load_saved_model()[1:])

    # 打印预测结果
    print("预测结果 shape:", predictions.shape)
    print("预测结果 ( 每个监测点各10个时间步):")
    print(predictions)
    #保存预测结果为.XLSX文件
    import pandas as pd
    df = pd.DataFrame(predictions.reshape(-1,10))
    df.to_excel('data/prd单次预测结果.xlsx', sheet_name='监测点预测', index=False)

