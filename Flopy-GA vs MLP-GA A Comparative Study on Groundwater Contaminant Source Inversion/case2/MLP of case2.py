# # 版本：v1.0
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from torch.utils.data import TensorDataset, DataLoader
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# plt.rcParams['font.family'] = 'SimHei'
# plt.rcParams['axes.unicode_minus'] = False
# # 设置随机种子以确保结果可复现
# torch.manual_seed(42)
# np.random.seed(42)
# class NN_MonitorModel(nn.Module):
#     def __init__(self, input_dim=20, output_dim=70, hidden_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.Tanh(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#     def forward(self, x):
#         return self.net(x)
# def load_and_scale_data(file_X, file_Y):
#     """加载并标准化数据"""
#     try:
#         data_X_np = pd.read_excel(file_X, header=None).values
#         data_Y_np = pd.read_excel(file_Y, header=None).values
#         print(f"✅ 成功加载数据: {file_X} {file_Y}")
#         print(f"数据形状: X={data_X_np.shape}, Y={data_Y_np.shape}")
#
#         # 标准化
#         scaler_X = StandardScaler()
#         scaler_Y = StandardScaler()
#         data_X = scaler_X.fit_transform(data_X_np)
#         data_Y = scaler_Y.fit_transform(data_Y_np)
#
#         return (torch.tensor(data_X, dtype=torch.float32),
#                 torch.tensor(data_Y, dtype=torch.float32),
#                 scaler_X, scaler_Y)
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         raise
# def train_model():
#     # 加载训练集
#     train_X, train_Y, scaler_X, scaler_Y = load_and_scale_data(
#         'data/模型输入数据.xlsx',
#         'data/模型输出数据.xlsx'
#     )
#
#     # 加载验证集（使用训练集的scaler进行转换）
#     val_X_np = pd.read_excel('data/模型输入数据_验证.xlsx', header=None).values
#     val_Y_np = pd.read_excel('data/模型输出数据_验证.xlsx', header=None).values
#
#     val_X = torch.tensor(scaler_X.transform(val_X_np), dtype=torch.float32)
#     val_Y = torch.tensor(scaler_Y.transform(val_Y_np), dtype=torch.float32)
#
#     # 创建数据集和数据加载器
#     train_dataset = TensorDataset(train_X, train_Y)
#     val_dataset = TensorDataset(val_X, val_Y)
#
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
#     # 创建模型
#     model = NN_MonitorModel()
#
#     # 定义优化器和损失函数
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.MSELoss()
#
#     # 学习率调度器
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=50, verbose=True
#     )
#
#     # 训练循环
#     epochs = 1000
#     train_losses = []
#     val_losses = []
#     best_val_loss = float('inf')
#
#     print("\n开始训练模型...")
#     for epoch in range(epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0.0
#         for batch_X, batch_Y in train_loader:
#             optimizer.zero_grad()
#             pred_Y = model(batch_X)
#             loss = criterion(pred_Y, batch_Y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * batch_X.size(0)
#         train_loss /= len(train_loader.dataset)
#         train_losses.append(train_loss)
#
#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch_X, batch_Y in val_loader:
#                 pred_Y = model(batch_X)
#                 loss = criterion(pred_Y, batch_Y)
#                 val_loss += loss.item() * batch_X.size(0)
#         val_loss /= len(val_loader.dataset)
#         val_losses.append(val_loss)
#
#         # 更新学习率并保存最佳模型
#         scheduler.step(val_loss)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scaler_X_mean': scaler_X.mean_,
#                 'scaler_X_scale': scaler_X.scale_,
#                 'scaler_Y_mean': scaler_Y.mean_,
#                 'scaler_Y_scale': scaler_Y.scale_,
#                 'train_losses': train_losses,
#                 'val_losses': val_losses
#             }, 'nn_model.pth')
#             print(f"Epoch {epoch}: 验证损失改善 ({val_loss:.6f})")
#
#         if epoch % 100 == 0 or epoch == epochs - 1:
#             print(f"Epoch {epoch:4d}/{epochs}: "
#                   f"训练损失={train_loss:.6f}, "
#                   f"验证损失={val_loss:.6f}, "
#                   f"LR={optimizer.param_groups[0]['lr']:.2e}")
#
#     # 绘制损失曲线
#     plt.figure(figsize=(10, 5))
#     plt.semilogy(train_losses, label='训练损失')
#     plt.semilogy(val_losses, label='验证损失')
#     plt.xlabel('Epoch')
#     plt.ylabel('损失 (log scale)')
#     plt.legend()
#     plt.title('训练和验证损失曲线')
#     plt.savefig('png/loss_curve.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"\n✅ 训练完成 - 最佳验证损失: {best_val_loss:.6f}")
#     print(f"✅ 模型保存至: nn_model.pth")
#     print(f"✅ 损失曲线保存至: png/loss_curve.png")
# if __name__ == "__main__":
#     train_model()
#
# 版本：v2.0
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
class NN_MonitorModel(nn.Module):
    def __init__(self, input_dim=20, output_dim=70, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
def load_and_scale_data(file_X, file_Y):
    """加载并标准化数据"""
    try:
        data_X_np = pd.read_excel(file_X, header=None).values
        data_Y_np = pd.read_excel(file_Y, header=None).values
        print(f"✅ 成功加载数据: {file_X} {file_Y}")
        print(f"数据形状: X={data_X_np.shape}, Y={data_Y_np.shape}")

        # 标准化
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        data_X = scaler_X.fit_transform(data_X_np)
        data_Y = scaler_Y.fit_transform(data_Y_np)

        return (torch.tensor(data_X, dtype=torch.float32),
                torch.tensor(data_Y, dtype=torch.float32),
                scaler_X, scaler_Y)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise
def compute_mre(pred, target):
    """计算平均相对误差"""
    return torch.mean(torch.abs((pred - target) / (target + 1e-6)))

def combined_loss(pred, target, alpha=0.5):
    """混合损失函数：MSE + MRE"""
    mse = torch.mean((pred - target) ** 2)
    mre = compute_mre(pred, target)
    return alpha * mse + (1 - alpha) * mre


def train_model():
    # 加载训练集
    train_X, train_Y, scaler_X, scaler_Y = load_and_scale_data(
        'data/模型输入数据.xlsx',
        'data/模型输出数据.xlsx'
    )

    # 加载验证集
    val_X_np = pd.read_excel('data/模型输入数据_验证.xlsx', header=None).values
    val_Y_np = pd.read_excel('data/模型输出数据_验证.xlsx', header=None).values

    val_X = torch.tensor(scaler_X.transform(val_X_np), dtype=torch.float32)
    val_Y = torch.tensor(scaler_Y.transform(val_Y_np), dtype=torch.float32)

    # 数据加载器
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=32, shuffle=False)

    # 创建模型
    model = NN_MonitorModel(input_dim=20, output_dim=70, hidden_dim=256)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

    # 训练参数
    epochs = 10000
    train_losses, val_losses, val_mres = [], [], []
    best_val_mre = float('inf')

    print("\n开始训练模型...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            pred_Y = model(batch_X)
            loss = combined_loss(pred_Y, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss, val_mre_total = 0.0, 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                pred_Y = model(batch_X)

                # 计算损失和相对误差（解标准化后计算 MRE）
                loss = combined_loss(pred_Y, batch_Y)
                val_loss += loss.item() * batch_X.size(0)

                pred_np = scaler_Y.inverse_transform(pred_Y.cpu().numpy())
                true_np = scaler_Y.inverse_transform(batch_Y.cpu().numpy())
                batch_mre = np.mean(np.abs((pred_np - true_np) / (true_np + 1e-6)))
                val_mre_total += batch_mre * batch_X.size(0)

        val_loss /= len(val_loader.dataset)
        val_mre = val_mre_total / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_mres.append(val_mre)

        scheduler.step(val_loss)

        # 保存最优模型（基于 MRE）
        if val_mre < best_val_mre:
            best_val_mre = val_mre
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_X_mean': scaler_X.mean_,
                'scaler_X_scale': scaler_X.scale_,
                'scaler_Y_mean': scaler_Y.mean_,
                'scaler_Y_scale': scaler_Y.scale_,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_mres': val_mres
            }, 'nn_model_best_mre.pth')
            print(f"Epoch {epoch}: 验证 MRE 改善为 {val_mre:.6f}")

        # 打印进度
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}/{epochs}: "
                  f"训练损失={train_loss:.6f}, "
                  f"验证损失={val_loss:.6f}, "
                  f"验证MRE={val_mre:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

    # 损失曲线
    plt.figure(figsize=(10, 5))
    plt.semilogy(train_losses, label='训练损失')
    plt.semilogy(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失 (log scale)')
    plt.legend()
    plt.title('训练与验证损失曲线')
    plt.savefig('png/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # MRE 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(val_mres, label='验证MRE')
    plt.xlabel('Epoch')
    plt.ylabel('MRE')
    plt.legend()
    plt.title('验证集 MRE 曲线')
    plt.savefig('png/mre_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 训练完成 - 最佳验证 MRE: {best_val_mre:.6f}")
    print(f"✅ 最佳模型保存至: nn_model_best_mre.pth")
    print(f"✅ 损失曲线保存至: png/loss_curve.png")
    print(f"✅ MRE 曲线保存至: png/mre_curve.png")
if __name__ == "__main__":
    train_model()