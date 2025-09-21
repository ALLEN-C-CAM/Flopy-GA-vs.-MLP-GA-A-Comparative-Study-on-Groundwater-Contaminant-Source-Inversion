import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.callback import Callback
from torch.serialization import add_safe_globals
from nn import NN_MonitorModel

# 中文支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

add_safe_globals([
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.Float64DType
])


# 神经网络工具类
class nn:
    @staticmethod
    def predict_monitor_points(model, new_input, scaler_X_mean, scaler_X_scale, scaler_Y_mean=None, scaler_Y_scale=None):
        new_input = np.atleast_2d(new_input)
        new_input_scaled = (new_input - scaler_X_mean) / scaler_X_scale
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)
            prediction = model(input_tensor).numpy()
        if scaler_Y_mean is not None and scaler_Y_scale is not None:
            prediction = prediction * scaler_Y_scale + scaler_Y_mean
        return prediction.reshape(-1)

    @staticmethod
    def load_saved_model(path='nn_model.pth'):
        checkpoint = torch.load(path, weights_only=True)
        model = NN_MonitorModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['scaler_X_mean'], checkpoint['scaler_X_scale'], checkpoint.get('scaler_Y_mean'), checkpoint.get('scaler_Y_scale')

    @staticmethod
    def evaluate_model(y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        residuals = y_true - y_pred
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mre = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.mean(np.abs(y_true)) != 0 else 0
        TM = y_true - np.mean(y_true)
        TP = y_pred - np.mean(y_pred)
        r2 = (np.sum(TM * TP) ** 2) / (np.sum(TM ** 2) * np.sum(TP ** 2)) if np.sum(TM ** 2) * np.sum(TP ** 2) != 0 else 0
        nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2) if np.sum((y_true - np.mean(y_true)) ** 2) != 0 else 0
        return rmse, r2, nse, residuals, mre


class Objective_function(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=20, n_obj=1, n_constr=0,
            xl=np.array([4, 7, 4, 4] + [10] * 16),
            xu=np.array([4, 7, 4, 4] + [100] * 16)
        )
        self.y_true = self._load_real_data()

    def _load_real_data(self):
        try:
            df = pd.read_excel('data/监测井真实记录数据.xlsx', sheet_name='Sheet1', header=None)
            if df.empty:
                raise ValueError("文件为空")
            print(f"✅ 成功加载真实数据，形状: {df.shape}")
            return df.values.flatten()
        except Exception as e:
            print(f"❌ 加载真实数据失败: {e}")
            return np.random.rand(70)

    def _evaluate(self, x, out, *args, **kwargs):
        model, mean, scale, y_mean, y_scale = nn.load_saved_model()
        y_pred = nn.predict_monitor_points(model, x, mean, scale, y_mean, y_scale)
        *_, _, mre = nn.evaluate_model(self.y_true, y_pred)  # 直接获取MRE
        out["F"] = mre  # 目标是最小化MRE


class OptimizationCallback(Callback):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval
        self.start_time = time.time()
        self.mre_history = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % self.interval == 0:
            print(f"第 {gen} 代, 耗时: {time.time() - self.start_time:.2f}s")
        best_x = algorithm.pop.get("X")[np.argmin(algorithm.pop.get("F"))]
        model, mean, scale, y_mean, y_scale = nn.load_saved_model()
        pred = nn.predict_monitor_points(model, best_x, mean, scale, y_mean, y_scale)
        _, _, _, _, mre = nn.evaluate_model(algorithm.problem.y_true, pred)
        self.mre_history.append(mre)


class Optimization:
    def __init__(self):
        self.model, self.mean, self.scale, self.y_mean, self.y_scale = nn.load_saved_model()

    def run(self):
        problem = Objective_function()
        callback = OptimizationCallback()
        algorithm = GA(
            pop_size=100,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=3, vtype=float),
            mutation=PM(prob=0.02, eta=3, vtype=float),
            repair=RoundingRepair(),
            eliminate_duplicates=True
        )
        res = minimize(problem, algorithm, termination=('n_gen', 500), seed=1, callback=callback, verbose=False)
        best_solution = res.X
        prediction = nn.predict_monitor_points(self.model, best_solution, self.mean, self.scale, self.y_mean, self.y_scale)
        y_true = problem.y_true[:len(prediction)]
        _, _, _, residuals, mre = nn.evaluate_model(y_true, prediction)

        self.save_prediction_to_excel(y_true, prediction, residuals, mre)
        self.save_best_solution_to_excel(best_solution)
        self.visualize_prediction_vs_true(y_true, prediction, residuals, mre)
        self.visualize_mre(callback.mre_history)

        print("\n最优解参数:", best_solution)
        print(f"\n最优解预测 MRE: {res.F[0]:.4f}")
        return best_solution

    def save_prediction_to_excel(self, y_true, y_pred, residuals, mre_values, filename='data/预测结果2.xlsx'):
        df = pd.DataFrame({
            '真实值': y_true,
            '预测值': y_pred,
            '残差': residuals,
            '相对误差 (%)': np.abs((residuals / y_true)) * 100 if np.mean(np.abs(y_true)) != 0 else np.zeros_like(y_true)
        })
        df.to_excel(filename, sheet_name='预测结果', index=False)
        print(f"✅ 预测结果已保存到 {filename}")

    def save_best_solution_to_excel(self, best_solution, filename='data/最优结果2.xlsx'):
        df = pd.DataFrame({f'参数{i+1}': [val] for i, val in enumerate(best_solution)})
        df.to_excel(filename, sheet_name='最优参数', index=False)
        print(f"✅ 最优参数已保存到 {filename}")

    def visualize_prediction_vs_true(self, y_true, y_pred, residuals, mre, save_path='png/prediction_vs_true2.png'):
        time_points = np.arange(len(y_true))
        rmse, r2, nse, _, _ = nn.evaluate_model(y_true, y_pred)
        relative_error = np.abs(residuals / y_true) * 100  # 相对误差（%）

        plt.figure(figsize=(14, 12))

        # 1. 预测 vs 真实值
        plt.subplot(3, 1, 1)
        plt.plot(time_points, y_true, 'b-', label='真实值', linewidth=2)
        plt.plot(time_points, y_pred, 'r--', label='预测值', linewidth=2)
        plt.xlabel('时间点');
        plt.ylabel('监测值')
        plt.title(f'最优解对比图 (MRE: {mre:.2f}%)');
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.text(0.02, 0.95,
                 f'性能指标:\nRMSE: {rmse:.4f}\nMRE: {mre:.2f}%\nR²: {r2:.4f}\nNSE: {nse:.4f}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. 残差图
        plt.subplot(3, 1, 2)
        plt.bar(time_points, residuals, color='gray', alpha=0.7)
        plt.axhline(0, color='red', linewidth=1)
        plt.xlabel('时间点');
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 3. 相对误差图
        plt.subplot(3, 1, 3)
        plt.plot(time_points, relative_error, 'g-', marker='o', label='相对误差 (%)')
        plt.xlabel('时间点');
        plt.ylabel('相对误差 (%)')
        plt.title('相对误差变化图')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

    def visualize_mre(self, mre_history, save_path='png/mre_optimization2.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(mre_history) + 1), mre_history, 'b-', linewidth=2)
        min_mre = min(mre_history)
        min_gen = mre_history.index(min_mre) + 1
        plt.scatter(min_gen, min_mre, color='red', s=100, label=f'最优MRE: {min_mre:.2f}% (第{min_gen}代)')
        plt.xlabel('迭代代数'); plt.ylabel('平均相对误差 (MRE, %)')
        plt.title('MRE优化过程'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()


if __name__ == "__main__":
    optimizer = Optimization()
    optimizer.run()
