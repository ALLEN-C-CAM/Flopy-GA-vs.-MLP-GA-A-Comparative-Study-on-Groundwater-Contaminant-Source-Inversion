import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
from scipy.stats.qmc import Halton
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class LHSample:
    '拉丁超立方算法类'
    '''
    D = 0
    bounds = [[0,1],[0,1]]
    N = 0
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''
    def __init__(self, D, bounds, N):
        self.D = D
        self.bounds = bounds
        self.N = N

    def getSample(self):
        result = np.empty([self.N, self.D])
        temp = np.empty([self.N])
        d = 1.0 / self.N
        for i in range(self.D):
            for j in range(self.N):
                temp[j] = np.random.uniform(
                    low=j * d, high=(j + 1) * d, size=1)[0]
            np.random.shuffle(temp)
            for j in range(self.N):
                result[j, i] = temp[j]
        # 对样本数据进行拉伸
        b = np.array(self.bounds)
        lower_bounds = b[:, 0]
        upper_bounds = b[:, 1]
        if np.any(lower_bounds > upper_bounds):
            print('范围出错')
            return None
        #   sample * (upper_bound - lower_bound) + lower_bound
        np.add(np.multiply(result,
                           (upper_bounds - lower_bounds),
                           out=result),
               lower_bounds,
               out=result)
        return result

# 定义抽样参数
D1 = 2  # 参数个数（S1释放行位置和释放列位置）
D2 = 2 # 参数个数（S2释放行位置和释放列位置）
D3 = 8  # 参数个数（前8个应力期S1的释放浓度）
D4 = 8  # 参数个数（前8个应力期S2的释放浓度）
N = 1  # 拉丁超立方层数
bounds1 = [[3, 5], [6, 8]]  # 参数对应范围（S1释放行位置范围为3到5，S1释放列位置范围为6到8）
bounds2 = [[3, 5], [3, 5]]  # 参数对应范围（S2释放行位置范围为3到5，S2释放列位置范围为3到5）
bounds3 = [[10, 100]] * 8  # 参数对应范围（S1释放浓度范围为10到100）
bounds4 = [[10, 100]] * 8  # 参数对应范围（S2释放浓度范围为10到100）

print("开始进行LHS抽样...")
# 进行抽样
lhs1 = LHSample(D1, bounds1, N)
S1_locations = lhs1.getSample()
lhs2 = LHSample(D2, bounds2, N)
S2_locations = lhs2.getSample()
lhs3 = LHSample(D3, bounds3, N)
S1_concentrations = lhs3.getSample()
lhs4 = LHSample(D3, bounds4, N)
S2_concentrations = lhs4.getSample()

# 保存抽样结果到Excel
sampling_results = pd.DataFrame()
sampling_results['S1_row'] = S1_locations[:, 0].astype(int)
sampling_results['S1_col'] = S1_locations[:, 1].astype(int)
sampling_results['S2_row'] = S2_locations[:, 0].astype(int)
sampling_results['S2_col'] = S2_locations[:, 1].astype(int)
for i in range(8):
    sampling_results[f'S1_cwell_{i + 1}'] = S1_concentrations[:, i].round(2)
for i in range(8):
    sampling_results[f'S2_cwell_{i + 1}'] = S2_concentrations[:, i].round(2)
sampling_results.to_excel('模型输入数据.xlsx', index=False)
print("LHS抽样完成，结果已保存到模型输入数据.xlsx")
