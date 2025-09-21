
import numpy as np
import flopy
import pandas as pd
import matplotlib.pyplot as plt
import os
# 中文支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False



def build_transient_model():
    # ===== 模型基本参数 =====
    nlay = 1  # 含水层数
    nrow = 8  # 行数
    ncol = 13  # 列数
    delr = 100.0  # 列宽 (m)
    delc = 100.0  # 行宽 (m)
    top = 110.0  # 顶部高程
    botm = [70.5]  # 底部高程
    Lx = delc * ncol  # 模型长度X方向
    Ly = delr * nrow  # 模型长度Y方向

    # === 时间离散化设置 ===
    nper = 20  # 应力期数
    perlen = [90] * nper  # 每个应力期长度 (天)
    nstp = [5] * nper  # 每个应力期时间步数
    tsmult = [1] * nper  # 时间步增长因子

    # === 空间离散化 ===
    idomain = np.ones((nlay, nrow, ncol), dtype=int)#规则矩形案例

    # === 含水层参数 ===
    hk = 17.28  # 均质含水层，渗透系数 (m/day)

    # === 储水参数 ===
    sy = np.full((nlay, nrow, ncol), 0.3)  # 给水度 （潜水使用）
    ss = np.full((nlay, nrow, ncol), 1e-5)  # 储水率 (1/m)  （承压水使用）
    # === 初始条件 ===
    strt_water = np.ones((nlay, nrow, ncol), dtype=float)
    df = pd.read_excel('初始水位.xlsx', sheet_name='Sheet1', header=None)
    excel_data = df.values
    for i in range(nrow):
        for j in range(ncol):
            strt_water[0, i, j] = excel_data[i, j]

    # === 溶质运移参数 ===
    porosity = 0.3  # 孔隙率
    al = 40.0  # 纵向弥散度 (m)
    trpt = 0.24  # 横向弥散比
    ath1 = al * trpt  # 横向弥散度 (m)
    sconc = 0.0  # 初始浓度

    # === 源汇项参数 ===
    recharge = 0.0000864  # 补给率 (m/day)
    cwell2 = [47, 15,   37, 10 ] + [0] * (nper - 4) #g/s
    cwell1 = [30, 58.8, 10, 5  ] + [0] * (nper - 4)

    # === 创建模拟 ===
    sim = flopy.mf6.MFSimulation(
        sim_name='transient_case',
        sim_ws='simulation_transient_case3',
        exe_name='mf6'
    )

    # === 时间离散化 ===
    flopy.mf6.ModflowTdis(
        sim,
        nper=nper,
        perioddata=list(zip(perlen, nstp, tsmult)),
        time_units='DAYS'
    )

    # === 地下水流动模型 ===
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname='gwf_transient',
        save_flows=True,
        model_nam_file='gwf_transient.nam'
    )

    # === GWF迭代求解器 ===（关键修改点1：独立IMS配置）
    ims_gwf = flopy.mf6.ModflowIms(
        sim,
        pname='ims_gwf',
        complexity='MODERATE',
        filename='gwf_transient.ims',
        outer_dvclose=1e-6,
        inner_dvclose=1e-6,
        rcloserecord=1e-6,
        linear_acceleration='BICGSTAB'
    )
    sim.register_ims_package(ims_gwf, [gwf.name])  # 显式绑定到GWF

    # === 空间离散化 ===
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain
    )

    # === 含水层属性 ===
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,  # 1=非承压含水层
        k=hk,
        k33=hk,
        save_specific_discharge=True
    )

    # === 储水参数 ===
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=1,  # 非承压含水层
        ss=ss,
        sy=sy,
        steady_state={0: False},  # 初始为瞬态
        transient={0: True}
    )

    # === 初始条件 ===
    ic = flopy.mf6.ModflowGwfic(
        gwf,
        strt=strt_water
    )

    # === 边界条件 ===
    # 定水头边界
    chd_spd = []
    for i in np.arange(nrow):
        chd_spd.append([(0, i, 0), 100.0])
        chd_spd.append([(0, i, ncol - 1), 80.0])
    chd_spd = {0: chd_spd}

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_spd,
        print_input=True,
        print_flows=True
    )

    # === 井流 ===
    wel_spd = {}
    for per in range(nper):
        wel_spd[per] = [
            [(0, 3-1, 3-1), 1, cwell1[per]*1000*24*60*60/1000],
            [(0, 6-1, 3-1), 1, cwell2[per]*1000*24*60*60/1000]
        ]

    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_spd,
        auxiliary='CONCENTRATION',
        pname='WEL-1',
        print_input=True,
        print_flows=True
    )

    # === 输出控制 ===
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord='gwf_transient.hds',
        budget_filerecord='gwf_transient.bud',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
        printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
    )

    # ===== 溶质运移模型 =====
    gwt = flopy.mf6.ModflowGwt(
        sim,
        modelname='gwt_transient',
        save_flows=True,
        model_nam_file='gwt_transient.nam'
    )

    # === GWT迭代求解器 ===（关键修改点2：独立IMS配置）
    ims_gwt = flopy.mf6.ModflowIms(
        sim,
        pname='ims_gwt',
        complexity='MODERATE',
        filename='gwt_transient.ims',
        outer_dvclose=1e-6,
        inner_dvclose=1e-6,
        rcloserecord=1e-6,
        linear_acceleration='BICGSTAB'
    )
    sim.register_ims_package(ims_gwt, [gwt.name])  # 显式绑定到GWT

    # === 空间离散化 ===
    flopy.mf6.ModflowGwtdis(
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain
    )

    # === 初始浓度 ===
    flopy.mf6.ModflowGwtic(
        gwt,
        strt=sconc,
        filename='gwt_transient.ic'
    )

    # === 运移参数 ===
    flopy.mf6.ModflowGwtadv(
        gwt,
        scheme='UPSTREAM'
    )

    flopy.mf6.ModflowGwtdsp(
        gwt,
        alh=al,
        ath1=ath1,
        filename='gwt_transient.dsp'
    )

    # === 孔隙介质属性 ===
    flopy.mf6.ModflowGwtmst(
        gwt,
        porosity=porosity,
        filename='gwt_transient.mst'
    )

    # === 源汇项耦合 ===
    sourcerecarray = [('WEL-1', 'AUX', 'CONCENTRATION')]
    flopy.mf6.ModflowGwtssm(
        gwt,
        sources=sourcerecarray,
        filename='gwt_transient.ssm'
    )

    # === 输出控制 ===
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord='gwt_transient.ucn',
        budget_filerecord='gwt_transient.cbc',
        saverecord=[('CONCENTRATION', 'ALL'), ('BUDGET', 'ALL')],
        printrecord=[('CONCENTRATION', 'LAST'), ('BUDGET', 'LAST')]
    )

    # === 耦合GWF与GWT ===
    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype='GWF6-GWT6',
        exgmnamea=gwf.name,
        exgmnameb=gwt.name,
        filename='transient.gwfgwt'
    )

    return sim




# def plot_transient_results(sim):
#     # 结果可视化
#     gwf = sim.get_model('gwf_transient')
#     gwt = sim.get_model('gwt_transient')
#
#     # 创建结果目录
#     os.makedirs('images/head', exist_ok=True)
#     os.makedirs('images/concentration', exist_ok=True)
#     os.makedirs('images/flowpath', exist_ok=True)
#     os.makedirs('data', exist_ok=True)
#
#     # 获取模型网格信息
#     modelgrid = gwf.modelgrid
#     xcenters = modelgrid.xcellcenters
#     ycenters = modelgrid.ycellcenters
#
#     # 提取水头数据
#     head_obj = gwf.output.head()
#     # 提取浓度数据
#     conc_obj = gwt.output.concentration()
#     conc_data = conc_obj.get_alldata()
#     conc_data[conc_data > 1e10] = np.nan  # 处理异常值
#
#     # === 输出观测井数据至Excel ===
#     obs_locs = {
#         "WELL1": (0, 7 - 1, 9 - 1),  # 注意：Python索引从0开始
#         "WELL2": (0, 6 - 1, 4 - 1),
#         "WELL3": (0, 5 - 1, 7 - 1),
#         "WELL4": (0, 3 - 1, 5 - 1),
#         "WELL5": (0, 5 - 1, 12 - 1),
#         "WELL6": (0, 2 - 1, 7 - 1),
#     }
#
#     # 选择要可视化的应力期
#     target_periods = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # 应力期（从1开始）
#     time_list = head_obj.get_times()  # 获取所有时间步
#     steps_per_period = 5  # 每个应力期5步
#     time_indices = [(per - 1) * steps_per_period for per in target_periods]
#
#     # 收集观测井数据
#     records = []
#     for i, idx in enumerate(time_indices):
#         time = time_list[idx]
#         head = head_obj.get_data(totim=time)
#         conc = conc_obj.get_data(totim=time)
#
#         for well, loc in obs_locs.items():
#             row = {
#                 "Stress_Period": target_periods[i],
#                 "Time_Day": time,
#                 "Well": well,
#                 "Head": head[loc],
#                 "Concentration": conc[loc]
#             }
#             records.append(row)
#
#     df_obs = pd.DataFrame(records)
#     df_obs.to_excel("data/Observation_Wells.xlsx", index=False, sheet_name="Observation_Data")
#     print("观测井数据已保存至 'data/Observation_Wells.xlsx'")
#
#     # === 可视化 ===
#
#     # 1. 单独绘制水头分布图
#     for i, idx in enumerate(time_indices):
#         time = time_list[idx]
#         head = head_obj.get_data(totim=time)
#
#         plt.figure(figsize=(10, 8))
#         plt.title(f'水头分布 (应力期 {target_periods[i]}, 时间 {time:.1f} 天)')
#         im = plt.imshow(head[0], cmap='viridis', origin='upper',
#                         extent=[0, modelgrid.extent[1], 0, modelgrid.extent[3]])
#         plt.colorbar(im, label='水头 (m)')
#         plt.xlabel('X坐标 (m)')
#         plt.ylabel('Y坐标 (m)')
#
#         # 添加等值线
#         contour = plt.contour(xcenters[0, :], ycenters[:, 0], head[0], 10, colors='black', linewidths=0.5)
#         plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
#
#         # 标记观测井位置
#         for well, loc in obs_locs.items():
#             plt.plot(xcenters[loc[1], loc[2]], ycenters[loc[1], loc[2]], 'ro', markersize=5)
#             plt.annotate(well, (xcenters[loc[1], loc[2]], ycenters[loc[1], loc[2]]),
#                          textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
#
#         plt.tight_layout()
#         plt.savefig(f'images/head/head_{target_periods[i]:02d}.png', dpi=300, bbox_inches='tight')
#         plt.close()
#
#     # 2. 单独绘制浓度分布图
#     for i, idx in enumerate(time_indices):
#         time = time_list[idx]
#         conc = conc_obj.get_data(totim=time)
#
#         plt.figure(figsize=(10, 8))
#         plt.title(f'浓度分布 (应力期 {target_periods[i]}, 时间 {time:.1f} 天)')
#         im = plt.imshow(conc[0], cmap='Reds', origin='upper',
#                         extent=[0, modelgrid.extent[1], 0, modelgrid.extent[3]],
#                         vmin=0, vmax=2000)  # 限制浓度显示范围
#         plt.colorbar(im, label='浓度 (mg/L)')
#         plt.xlabel('X坐标 (m)')
#         plt.ylabel('Y坐标 (m)')
#
#         # 添加等值线
#         contour = plt.contour(xcenters[0, :], ycenters[:, 0], conc[0], 10, colors='black', linewidths=0.5)
#         plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
#
#         # 标记观测井位置
#         for well, loc in obs_locs.items():
#             plt.plot(xcenters[loc[1], loc[2]], ycenters[loc[1], loc[2]], 'bo', markersize=5)
#             plt.annotate(well, (xcenters[loc[1], loc[2]], ycenters[loc[1], loc[2]]),
#                          textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
#
#         plt.tight_layout()
#         plt.savefig(f'images/concentration/conc_{target_periods[i]:02d}.png', dpi=300, bbox_inches='tight')
#         plt.close()
#
#     # 3. 绘制观测井水头时间序列图
#     plt.figure(figsize=(12, 8))
#     for well in df_obs['Well'].unique():
#         well_data = df_obs[df_obs['Well'] == well]
#         plt.plot(well_data['Time_Day'], well_data['Head'], 'o-', label=well)
#
#     plt.title('观测井水头变化')
#     plt.xlabel('时间 (天)')
#     plt.ylabel('水头 (m)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('images/head_timeseries.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # 4. 绘制观测井浓度时间序列图
#     plt.figure(figsize=(12, 8))
#     for well in df_obs['Well'].unique():
#         well_data = df_obs[df_obs['Well'] == well]
#         plt.plot(well_data['Time_Day'], well_data['Concentration'], 'o-', label=well)
#
#     plt.title('观测井污染物浓度变化')
#     plt.xlabel('时间 (天)')
#     plt.ylabel('浓度 (mg/L)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('images/concentration_timeseries.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # 5. 绘制流线图（使用比流量）
#     for i, idx in enumerate(time_indices):
#         time = time_list[idx]
#         head = head_obj.get_data(totim=time)
#
#         # 获取比流量数据
#         spdis = gwf.output.budget().get_data(text='DATA-SPDIS', totim=time)[0]
#         qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
#
#         plt.figure(figsize=(10, 8))
#         plt.title(f'流线图 (应力期 {target_periods[i]}, 时间 {time:.1f} 天)')
#
#         # 确保y坐标严格递增
#         y_centers = ycenters[:, 0].copy()
#         y_reversed = False
#         if not np.all(np.diff(y_centers) > 0):
#             # 如果y坐标不是严格递增，反转y轴和qy
#             y_centers = y_centers[::-1]
#             qy = qy[::-1, :].copy()  # 复制以避免视图问题
#             y_reversed = True
#
#         # 绘制水头等值线
#         contour = plt.contour(xcenters[0, :], y_centers, head[0], 10, colors='black', linewidths=0.5)
#         plt.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
#
#         # 绘制流线
#         plt.streamplot(xcenters[0, :], y_centers, -qx[0], -qy,
#                        density=1.5, color='blue', linewidth=1, arrowsize=1.5)
#
#         # 标记观测井位置
#         for well, loc in obs_locs.items():
#             row, col = loc[1], loc[2]
#             # 如果y轴反转，需要调整观测井的y坐标
#             if y_reversed:
#                 true_y = y_centers[-(row + 1)]  # 反转后的y坐标
#             else:
#                 true_y = y_centers[row]
#             plt.plot(xcenters[row, col], true_y, 'ro', markersize=5)
#             plt.annotate(well, (xcenters[row, col], true_y),
#                          textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
#
#         plt.xlabel('X坐标 (m)')
#         plt.ylabel('Y坐标 (m)')
#         plt.tight_layout()
#         plt.savefig(f'images/flowpath/flowpath_{target_periods[i]:02d}.png', dpi=300, bbox_inches='tight')
#         plt.close()
#
#     print(f"可视化结果已保存至 images' 文件夹")
def plot_transient_results(sim):
    # 要提取的应力期（第4、6、8、10、12、14、16、18、20期）
    target_periods = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # 每个应力期内固定取第5个时间步（索引4，因为从0开始）
    time_step_in_per = 4  # 对应第5个时间步

    # 每个应力期长度（天）
    perlen = [90] * 20
    # 计算每个目标应力期对应的总天数
    days_list = [p * 90 for p in target_periods]
    # 修正时间索引计算：每个应力期5个时间步，因此乘以5
    time_indices = [(p - 1) * 5 + time_step_in_per for p in target_periods]

    # 获取浓度数据
    gwt = sim.get_model('gwt_transient')
    conc_obj = gwt.output.concentration()
    conc_alldata = conc_obj.get_alldata()
    print(f"浓度数据维度: {conc_alldata.shape}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # 过滤超出范围的应力期（确保索引不越界）
    valid_indices = [i for i, idx in enumerate(time_indices) if idx < conc_alldata.shape[0]]
    if len(valid_indices) < len(target_periods):
        print(f"警告：部分应力期索引({len(target_periods) - len(valid_indices)})超出数据范围，已过滤")
        target_periods = [target_periods[i] for i in valid_indices]
        days_list = [days_list[i] for i in valid_indices]
        time_indices = [time_indices[i] for i in valid_indices]

    for i, (time_idx, days) in enumerate(zip(time_indices, days_list)):
        print(
            f"提取第 {time_idx} 个时间步的数据 (对应第 {target_periods[i]} 期, 第 {time_step_in_per + 1} 步, 第 {days} 天)")
        conc_data = conc_alldata[time_idx, 0, :, :]
        print(f"数据范围: {conc_data.min()} ~ {conc_data.max()}")

        ax = axes[i]
        # 使用imshow实现平滑显示
        im = ax.imshow(
            conc_data,
            cmap='viridis',
            interpolation='bicubic',  # 平滑插值
            aspect='auto',
            extent=[0, gwt.modelgrid.ncol * 100, 0, gwt.modelgrid.nrow * 100]
        )
        ax.set_title(f'第 {days} 天', fontsize=12)
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)


    # 统一设置图例
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='浓度(mg/L)')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig('模拟结果.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 构建并运行模型
    sim = build_transient_model()
    sim.write_simulation()

    success, buff = sim.run_simulation(
        normal_msg="normal termination",
    )

    if success:
        print("瞬态模拟成功完成!")
        plot_transient_results(sim)
    else:
        print("模拟失败，请检查错误信息")