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
    nrow = 15  # 行数
    ncol = 25  # 列数
    delr = 100.0  # 列宽 (m)
    delc = 100.0  # 行宽 (m)
    top = 110.0  # 顶部高程
    botm = [70.5]  # 底部高程
    Lx = delc * ncol  # 模型长度X方向
    Ly = delr * nrow  # 模型长度Y方向

    # === 时间离散化设置 ===
    nper = 20  # 应力期数
    perlen = [180] * nper  # 每个应力期长度 (天)
    nstp = [10] * nper  # 每个应力期时间步数
    tsmult = [1] * nper  # 时间步增长因子

    # === 空间离散化 ===
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    df = pd.read_excel('有效网格.xlsx', sheet_name='Sheet1', header=None)
    excel_data = df.values
    excel_data = np.nan_to_num(excel_data, nan=0)
    for i in range(nrow):
        for j in range(ncol):
            idomain[0, i, j] = excel_data[i, j]

    # === 含水层参数 ===
    hk = np.ones((nlay, nrow, ncol), dtype=float)
    df = pd.read_excel('非均质渗透系数场.xlsx', sheet_name='Sheet1', header=None)
    excel_data = df.values
    for i in range(nrow):
        for j in range(ncol):
            hk[0, i, j] = excel_data[i, j]

    # === 储水参数 ===
    sy = np.full((nlay, nrow, ncol), 0.28)  # 给水度
    ss = np.full((nlay, nrow, ncol), 1e-5)  # 储水率 (1/m)

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
    cwell2 = [35, 35, 90, 90, 63, 63, 74, 47] + [0] * (nper - 8) #g/s
    cwell1 = [24, 24, 56, 56, 43, 43, 35, 35] + [0] * (nper - 8)

    # === 创建模拟 ===
    sim = flopy.mf6.MFSimulation(
        sim_name='transient_case',
        sim_ws='simulation_transient',
        exe_name='"E:\mf6.6.2_win64\bin\mf6.exe"'
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
    chd_spd = {}
    for per in range(nper):
        chd_values = [
            [(0, 1, 2), 100],
            [(0, 3, 1), 100],
            [(0, 4, 0), 100],
            [(0, 11, 23), 80],
            [(0, 12, 22), 80],
            [(0, 13, 21), 80],
            [(0, 14, 20), 80]
        ]
        chd_spd[per] = chd_values

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
            [(0, 3, 3), 1, cwell1[per]*1000*24*60*60/1000],
            [(0, 3, 6), 1, cwell2[per]*1000*24*60*60/1000]
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

    return sim, gwt  # 返回 gwt 方便后续绘图


def plot_transient_results(sim, gwt):
    # 要提取的应力期（第4、6、8、10、12、14、16、18、20期）
    target_periods = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # 每个应力期内固定取第10个时间步（注意：索引从0开始，所以是9）
    time_step_in_per = 9  # 对应第10个时间步

    # 每个应力期长度（天）
    perlen = [180] * 20
    # 计算每个目标应力期对应的总天数
    days_list = [p * 180 for p in target_periods]
    # 计算对应的全局时间索引（注意：数据第一维是按时间顺序排列的所有时间步）
    time_indices = [(p - 1) * 10 + time_step_in_per for p in target_periods]

    # 获取所有浓度数据
    conc_obj = gwt.output.concentration()
    conc_alldata = conc_obj.get_alldata()
    print(f"浓度数据维度: {conc_alldata.shape}")  # 确认维度: (200, 1, 15, 25)

    fig, axes = plt.subplots(5, 2, figsize=(15, 12))
    axes = axes.flatten()
    #同时将数据保存到EXCL中，每一个工作表为一个应力期。
    for i, (time_idx, days) in enumerate(zip(time_indices, days_list)):
        print(
            f"提取第 {time_idx} 个时间步的数据 (对应第 {target_periods[i]} 期, 第 {time_step_in_per + 1} 步, 第 {days} 天)")
        conc_data = conc_alldata[time_idx, 0, :, :]
        # 同时将数据保存到EXCL中，每一个工作表为一个应力期。

        #处理极端值
        conc_data[conc_data > 1000000] = np.nan
        conc_data[conc_data < -30] = np.nan
        print(f"数据范围: {conc_data.min()} ~ {conc_data.max()}")
        ax = axes[i]
        im = ax.imshow(
            conc_data,
            cmap='viridis',
            interpolation='bicubic',  # 平滑插值
            aspect='auto',
            extent=[0, gwt.modelgrid.ncol * 100, 0, gwt.modelgrid.nrow * 100]
        )
        ax.set_title(f'第 {days} 天', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True)

    # 统一设置图例,并调整位置
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=0.6, label='浓度(mg/L)',)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


    plt.show()
    #保存为PDF文件
    plt.savefig('模拟结果.pdf')


if __name__ == "__main__":
    # 构建并运行模型
    sim, gwt = build_transient_model()
    sim.write_simulation()

    # 关键修改2：指定系统编码为gbk（若需）
    success, buff = sim.run_simulation(
        normal_msg="normal termination",
    )

    if success:
        print("瞬态模拟成功完成!")
        # plot_transient_results(sim, gwt)
    else:
        print("模拟失败，请检查错误信息")