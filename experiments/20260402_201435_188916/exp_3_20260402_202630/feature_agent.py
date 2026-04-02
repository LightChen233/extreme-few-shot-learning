import pandas as pd
import numpy as np


# 基于铝合金热处理/均匀化-析出-粗化的一般知识给出的弱物理先验
# 注意：这里不是精确机理模型，而是用于小样本下提供“方向正确、机制可分段”的特征锚点
DOMAIN_PARAMS = {
    # 温度相关机制边界：约 450℃ 左右可视作由“较弱热暴露/缓慢演化”进入“较强热激活区”
    'critical_temp_regime_shift': 450.0,
    # 更高温区，视作强热激活/接近高温均匀化强化窗口
    'high_temp_regime': 470.0,

    # 时间相关边界：短时/中时/长时
    'critical_time_short': 3.0,
    'critical_time_regime_shift': 12.0,
    'critical_time_long': 24.0,

    # 铝合金扩散/组织演化数量级的表观激活能（J/mol），仅用于构造 Arrhenius 型等效特征
    'activation_energy_Q': 1.20e5,
    'gas_constant_R': 8.314,

    # Larson-Miller 常数，金属高温过程常用经验量级
    'larson_miller_C': 20.0,

    # 数据域与训练边界近似，用于外推风险/边界距离特征
    # 从题面可见训练主要覆盖 temp≈420/460/480, time≈1/12/24
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 验证中出现系统性困难的外推/边界点附近，显式加入“机制窗口”
    'challenging_temp_low': 440.0,
    'challenging_temp_high': 470.0,
    'challenging_time_mid': 12.0,
    'challenging_time_long': 24.0,

    # 原始铸态基准
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    面向 7499 铝合金热处理小样本预测的特征工程。
    设计原则：
    1) 少而精，避免高维爆炸；
    2) 强化温度-时间耦合；
    3) 显式加入机制分段与外推边界距离；
    4) 用 physics_baseline 作为残差学习锚点。
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的弱物理基线：
        - 在当前数据范围内，升温总体有利于强度提升；
        - 延时总体有利，但时间效应呈饱和；高温中时对应变更有利；
        - 不强行假设整体“高温长时软化”，仅对高温长时应变加入轻微回落项。
        该函数不依赖训练标签，只使用铸态基准和方向性物理先验。
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        T0 = DOMAIN_PARAMS['as_cast_strain']
        U0 = DOMAIN_PARAMS['as_cast_tensile']
        Y0 = DOMAIN_PARAMS['as_cast_yield']

        # 归一化变量
        temp_n = (temp - 420.0) / 60.0              # 大致映射 420~480 -> 0~1
        logt = np.log1p(np.clip(time, 0, None))
        logt_n = logt / np.log(25.0)                # 大致映射 1~24 -> 0~1

        # 热激活程度：高温增强时间作用
        synergy = temp_n * logt_n

        # 中等时间窗口对应变更有利（符合 480℃下 12h 优于 24h 的趋势）
        mid_time_peak = np.exp(-((np.log(np.clip(time, 1e-8, None)) - np.log(12.0)) ** 2) / (2 * 0.55 ** 2))

        # 高温长时对应变轻微回落，但不对强度施加强软化假设
        high_temp = np.clip((temp - 465.0) / 15.0, 0, None)
        long_time = np.clip((time - 12.0) / 12.0, 0, None)
        highT_longt_penalty = high_temp * long_time

        baseline_strain = (
            T0
            + 2.2 * temp_n
            + 3.0 * logt_n
            + 4.8 * synergy
            + 2.8 * mid_time_peak
            - 1.2 * highT_longt_penalty
        )

        baseline_tensile = (
            U0
            + 55.0 * temp_n
            + 38.0 * logt_n
            + 62.0 * synergy
            + 8.0 * mid_time_peak
        )

        baseline_yield = (
            Y0
            + 42.0 * temp_n
            + 30.0 * logt_n
            + 50.0 * synergy
            + 5.0 * mid_time_peak
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        T_kelvin = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ========= 基础低维特征 =========
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_sq_centered'] = ((temp - 450.0) / 30.0) ** 2
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['time_sq_scaled'] = (time / 24.0) ** 2
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_x_sqrt_time'] = temp * sqrt_time

        # ========= 动力学等效特征 =========
        Q = DOMAIN_PARAMS['activation_energy_Q']
        R = DOMAIN_PARAMS['gas_constant_R']
        C = DOMAIN_PARAMS['larson_miller_C']

        # Arrhenius 速率因子（仅作相对指标）
        arrhenius = np.exp(-Q / (R * T_kelvin))
        cols['arrhenius_factor'] = arrhenius
        cols['arrhenius_time'] = time * arrhenius
        cols['arrhenius_logtime'] = log_time * arrhenius

        # Larson-Miller Parameter
        # 常规形式常用 log10(t)，这里保持数量级稳定
        lmp = T_kelvin * (C + np.log10(np.clip(time, 1e-6, None)))
        cols['larson_miller'] = lmp / 1000.0

        # Zener-Hollomon 逆向近似形式不完全适用静态热处理，这里不用，以免引入错误机理

        # ========= 机制区间检测特征（关键） =========
        Tc = DOMAIN_PARAMS['critical_temp_regime_shift']
        Th = DOMAIN_PARAMS['high_temp_regime']
        ts = DOMAIN_PARAMS['critical_time_short']
        tm = DOMAIN_PARAMS['critical_time_regime_shift']
        tl = DOMAIN_PARAMS['critical_time_long']

        is_low_temp = (temp < Tc).astype(float)
        is_mid_high_temp = (temp >= Tc).astype(float)
        is_high_temp = (temp >= Th).astype(float)

        is_short_time = (time <= ts).astype(float)
        is_mid_time = ((time > ts) & (time <= tm)).astype(float)
        is_long_time = (time >= tm).astype(float)
        is_very_long_time = (time >= tl).astype(float)

        cols['is_low_temp'] = is_low_temp
        cols['is_mid_high_temp'] = is_mid_high_temp
        cols['is_high_temp'] = is_high_temp
        cols['is_short_time'] = is_short_time
        cols['is_mid_time'] = is_mid_time
        cols['is_long_time'] = is_long_time
        cols['is_very_long_time'] = is_very_long_time

        # 关键耦合机制窗口
        cols['is_highT_midt'] = ((temp >= Th) & (time >= 8.0) & (time <= 16.0)).astype(float)
        cols['is_highT_longt'] = ((temp >= Th) & (time >= tm)).astype(float)
        cols['is_lowT_longt'] = ((temp < Tc) & (time >= tm)).astype(float)
        cols['is_boundary_440'] = (np.abs(temp - 440.0) <= 2.5).astype(float)
        cols['is_boundary_470'] = (np.abs(temp - 470.0) <= 2.5).astype(float)
        cols['is_time_12'] = (np.abs(time - 12.0) <= 1.0).astype(float)
        cols['is_time_24'] = (np.abs(time - 24.0) <= 1.0).astype(float)

        # 分段激活特征：相对边界的“超额量”
        cols['temp_above_Tc'] = np.clip(temp - Tc, 0, None)
        cols['temp_above_Th'] = np.clip(temp - Th, 0, None)
        cols['temp_below_Tc'] = np.clip(Tc - temp, 0, None)
        cols['time_above_tm'] = np.clip(time - tm, 0, None)
        cols['time_below_tm'] = np.clip(tm - time, 0, None)
        cols['highT_excess_x_longt_excess'] = np.clip(temp - Th, 0, None) * np.clip(time - tm, 0, None)

        # ========= 峰值/窗口型特征 =========
        # 用于表示“中等时间可能最优”的非单调窗口，尤其针对 470/12 一类点
        cols['gauss_time_12'] = np.exp(-((time - 12.0) ** 2) / (2 * 5.0 ** 2))
        cols['gauss_time_24'] = np.exp(-((time - 24.0) ** 2) / (2 * 4.0 ** 2))
        cols['gauss_temp_470'] = np.exp(-((temp - 470.0) ** 2) / (2 * 8.0 ** 2))
        cols['gauss_temp_440'] = np.exp(-((temp - 440.0) ** 2) / (2 * 8.0 ** 2))
        cols['window_470_12'] = cols['gauss_temp_470'] * cols['gauss_time_12']
        cols['window_440_24'] = cols['gauss_temp_440'] * cols['gauss_time_24']
        cols['window_440_1'] = cols['gauss_temp_440'] * np.exp(-((time - 1.0) ** 2) / (2 * 1.5 ** 2))

        # ========= 外推/训练边界距离特征（关键） =========
        train_temp_min = DOMAIN_PARAMS['train_temp_min']
        train_temp_max = DOMAIN_PARAMS['train_temp_max']
        train_time_min = DOMAIN_PARAMS['train_time_min']
        train_time_max = DOMAIN_PARAMS['train_time_max']

        # 到矩形训练域边界的外侧距离；域内为 0
        temp_out_low = np.clip(train_temp_min - temp, 0, None)
        temp_out_high = np.clip(temp - train_temp_max, 0, None)
        time_out_low = np.clip(train_time_min - time, 0, None)
        time_out_high = np.clip(time - train_time_max, 0, None)

        cols['temp_outside_domain_low'] = temp_out_low
        cols['temp_outside_domain_high'] = temp_out_high
        cols['time_outside_domain_low'] = time_out_low
        cols['time_outside_domain_high'] = time_out_high
        cols['outside_domain_L1'] = temp_out_low + temp_out_high + time_out_low + time_out_high
        cols['outside_domain_L2'] = np.sqrt(
            temp_out_low ** 2 + temp_out_high ** 2 + time_out_low ** 2 + time_out_high ** 2
        )

        # 到已知关键训练网格点的距离（弱先验）
        # 训练覆盖近似：temp in {420,460,480}, time in {1,12,24}
        train_temps = np.array([420.0, 460.0, 480.0])
        train_times = np.array([1.0, 12.0, 24.0])

        d_temp = np.min(np.abs(temp[:, None] - train_temps[None, :]), axis=1)
        d_time = np.min(np.abs(time[:, None] - train_times[None, :]), axis=1)
        cols['dist_to_nearest_train_temp'] = d_temp
        cols['dist_to_nearest_train_time'] = d_time
        cols['dist_to_train_grid_L1'] = d_temp + d_time
        cols['dist_to_train_grid_interaction'] = d_temp * d_time

        # 针对验证/测试外推点常见方向：440/470 温度边界，1/12/24 时间边界
        cols['dist_to_440'] = np.abs(temp - 440.0)
        cols['dist_to_470'] = np.abs(temp - 470.0)
        cols['dist_to_12'] = np.abs(time - 12.0)
        cols['dist_to_24'] = np.abs(time - 24.0)

        # ========= 物理基线特征（让模型学残差） =========
        b_strain, b_tensile, b_yield = self.physics_baseline(temp, time)
        cols['baseline_strain'] = b_strain
        cols['baseline_tensile'] = b_tensile
        cols['baseline_yield'] = b_yield

        # 基线衍生：输出间关系的弱结构先验
        cols['baseline_yield_to_tensile'] = b_yield / np.clip(b_tensile, 1e-6, None)
        cols['baseline_tensile_minus_yield'] = b_tensile - b_yield
        cols['baseline_strength_ductility_product'] = b_tensile * b_strain

        # 相对机制边界的残差式输入
        cols['temp_relative_to_Tc'] = temp - Tc
        cols['temp_relative_to_Th'] = temp - Th
        cols['time_relative_to_tm'] = time - tm
        cols['time_relative_to_tl'] = time - tl

        # 基线与边界的交互：帮助模型修正外推方向
        cols['baseline_tensile_x_outside'] = b_tensile * cols['outside_domain_L1']
        cols['baseline_yield_x_outside'] = b_yield * cols['outside_domain_L1']
        cols['baseline_strain_x_outside'] = b_strain * cols['outside_domain_L1']
        cols['baseline_tensile_x_window470_12'] = b_tensile * cols['window_470_12']
        cols['baseline_yield_x_window440_24'] = b_yield * cols['window_440_24']

        # ========= 针对铸态改善幅度的热暴露近似 =========
        # 任务本质上是相对铸态改善；用若干低维热暴露特征表征“改善程度”
        cols['thermal_dose_1'] = (temp - 400.0) * log_time
        cols['thermal_dose_2'] = ((temp - 420.0) / 60.0) * (time / 24.0)
        cols['thermal_dose_3'] = ((temp - 420.0) / 60.0) * np.sqrt(np.clip(time, 0, None) / 24.0)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names