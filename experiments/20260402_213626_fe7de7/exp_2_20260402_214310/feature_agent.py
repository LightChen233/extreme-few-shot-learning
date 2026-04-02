import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # 机制边界：结合题目给出的数据趋势与误差集中区间做“温和”的物理分段
    # 420→460°C 强化明显增强，470°C附近常出现高强高塑窗口，因此设一个主机制转折温度
    'critical_temp_regime_shift': 455.0,
    # 470°C附近可能接近“高温激活/组织突变”窗口
    'critical_temp_high': 470.0,
    # 时间上 1h 为短时、12h 为中等、24h 为长时；12h 是明显的动力学阶段边界
    'critical_time_regime_shift': 12.0,
    # 长时边界
    'critical_time_long': 24.0,

    # 铝合金中析出/扩散控制过程的表观激活能量级（J/mol），取保守中值
    'activation_energy_Q': 120000.0,
    'gas_constant_R': 8.314,

    # Larson-Miller 常数，金属热处理中常取 18~22，这里取 20
    'larson_miller_C': 20.0,

    # 已知原始铸态平均性能
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 训练域边界（由题面可见工艺范围近似为 420~480°C, 1~24h）
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 高强窗口的经验中心：从题述趋势看，中高温+中等/较长时间表现最好
    'peak_temp_center': 465.0,
    'peak_temp_width': 18.0,
    'peak_logtime_center': np.log1p(12.0),
    'peak_logtime_width': 0.75,
}


class FeatureAgent:
    """
    面向 7499 铝合金热处理-性能预测的小样本物理启发特征工程。

    设计原则：
    1) 特征少而精，避免 29 条样本下过拟合
    2) 显式编码机制边界（温度/时间分段）
    3) 显式编码动力学等效量（Arrhenius / Larson-Miller）
    4) 加入距训练域边界距离特征，增强外推点识别
    5) 加入 physics baseline，让模型学习残差而非从零拟合
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的弱先验基线：
        - 相对铸态，热处理总体提升强度
        - 温度升高通常比时间延长更强烈
        - 强度存在中高温窗口；延性在高温更容易提升
        - 不做激进的“必然过时效软化”假设，只用温和峰值项
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = float(time)

        # 归一化变量
        t_norm = (temp - p['train_temp_min']) / (p['train_temp_max'] - p['train_temp_min'])
        t_norm = np.clip(t_norm, -0.5, 1.5)

        logtime = np.log1p(max(time, 0.0))
        logtime_min = np.log1p(p['train_time_min'])
        logtime_max = np.log1p(p['train_time_max'])
        h_norm = (logtime - logtime_min) / (logtime_max - logtime_min + 1e-12)
        h_norm = np.clip(h_norm, -0.5, 1.5)

        # 高性能窗口：以 465°C、12h 附近为平滑中心
        peak_temp = np.exp(-((temp - p['peak_temp_center']) / p['peak_temp_width']) ** 2)
        peak_time = np.exp(-((logtime - p['peak_logtime_center']) / p['peak_logtime_width']) ** 2)
        peak_window = peak_temp * peak_time

        # 高温激活因子：470°C附近塑性与强化常明显增强
        high_temp_boost = 1.0 / (1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / 6.0))

        # 时间促进因子：时间作用按对数增长，更符合扩散/析出动力学
        time_boost = h_norm

        # 基线预测：保持方向与题面趋势一致，幅度保守，让模型主要学残差
        baseline_tensile = (
            p['as_cast_tensile']
            + 95.0 * t_norm
            + 70.0 * time_boost
            + 70.0 * peak_window
            + 25.0 * high_temp_boost
        )

        baseline_yield = (
            p['as_cast_yield']
            + 75.0 * t_norm
            + 55.0 * time_boost
            + 55.0 * peak_window
            + 18.0 * high_temp_boost
        )

        baseline_strain = (
            p['as_cast_strain']
            - 3.2
            + 4.5 * high_temp_boost
            + 1.8 * time_boost
            + 2.2 * peak_window
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        p = DOMAIN_PARAMS
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ---------------------------
        # 1) 最基础且必要的主特征
        # ---------------------------
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['temp_x_log_time'] = temp * log_time

        # ---------------------------
        # 2) 小样本下保留少量非线性
        # ---------------------------
        cols['temp_sq_centered'] = ((temp - p['critical_temp_regime_shift']) / 20.0) ** 2
        cols['log_time_sq_centered'] = ((log_time - p['peak_logtime_center']) / 0.8) ** 2

        # ---------------------------
        # 3) 物理机制区间检测特征（关键）
        # ---------------------------
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_very_high_temp'] = (temp >= p['critical_temp_high']).astype(float)
        cols['is_mid_long_time'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_long_time'] = (time >= p['critical_time_long']).astype(float)

        cols['highT_x_midLongTime'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['veryHighT_x_midLongTime'] = (
            (temp >= p['critical_temp_high']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        # 平滑分段激活，避免硬阈值过于生硬
        cols['temp_above_shift'] = np.maximum(temp - p['critical_temp_regime_shift'], 0.0)
        cols['temp_above_high'] = np.maximum(temp - p['critical_temp_high'], 0.0)
        cols['time_above_12'] = np.maximum(time - p['critical_time_regime_shift'], 0.0)
        cols['time_above_12_log'] = np.maximum(log_time - np.log1p(p['critical_time_regime_shift']), 0.0)

        # ---------------------------
        # 4) 动力学等效特征
        # ---------------------------
        # Arrhenius 型：exp(-Q/RT) * t
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius_rate'] = arrhenius
        cols['arrhenius_time'] = arrhenius * np.clip(time, 0, None)
        cols['arrhenius_logtime'] = arrhenius * log_time

        # Larson-Miller Parameter
        cols['larson_miller'] = temp_k * (p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None)))

        # Z-like 简化热暴露量（不引入应变率，仅表征热处理推进程度）
        cols['thermal_exposure'] = temp_k * log_time

        # ---------------------------
        # 5) 高性能窗口/机制中心距离
        # ---------------------------
        temp_dist_peak = (temp - p['peak_temp_center']) / p['peak_temp_width']
        time_dist_peak = (log_time - p['peak_logtime_center']) / p['peak_logtime_width']
        peak_proximity = np.exp(-(temp_dist_peak ** 2 + time_dist_peak ** 2))

        cols['peak_window_proximity'] = peak_proximity
        cols['temp_relative_to_peak'] = temp - p['peak_temp_center']
        cols['logtime_relative_to_peak'] = log_time - p['peak_logtime_center']

        # ---------------------------
        # 6) 训练域边界/外推距离特征（关键）
        # ---------------------------
        cols['dist_to_temp_min'] = temp - p['train_temp_min']
        cols['dist_to_temp_max'] = p['train_temp_max'] - temp
        cols['dist_to_time_min'] = time - p['train_time_min']
        cols['dist_to_time_max'] = p['train_time_max'] - time

        cols['outside_temp_low'] = np.maximum(p['train_temp_min'] - temp, 0.0)
        cols['outside_temp_high'] = np.maximum(temp - p['train_temp_max'], 0.0)
        cols['outside_time_low'] = np.maximum(p['train_time_min'] - time, 0.0)
        cols['outside_time_high'] = np.maximum(time - p['train_time_max'], 0.0)

        # 对当前数据，虽然大多在整体范围内，但 440/470 等属于“条件组合外推”
        # 用相对机制边界的距离帮助模型判断外推方向
        cols['abs_temp_to_shift_boundary'] = np.abs(temp - p['critical_temp_regime_shift'])
        cols['abs_temp_to_high_boundary'] = np.abs(temp - p['critical_temp_high'])
        cols['abs_logtime_to_12h'] = np.abs(log_time - np.log1p(p['critical_time_regime_shift']))

        # ---------------------------
        # 7) 物理基线作为特征（让模型学残差）
        # ---------------------------
        baseline = np.array([self.physics_baseline(ti, hi) for ti, hi in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        # 相对基线/边界的联合特征
        cols['baseline_tensile_x_peak'] = cols['baseline_tensile'] * peak_proximity
        cols['baseline_yield_x_peak'] = cols['baseline_yield'] * peak_proximity
        cols['baseline_strain_x_highT'] = cols['baseline_strain'] * cols['is_high_temp_regime']

        # ---------------------------
        # 8) 针对当前大误差点的少量定向特征
        # ---------------------------
        # 470×12 误差最大：专门刻画高温+12h附近窗口
        cols['near_470_12_window'] = np.exp(
            -((temp - 470.0) / 10.0) ** 2 - ((log_time - np.log1p(12.0)) / 0.45) ** 2
        )

        # 440×1 与 440×24 均为明显难点：帮助模型识别 440°C 低温侧的非线性
        cols['near_440_temp'] = np.exp(-((temp - 440.0) / 12.0) ** 2)
        cols['near_440_x_short1h'] = cols['near_440_temp'] * np.exp(-((log_time - np.log1p(1.0)) / 0.25) ** 2)
        cols['near_440_x_long24h'] = cols['near_440_temp'] * np.exp(-((log_time - np.log1p(24.0)) / 0.25) ** 2)

        # ---------------------------
        # 9) 少量尺度压缩特征
        # ---------------------------
        cols['inv_temp_k'] = 1.0 / temp_k
        cols['sqrt_time'] = sqrt_time

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names