import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # 机制边界：结合题目给出的趋势，460℃附近常表现优异；470℃以上更容易进入高温机制区
    'critical_temp_regime_shift': 465.0,
    'high_temp_threshold': 470.0,
    'low_temp_threshold': 440.0,

    # 时间机制边界：1h 近似短时，12h 常接近强化优区，24h 可能进入长时粗化/再分布区
    'critical_time_regime_shift': 12.0,
    'short_time_threshold': 3.0,
    'long_time_threshold': 18.0,

    # 强化窗口中心：用于构造“距最优窗口距离”特征
    'optimal_temp_center': 462.0,
    'optimal_time_center': 12.0,

    # 动力学参数（铝合金扩散/析出过程量级估计）
    'activation_energy_Q': 135000.0,   # J/mol
    'gas_constant_R': 8.314,
    'lmp_C': 20.0,

    # 原始铸态基线
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 温度尺度，用于边界/外推距离归一化
    'temp_scale': 20.0,
    'time_scale_log': np.log1p(24.0),

    # 已知训练覆盖边界（由题目可见训练温度主要覆盖 420/460/470/480，时间覆盖 1/12/24）
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,
}


class FeatureAgent:
    """
    小样本材料热处理特征工程：
    1) 少而精，突出物理先验
    2) 显式加入机制区间/边界特征
    3) 使用 physics_baseline 作为残差学习锚点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于世界知识的粗基线：
        - 相对铸态，热处理后整体强度显著升高
        - 在当前数据范围内，中高温 + 适中时间（约460℃, 12h）接近强化优区
        - 470-480℃长时可能出现部分非单调，尤其塑性更敏感
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度贡献：从420到460附近快速增强，之后趋缓
        temp_gain = 1.0 / (1.0 + np.exp(-(temp - 445.0) / 10.0))

        # 时间贡献：前期增益明显，随后趋于饱和
        time_gain = 1.0 - np.exp(-time / 8.0)

        # 优化窗口：460-465℃, ~12h附近最优
        temp_opt_penalty = ((temp - p['optimal_temp_center']) / 18.0) ** 2
        time_opt_penalty = ((np.log1p(time) - np.log1p(p['optimal_time_center'])) / 0.85) ** 2
        window_factor = np.exp(-0.5 * (temp_opt_penalty + time_opt_penalty))

        # 高温长时区域的温和惩罚，反映可能的粗化/机制切换
        high_temp_long_time = max(0.0, (temp - p['high_temp_threshold']) / 12.0) * max(0.0, np.log1p(time / 12.0))
        over_process = min(high_temp_long_time, 1.5)

        # 强度：从铸态上升到热处理强化水平
        base_tensile = (
            p['as_cast_tensile']
            + 120.0 * temp_gain
            + 85.0 * time_gain
            + 70.0 * window_factor
            - 20.0 * over_process
        )

        base_yield = (
            p['as_cast_yield']
            + 95.0 * temp_gain
            + 70.0 * time_gain
            + 55.0 * window_factor
            - 16.0 * over_process
        )

        # 应变：整体可改善，但在高温长时下更容易回落；最佳窗口有协同提升
        base_strain = (
            p['as_cast_strain']
            + 1.8 * temp_gain
            + 2.0 * time_gain
            + 2.2 * window_factor
            - 1.2 * over_process
        )

        return base_strain, base_tensile, base_yield

    def engineer_features(self, df):
        p = DOMAIN_PARAMS
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        time_safe = np.clip(time, 1e-6, None)
        temp_k = temp + 273.15
        logt = np.log1p(time_safe)

        # 基础主效应：控制数量，避免过拟合
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_centered'] = temp - p['optimal_temp_center']
        cols['log_time'] = logt
        cols['inv_temp_k'] = 1.0 / temp_k

        # 基础非线性
        cols['temp_centered_sq'] = cols['temp_centered'] ** 2
        cols['log_time_sq'] = logt ** 2

        # 核心交互：温度-时间耦合
        cols['temp_x_log_time'] = temp * logt
        cols['temp_centered_x_log_time'] = cols['temp_centered'] * logt

        # 动力学等效特征：Arrhenius / Larson-Miller / 热暴露指数
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius'] = arrhenius
        cols['time_x_arrhenius'] = time_safe * arrhenius
        cols['log_time_over_temp'] = logt / temp_k
        cols['lmp'] = temp_k * (p['lmp_C'] + np.log10(np.clip(time_safe, 1e-6, None)))
        cols['thermal_exposure'] = temp_k * logt

        # 机制区间检测特征
        cols['is_high_temp'] = (temp >= p['high_temp_threshold']).astype(float)
        cols['is_low_temp'] = (temp <= p['low_temp_threshold']).astype(float)
        cols['is_long_time'] = (time >= p['long_time_threshold']).astype(float)
        cols['is_short_time'] = (time <= p['short_time_threshold']).astype(float)
        cols['is_mid_time_window'] = ((time >= 8.0) & (time <= 16.0)).astype(float)

        cols['is_regime_shift_temp'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_regime_shift_time'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_high_temp_long_time'] = ((temp >= p['high_temp_threshold']) & (time >= p['critical_time_regime_shift'])).astype(float)
        cols['is_opt_window'] = ((temp >= 455.0) & (temp <= 468.0) & (time >= 8.0) & (time <= 16.0)).astype(float)

        # 分段激活：帮助模型表达“跨过边界后变化更快”
        cols['temp_above_shift'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_below_shift'] = np.clip(p['critical_temp_regime_shift'] - temp, 0, None)
        cols['log_time_above_shift'] = np.clip(logt - np.log1p(p['critical_time_regime_shift']), 0, None)
        cols['log_time_below_shift'] = np.clip(np.log1p(p['critical_time_regime_shift']) - logt, 0, None)

        # 距最优工艺窗口距离：针对 470,12 / 440,24 这类“离优区不远但机制不同”的点
        cols['dist_to_opt_temp'] = np.abs(temp - p['optimal_temp_center']) / p['temp_scale']
        cols['dist_to_opt_log_time'] = np.abs(logt - np.log1p(p['optimal_time_center'])) / np.log1p(p['optimal_time_center'])
        cols['dist_to_opt_window'] = np.sqrt(cols['dist_to_opt_temp'] ** 2 + cols['dist_to_opt_log_time'] ** 2)

        # 外推/边界距离特征：重点服务验证/测试外推点
        temp_below = np.clip(p['train_temp_min'] - temp, 0, None)
        temp_above = np.clip(temp - p['train_temp_max'], 0, None)
        time_below = np.clip(p['train_time_min'] - time, 0, None)
        time_above = np.clip(time - p['train_time_max'], 0, None)

        cols['temp_extrapolation_distance'] = (temp_below + temp_above) / p['temp_scale']
        cols['time_extrapolation_distance'] = np.log1p(time_below + time_above) / p['time_scale_log']
        cols['is_temp_extrapolation'] = ((temp < p['train_temp_min']) | (temp > p['train_temp_max'])).astype(float)
        cols['is_time_extrapolation'] = ((time < p['train_time_min']) | (time > p['train_time_max'])).astype(float)

        # 即使在范围内，也可能落在稀疏边界附近；加入“靠近边界”特征
        cols['temp_edge_proximity'] = np.minimum(np.abs(temp - p['train_temp_min']), np.abs(temp - p['train_temp_max'])) / p['temp_scale']
        cols['time_edge_proximity'] = np.minimum(np.abs(np.log1p(time) - np.log1p(p['train_time_min'])),
                                                 np.abs(np.log1p(time) - np.log1p(p['train_time_max']))) / p['time_scale_log']

        # 物理基线特征：让模型学习残差
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []
        for t, ti in zip(temp, time):
            s, uts, ys = self.physics_baseline(t, ti)
            baseline_strain.append(s)
            baseline_tensile.append(uts)
            baseline_yield.append(ys)

        baseline_strain = np.array(baseline_strain)
        baseline_tensile = np.array(baseline_tensile)
        baseline_yield = np.array(baseline_yield)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 相对铸态增量基线
        cols['baseline_strain_gain_over_cast'] = baseline_strain - p['as_cast_strain']
        cols['baseline_tensile_gain_over_cast'] = baseline_tensile - p['as_cast_tensile']
        cols['baseline_yield_gain_over_cast'] = baseline_yield - p['as_cast_yield']

        # 基线相对机制边界的调制项
        cols['baseline_tensile_x_high_temp'] = baseline_tensile * cols['is_high_temp']
        cols['baseline_yield_x_high_temp_long_time'] = baseline_yield * cols['is_high_temp_long_time']
        cols['baseline_strain_x_opt_window'] = baseline_strain * cols['is_opt_window']

        # 反映强化-塑性协同/竞争的无标签组合特征
        cols['baseline_strength_sum'] = baseline_tensile + baseline_yield
        cols['baseline_strength_ratio'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)
        cols['baseline_ductility_strength_tradeoff'] = baseline_strain / np.clip(baseline_tensile, 1e-6, None)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names