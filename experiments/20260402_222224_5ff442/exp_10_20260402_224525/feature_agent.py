import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 数据显示 460~470℃附近进入更强强化区，470@12h 外推误差最大，说明此处存在局部斜率变化
    'critical_temp_regime_shift': 465.0,
    # 12h 是关键时间节点：1->12h 通常提升明显，12->24h 进入另一响应区
    'critical_time_regime_shift': 12.0,
    # 长时边界
    'long_time_threshold': 24.0,

    # 经验优区中心
    'opt_temp_center': 462.0,
    'opt_time_center': 12.0,

    # 局部高误差热点
    'low_extrap_temp_center': 440.0,
    'high_extrap_temp_center': 470.0,

    # ===== 动力学参数 =====
    # 7499铝合金热处理相关扩散/析出有效激活能，取温和经验值
    'activation_energy_Q': 115000.0,   # J/mol
    'gas_constant_R': 8.314,           # J/(mol*K)
    'larson_miller_C': 20.0,

    # ===== 训练覆盖边界 =====
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # ===== 原始铸态基准 =====
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    小样本材料热处理特征工程：
    - 少而精，优先保留稳定主干
    - 强调机制分段、动力学等效、外推边界感知
    - 通过 physics baseline 让模型做残差修正
    - 仅做小幅针对性增强，避免小样本下过拟合
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于物理先验的粗基线：
        1) 当前数据范围内，热处理相对铸态整体提升强度与延性
        2) 时间效应不能用饱和函数压平；采用 log(time) + 线性分段增益，保持长时仍可增长
        3) 460~470℃附近存在强化增强区
        4) 24h 不强行假设软化，只给极温长时一个非常温和的竞争项
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度主效应：在当前窗口内总体升温有利，但保持局部优区
        temp_gain_linear = 1.55 * (temp - 420.0)
        temp_gain_peak = 62.0 * np.exp(-((temp - p['opt_temp_center']) / 20.0) ** 2)

        # 时间主效应：禁止饱和型，采用对数 + 长时线性项
        time_gain_log = 33.0 * logt
        time_gain_long = 1.8 * max(time - p['critical_time_regime_shift'], 0.0)

        # 高温激活区：465℃以上强化动力更强，且在12h附近更显著
        high_temp_activation = max(temp - p['critical_temp_regime_shift'], 0.0)
        mid_time_boost = np.exp(-((time - 12.0) / 6.0) ** 2)

        # 极高温长时仅施加极弱竞争项，避免错误压低24h
        mild_over = (
            max(temp - 475.0, 0.0) / 10.0
        ) * (
            max(time - 18.0, 0.0) / 12.0
        )

        tensile = (
            p['as_cast_tensile']
            + temp_gain_linear
            + temp_gain_peak
            + time_gain_log
            + time_gain_long
            + 1.8 * high_temp_activation * mid_time_boost
            - 4.0 * mild_over
        )

        yield_strength = (
            p['as_cast_yield']
            + 1.18 * (temp - 420.0)
            + 47.0 * np.exp(-((temp - p['opt_temp_center']) / 21.0) ** 2)
            + 25.0 * logt
            + 1.2 * max(time - p['critical_time_regime_shift'], 0.0)
            + 1.15 * high_temp_activation * mid_time_boost
            - 3.0 * mild_over
        )

        strain = (
            p['as_cast_strain']
            + 0.050 * (temp - 420.0)
            + 1.45 * np.exp(-((temp - 468.0) / 18.0) ** 2)
            + 1.10 * logt
            + 0.050 * max(time - p['critical_time_regime_shift'], 0.0)
            + 0.030 * high_temp_activation * mid_time_boost
            - 0.18 * mild_over
        )

        return float(strain), float(tensile), float(yield_strength)

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        p = DOMAIN_PARAMS
        temp_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ========== 1) 基础主干特征 ==========
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_sq_centered'] = ((temp - p['opt_temp_center']) / 20.0) ** 2
        cols['temp_x_log_time'] = temp * log_time

        # ========== 2) 机制区间 / 分段激活 ==========
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_long_time_regime'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_high_temp_long_time'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['temp_above_crit'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['time_above_crit'] = np.clip(time - p['critical_time_regime_shift'], 0, None)

        cols['highT_logt_activation'] = (
            np.clip(temp - p['critical_temp_regime_shift'], 0, None) *
            np.clip(log_time - np.log1p(p['critical_time_regime_shift']), 0, None)
        )

        cols['dist_temp_to_opt'] = np.abs(temp - p['opt_temp_center'])
        cols['dist_logtime_to_opt'] = np.abs(log_time - np.log1p(p['opt_time_center']))
        cols['elliptic_dist_to_opt'] = np.sqrt(
            ((temp - p['opt_temp_center']) / 20.0) ** 2 +
            ((log_time - np.log1p(p['opt_time_center'])) / 0.9) ** 2
        )

        # ========== 3) 动力学等效特征 ==========
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius'] = arrhenius
        cols['time_x_arrhenius'] = time * arrhenius
        cols['logtime_x_arrhenius'] = log_time * arrhenius

        cols['thermal_exposure_index'] = temp_k * log_time
        cols['inv_temp_k'] = 1.0 / temp_k
        cols['larson_miller'] = temp_k * (
            p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None))
        )
        cols['log_time_over_tempk'] = log_time / temp_k

        # ========== 4) 外推 / 边界距离特征 ==========
        cols['dist_to_temp_min'] = np.clip(p['train_temp_min'] - temp, 0, None)
        cols['dist_to_temp_max'] = np.clip(temp - p['train_temp_max'], 0, None)
        cols['dist_to_time_min'] = np.clip(p['train_time_min'] - time, 0, None)
        cols['dist_to_time_max'] = np.clip(time - p['train_time_max'], 0, None)

        known_temp_levels = [420.0, 460.0, 470.0, 480.0]
        known_time_levels = [1.0, 12.0, 24.0]

        nearest_temp_grid = np.minimum.reduce([np.abs(temp - x) for x in known_temp_levels])
        nearest_time_grid = np.minimum.reduce([np.abs(time - x) for x in known_time_levels])

        cols['nearest_temp_grid_dist'] = nearest_temp_grid
        cols['nearest_time_grid_dist'] = nearest_time_grid
        cols['grid_dist_product'] = nearest_temp_grid * nearest_time_grid

        cols['dist_to_known_temp_levels'] = nearest_temp_grid
        cols['dist_to_known_time_levels'] = nearest_time_grid

        # 对“未观测温度层”的风险更敏感：440 和 470/480之间的温度带
        cols['temp_rel_440'] = temp - 440.0
        cols['temp_rel_470'] = temp - 470.0
        cols['is_440_band'] = (np.abs(temp - 440.0) <= 5.0).astype(float)
        cols['is_470_band'] = (np.abs(temp - 470.0) <= 5.0).astype(float)

        # ========== 5) baseline 预测特征 ==========
        baseline = np.array([self.physics_baseline(t, tm) for t, tm in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        cols['baseline_strength_gap'] = baseline[:, 1] - baseline[:, 2]
        cols['baseline_strength_ratio'] = baseline[:, 2] / np.clip(baseline[:, 1], 1e-6, None)
        cols['baseline_strain_strength_coupling'] = baseline[:, 0] * baseline[:, 1]

        cols['temp_relative_to_boundary'] = (temp - p['critical_temp_regime_shift']) / 20.0
        cols['time_relative_to_boundary'] = (
            log_time - np.log1p(p['critical_time_regime_shift'])
        )
        cols['boundary_interaction'] = (
            cols['temp_relative_to_boundary'] * cols['time_relative_to_boundary']
        )

        # ========== 6) 少量局部修正特征 ==========
        cols['temp_piece_low'] = np.clip(450.0 - temp, 0, None)
        cols['temp_piece_midhigh'] = np.clip(temp - 450.0, 0, None)
        cols['temp_piece_470plus'] = np.clip(temp - 470.0, 0, None)

        cols['time_piece_12minus'] = np.clip(12.0 - time, 0, None)
        cols['time_piece_12plus'] = np.clip(time - 12.0, 0, None)

        cols['temp470_time12_focus'] = (
            np.exp(-((temp - 470.0) / 8.0) ** 2) *
            np.exp(-((time - 12.0) / 4.0) ** 2)
        )
        cols['temp440_time24_focus'] = (
            np.exp(-((temp - 440.0) / 8.0) ** 2) *
            np.exp(-((time - 24.0) / 6.0) ** 2)
        )
        cols['temp440_time1_focus'] = (
            np.exp(-((temp - 440.0) / 8.0) ** 2) *
            np.exp(-((time - 1.0) / 2.0) ** 2)
        )

        # 针对最大误差热点做极小幅增强
        # 1) 470@12：高温关键时长下的局部增强，缓解系统性低估
        cols['focus47012_with_boundary'] = (
            cols['temp470_time12_focus'] *
            (1.0 + np.clip(temp - p['critical_temp_regime_shift'], 0, None) / 10.0)
        )

        # 2) 440@24：低于主强化温区时，长时行为可能偏离420@24到460@24的简单外推
        cols['focus44024_with_time'] = (
            cols['temp440_time24_focus'] *
            (1.0 + np.clip(time - p['critical_time_regime_shift'], 0, None) / 12.0)
        )

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names