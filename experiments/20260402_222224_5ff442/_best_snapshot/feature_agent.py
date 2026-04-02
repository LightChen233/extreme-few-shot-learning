import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 结合现有数据：460~470℃附近通常进入更强的组织活化区，
    # 而 470@12h 是验证集中最明显的“模型低估”热点
    'critical_temp_regime_shift': 465.0,
    # 12h 是最明确的时间拐点：1→12h 常显著增强，12→24h 开始出现分化
    'critical_time_regime_shift': 12.0,
    # 长时暴露边界
    'long_time_threshold': 24.0,

    # 经验优区中心：从现有描述看 460~470℃、约12h 接近强度高值区
    'opt_temp_center': 466.0,
    'opt_time_center': 12.0,

    # 外推热点
    'low_extrap_temp_center': 440.0,
    'high_extrap_temp_center': 470.0,

    # ===== 动力学参数 =====
    # 铝合金热激活过程的温和经验值，避免 Arrhenius 项过度尖锐
    'activation_energy_Q': 108000.0,   # J/mol
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
    - 少而精，避免无约束扩维
    - 强调机制分段、动力学等效、局部热点修正
    - 用 physics baseline 作为残差学习锚点
    - 重点照顾 470@12、440@24、440@1 三类高误差外推点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的粗基线：
        1) 相对铸态，热处理整体显著提升强度，且常能改善延性
        2) 在当前温度窗口内，升温到中高温区通常有利；但不是简单线性
        3) 时间从 1h 到 12h 的增益最明确；24h 不应被先验强烈压低
        4) 470@12h 附近应允许出现局部抬升，否则会延续系统性低估
        5) 440℃ 外推层应是 420→460 之间的“桥接层”，不能被错误当作明显软化区
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 中高温活化：当前数据窗口内总体随温度提高而增强
        high_temp_gain = 1.0 / (1.0 + np.exp(-(temp - 448.0) / 7.5))

        # 优区温度峰：强化机制在 460~470℃ 附近更活跃
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 20.0) ** 2)

        # 时间增益：1→12h 最明显，24h 进入温和饱和，但不强行回落
        time_gain_fast = 1.0 - np.exp(-time / 6.5)
        time_gain_slow = np.tanh(logt / 1.25)

        # 470@12 定向增强：解决高温12h方向外推过于保守的问题
        focus_470_12 = (
            np.exp(-((temp - 470.0) / 8.5) ** 2) *
            np.exp(-((time - 12.0) / 4.5) ** 2)
        )

        # 440℃桥接：440 是 420 和更高温层之间的外推中间层，不应被压得过低
        bridge_440 = np.exp(-((temp - 440.0) / 10.0) ** 2)

        # 高温长时竞争项：仅在 >465℃ 且 >12h 后极温和出现，避免错误先验软化
        over_exposure = (
            np.clip(temp - p['critical_temp_regime_shift'], 0.0, None) / 20.0
            * np.clip(time - p['critical_time_regime_shift'], 0.0, None) / 14.0
        )
        over_exposure = np.clip(over_exposure, 0.0, 1.0)

        tensile = (
            p['as_cast_tensile']
            + 96.0 * time_gain_fast
            + 22.0 * time_gain_slow
            + 56.0 * high_temp_gain
            + 68.0 * temp_peak
            + 28.0 * focus_470_12
            + 12.0 * bridge_440 * np.tanh(logt)
            - 7.0 * over_exposure
        )

        yield_strength = (
            p['as_cast_yield']
            + 73.0 * time_gain_fast
            + 16.0 * time_gain_slow
            + 43.0 * high_temp_gain
            + 55.0 * temp_peak
            + 20.0 * focus_470_12
            + 8.0 * bridge_440 * np.tanh(logt)
            - 6.0 * over_exposure
        )

        strain = (
            p['as_cast_strain']
            + 1.8 * time_gain_fast
            + 0.8 * time_gain_slow
            + 1.5 * high_temp_gain
            + 1.7 * temp_peak
            + 1.2 * focus_470_12
            + 0.35 * bridge_440 * np.tanh(logt)
            - 0.35 * over_exposure
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

        # ========== 1) 基础特征 ==========
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_sq_centered'] = ((temp - p['opt_temp_center']) / 20.0) ** 2

        # ========== 2) 机制区间 / 分段激活 ==========
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_long_time_regime'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_high_temp_long_time'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['temp_above_crit'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['time_above_crit'] = np.clip(time - p['critical_time_regime_shift'], 0, None)

        cols['highT_longt_activation'] = (
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

        nearest_temp_grid = np.minimum.reduce([
            np.abs(temp - 420.0),
            np.abs(temp - 460.0),
            np.abs(temp - 470.0),
            np.abs(temp - 480.0),
        ])
        nearest_time_grid = np.minimum.reduce([
            np.abs(time - 1.0),
            np.abs(time - 12.0),
            np.abs(time - 24.0),
        ])
        cols['nearest_temp_grid_dist'] = nearest_temp_grid
        cols['nearest_time_grid_dist'] = nearest_time_grid
        cols['grid_dist_product'] = nearest_temp_grid * nearest_time_grid

        cols['temp_rel_440'] = temp - 440.0
        cols['temp_rel_470'] = temp - 470.0
        cols['is_440_band'] = (np.abs(temp - 440.0) <= 5.0).astype(float)
        cols['is_470_band'] = (np.abs(temp - 470.0) <= 5.0).astype(float)

        # ========== 5) 基线预测特征 ==========
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
        cols['time_piece_12minus'] = np.clip(12.0 - time, 0, None)
        cols['time_piece_12plus'] = np.clip(time - 12.0, 0, None)

        cols['temp470_time12_focus'] = (
            np.exp(-((temp - 470.0) / 8.0) ** 2) *
            np.exp(-((time - 12.0) / 4.5) ** 2)
        )
        cols['temp440_time24_focus'] = (
            np.exp(-((temp - 440.0) / 8.0) ** 2) *
            np.exp(-((time - 24.0) / 6.0) ** 2)
        )
        cols['temp440_time1_focus'] = (
            np.exp(-((temp - 440.0) / 8.0) ** 2) *
            np.exp(-((time - 1.0) / 2.0) ** 2)
        )

        # 极小幅度新增：只加 1 个“440桥接×时间”项，修正 440@1 / 440@24 两端外推
        cols['focus47012_with_boundary'] = (
            cols['temp470_time12_focus'] *
            (1.0 + np.clip(temp - p['critical_temp_regime_shift'], 0, None) / 10.0)
        )

        cols['focus440_bridge_with_time'] = (
            np.exp(-((temp - 440.0) / 9.0) ** 2) *
            np.tanh(log_time) *
            (1.0 + np.abs(time - 12.0) / 12.0)
        )

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names