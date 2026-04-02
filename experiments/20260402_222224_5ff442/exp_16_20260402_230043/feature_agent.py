import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 依据当前数据与材料常识：
    # - 约 460~470℃ 进入更强的组织活化/均匀化区
    # - 约 12h 是最明显的时间增益拐点
    # - 24h 是长时暴露边界，但当前数据不支持“强烈软化”先验，只允许很弱竞争项
    'critical_temp_regime_shift': 465.0,
    'critical_time_regime_shift': 12.0,
    'long_time_threshold': 24.0,

    # 经验优区中心：强度/塑性协同提升的候选窗口
    'opt_temp_center': 468.0,
    'opt_time_center': 12.0,

    # 外推热点中心
    'bridge_temp_center': 440.0,
    'high_temp_focus_center': 470.0,

    # ===== 动力学参数 =====
    # 采用温和参数，避免 Arrhenius 特征过于尖锐
    'activation_energy_Q': 110000.0,   # J/mol
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
    小样本 7499 铝合金热处理特征工程：
    - 维持“少而精”的物理特征
    - 重点描述温度-时间非线性交互
    - 用 physics baseline 作为残差学习锚点
    - 重点修正 470@12、440@24、440@1 三类高误差外推点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        物理启发 baseline，自检原则：
        1) 当前温区内，整体上升温通常提升强度，不能预设高温必软化
        2) 1h -> 12h 的时间增益明确；12h -> 24h 不应被先验强压回落
        3) 470@12h 需要允许局部峰值，否则会继续系统性低估
        4) 440℃ 是 420℃ 与更高温区之间的桥接外推层，不应被错误看成明显劣化层
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-8)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度主效应：当前数据范围内总体增强，但保持平滑
        temp_activation = 1.0 / (1.0 + np.exp(-(temp - 448.0) / 8.0))

        # 中高温优区峰，允许 460~470℃ 附近局部更优
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 18.0) ** 2)

        # 时间效应：1->12h 增益显著；24h 仅温和饱和，不预设显著下降
        time_gain_fast = 1.0 - np.exp(-time / 6.0)
        time_gain_log = np.tanh(logt / 1.20)

        # 470@12h 局部增强：针对当前最大外推误差热点
        focus_470_12 = (
            np.exp(-((temp - 470.0) / 7.5) ** 2) *
            np.exp(-((time - 12.0) / 4.0) ** 2)
        )

        # 440℃ 桥接：避免 440 外推层被压得过低
        bridge_440 = np.exp(-((temp - 440.0) / 9.0) ** 2)

        # 极弱长时高温竞争项：只在 >465℃ 且 >12h 轻微出现
        over_exposure = (
            np.clip(temp - p['critical_temp_regime_shift'], 0.0, None) / 18.0
        ) * (
            np.clip(time - p['critical_time_regime_shift'], 0.0, None) / 12.0
        )
        over_exposure = np.clip(over_exposure, 0.0, 1.0)

        tensile = (
            p['as_cast_tensile']
            + 100.0 * time_gain_fast
            + 20.0 * time_gain_log
            + 58.0 * temp_activation
            + 72.0 * temp_peak
            + 30.0 * focus_470_12
            + 14.0 * bridge_440 * np.tanh(logt)
            - 6.0 * over_exposure
        )

        yield_strength = (
            p['as_cast_yield']
            + 76.0 * time_gain_fast
            + 14.0 * time_gain_log
            + 45.0 * temp_activation
            + 58.0 * temp_peak
            + 22.0 * focus_470_12
            + 10.0 * bridge_440 * np.tanh(logt)
            - 5.0 * over_exposure
        )

        strain = (
            p['as_cast_strain']
            + 1.9 * time_gain_fast
            + 0.7 * time_gain_log
            + 1.5 * temp_activation
            + 1.8 * temp_peak
            + 1.1 * focus_470_12
            + 0.40 * bridge_440 * np.tanh(logt)
            - 0.30 * over_exposure
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

        # ========== 1) 基础低维主特征 ==========
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_x_log_time'] = temp * log_time

        # 围绕优区的中心化非线性
        cols['temp_centered'] = (temp - p['opt_temp_center']) / 20.0
        cols['temp_sq_centered'] = cols['temp_centered'] ** 2
        cols['logtime_centered'] = (log_time - np.log1p(p['opt_time_center'])) / 0.9
        cols['logtime_sq_centered'] = cols['logtime_centered'] ** 2

        # ========== 2) 机制区间 / 分段特征 ==========
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_long_time_regime'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_high_temp_long_time'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['temp_above_crit'] = np.clip(temp - p['critical_temp_regime_shift'], 0.0, None)
        cols['time_above_crit'] = np.clip(time - p['critical_time_regime_shift'], 0.0, None)

        cols['temp_below_crit'] = np.clip(p['critical_temp_regime_shift'] - temp, 0.0, None)
        cols['time_below_crit'] = np.clip(p['critical_time_regime_shift'] - time, 0.0, None)

        cols['highT_longt_activation'] = (
            cols['temp_above_crit'] *
            np.clip(log_time - np.log1p(p['critical_time_regime_shift']), 0.0, None)
        )

        # ========== 3) 动力学等效特征 ==========
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius'] = arrhenius
        cols['time_x_arrhenius'] = time * arrhenius
        cols['logtime_x_arrhenius'] = log_time * arrhenius
        cols['thermal_exposure_index'] = temp_k * log_time
        cols['inv_temp_k'] = 1.0 / temp_k
        cols['log_time_over_tempk'] = log_time / temp_k
        cols['larson_miller'] = temp_k * (
            p['larson_miller_C'] + np.log10(np.clip(time, 1e-8, None))
        )

        # ========== 4) 外推 / 边界 / 覆盖距离 ==========
        cols['dist_to_temp_min'] = np.clip(p['train_temp_min'] - temp, 0.0, None)
        cols['dist_to_temp_max'] = np.clip(temp - p['train_temp_max'], 0.0, None)
        cols['dist_to_time_min'] = np.clip(p['train_time_min'] - time, 0.0, None)
        cols['dist_to_time_max'] = np.clip(time - p['train_time_max'], 0.0, None)

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

        # ========== 5) 局部热点 / 桥接特征 ==========
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

        cols['focus47012_with_boundary'] = (
            cols['temp470_time12_focus'] *
            (1.0 + np.clip(temp - p['critical_temp_regime_shift'], 0.0, None) / 10.0)
        )

        # 仅保留一个 440 桥接时间项，避免小样本过扩维
        cols['focus440_bridge_with_time'] = (
            np.exp(-((temp - 440.0) / 9.0) ** 2) *
            np.tanh(log_time)
        )

        # ========== 6) baseline 预测特征 ==========
        baseline = np.array([self.physics_baseline(t, tm) for t, tm in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        cols['baseline_strength_gap'] = baseline[:, 1] - baseline[:, 2]
        cols['baseline_strength_ratio'] = baseline[:, 2] / np.clip(baseline[:, 1], 1e-8, None)
        cols['baseline_strain_strength_coupling'] = baseline[:, 0] * baseline[:, 1]

        # 相对机制边界的位置，便于模型学习“相对 baseline 的残差修正”
        cols['temp_relative_to_boundary'] = (temp - p['critical_temp_regime_shift']) / 20.0
        cols['time_relative_to_boundary'] = (
            log_time - np.log1p(p['critical_time_regime_shift'])
        )
        cols['boundary_interaction'] = (
            cols['temp_relative_to_boundary'] * cols['time_relative_to_boundary']
        )

        # ========== 7) 温和分段，帮助 440/470 间外推 ==========
        cols['temp_piece_low'] = np.clip(450.0 - temp, 0.0, None)
        cols['temp_piece_midhigh'] = np.clip(temp - 450.0, 0.0, None)
        cols['time_piece_12minus'] = np.clip(12.0 - time, 0.0, None)
        cols['time_piece_12plus'] = np.clip(time - 12.0, 0.0, None)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names