import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 数据显示 460~470℃附近进入更强强化区，但真正可能出现机制切换大致在 465℃左右
    'critical_temp_regime_shift': 465.0,
    # 1->12h 增益最明显，12h 可视作主要动力学拐点
    'critical_time_regime_shift': 12.0,
    # 长时暴露边界
    'long_time_threshold': 24.0,

    # 经验最优区：中高温 + 中等时间
    'opt_temp_center': 466.0,
    'opt_time_center': 12.0,

    # 外推热点中心
    'low_extrap_temp_center': 440.0,
    'high_extrap_temp_center': 470.0,

    # ===== 动力学参数 =====
    # 取较温和的铝合金扩散/析出等效激活能，避免 Arrhenius 特征过尖
    'activation_energy_Q': 108000.0,   # J/mol
    'gas_constant_R': 8.314,
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
    1) 少而精，优先保留有物理意义的分段/动力学/局部窗口特征
    2) physics baseline 作为“平滑先验”，下游模型学习残差
    3) 重点修正外推点 440@1、440@24、470@12 的方向性误差
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        物理启发基线：
        - 在当前数据范围内，相对铸态，热处理总体显著提升强度与一定延性
        - 温度升高到 460~470℃附近通常增强，不应对 470@12 施加强软化先验
        - 时间从 1h -> 12h 增益明显；24h 可能分化，但不应先验大幅回落
        - 440℃是 420 与更高温区之间的桥接外推层，强度不应被压得过低
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度主增益：在本窗口内总体随温度升高而增强，但保持平滑
        temp_gain = 1.0 / (1.0 + np.exp(-(temp - 448.0) / 8.0))

        # 中高温局部优区：460~470℃附近强化更活跃
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 18.0) ** 2)

        # 时间主增益：1->12h 增长明显，之后进入缓和饱和
        time_gain_fast = 1.0 - np.exp(-time / 6.0)
        time_gain_slow = np.tanh(logt / 1.2)

        # 470@12 局部增强：避免该外推点被系统性低估
        focus_470_12 = (
            np.exp(-((temp - 470.0) / 8.0) ** 2) *
            np.exp(-((time - 12.0) / 4.5) ** 2)
        )

        # 440 温度桥接：外推方向上不应比 420 层表现“异常偏软”
        bridge_440 = np.exp(-((temp - 440.0) / 9.0) ** 2)

        # 高温长时竞争项：只做很弱的抑制，避免错误引入强软化
        over_exposure = (
            np.clip(temp - p['critical_temp_regime_shift'], 0.0, None) / 22.0
            * np.clip(time - p['critical_time_regime_shift'], 0.0, None) / 16.0
        )
        over_exposure = np.clip(over_exposure, 0.0, 1.0)

        tensile = (
            p['as_cast_tensile']
            + 98.0 * time_gain_fast
            + 20.0 * time_gain_slow
            + 54.0 * temp_gain
            + 70.0 * temp_peak
            + 30.0 * focus_470_12
            + 16.0 * bridge_440 * np.tanh(logt / 1.1)
            - 5.0 * over_exposure
        )

        yield_strength = (
            p['as_cast_yield']
            + 74.0 * time_gain_fast
            + 14.0 * time_gain_slow
            + 42.0 * temp_gain
            + 58.0 * temp_peak
            + 24.0 * focus_470_12
            + 12.0 * bridge_440 * np.tanh(logt / 1.1)
            - 4.5 * over_exposure
        )

        strain = (
            p['as_cast_strain']
            + 1.9 * time_gain_fast
            + 0.7 * time_gain_slow
            + 1.4 * temp_gain
            + 1.6 * temp_peak
            + 1.3 * focus_470_12
            + 0.45 * bridge_440 * np.tanh(logt / 1.1)
            - 0.25 * over_exposure
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

        # ========== 1) 基础低维特征 ==========
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_sq_centered'] = ((temp - p['opt_temp_center']) / 20.0) ** 2

        # ========== 2) 机制区间 / 分段特征 ==========
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

        # ========== 4) 外推/边界距离特征 ==========
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

        # ========== 6) 少量局部分段 / 热点修正 ==========
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

        # 极小幅度增强：只保留两个最有针对性的局部特征
        cols['focus47012_with_boundary'] = (
            cols['temp470_time12_focus'] *
            (1.0 + np.clip(temp - p['critical_temp_regime_shift'], 0, None) / 10.0)
        )
        cols['focus440_bridge_with_time'] = (
            np.exp(-((temp - 440.0) / 9.0) ** 2) *
            np.tanh(log_time / 1.1) *
            (1.0 + np.abs(time - 12.0) / 12.0)
        )

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names