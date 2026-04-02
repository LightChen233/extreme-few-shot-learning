import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 从现有描述看：460~470℃附近进入更强组织响应区，
    # 但高温长时并非普遍软化，因此只设置“机制转折”而非强退化阈值
    'critical_temp_regime_shift': 465.0,
    'critical_time_regime_shift': 12.0,
    'long_time_threshold': 24.0,

    # 经验最优区：中高温 + 中等时间
    'opt_temp_center': 466.0,
    'opt_time_center': 12.0,

    # 外推热点中心
    'low_extrap_temp_center': 440.0,
    'high_extrap_temp_center': 470.0,

    # ===== 动力学参数 =====
    # 取温和经验值，避免 Arrhenius 项过于尖锐
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
    小样本材料热处理特征工程：
    - 少而精，避免过度扩维
    - 用分段机制 + 动力学等效 + baseline 残差学习
    - 重点修正外推点：440@1, 440@24, 470@12
    - 保持对屈服强度的稳定性，避免只优化应变/抗拉
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        物理启发基线，自检原则：
        1) 当前数据窗口内，热处理相对铸态整体提升强度与延性
        2) 420→440 的外推不应被错误看成明显软化，因此 440 应位于中间桥接层
        3) 1→12h 的时间增益最明确；到 24h 只允许温和分化，不能先验强烈下压
        4) 470@12h 是已知低估热点，因此需允许局部峰值抬升
        5) 屈服与抗拉共享主强化趋势，但屈服对局部机制更敏感，故保留独立幅值
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度主增益：在当前窗口内总体随升温增强
        temp_gain = 1.0 / (1.0 + np.exp(-(temp - 448.0) / 8.0))

        # 中高温优区峰值：反映 460~470℃附近更活跃的组织强化
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 18.0) ** 2)

        # 时间增益：1→12h 增长快，12→24h 转为缓增/饱和，但不强制衰减
        time_gain_fast = 1.0 - np.exp(-time / 6.0)
        time_gain_slow = np.tanh(logt / 1.20)

        # 470@12 局部抬升：修正高温12h方向外推过保守
        focus_470_12 = (
            np.exp(-((temp - 470.0) / 8.0) ** 2) *
            np.exp(-((time - 12.0) / 4.5) ** 2)
        )

        # 440 桥接：440 是 420 与更高温区之间的中间层，不应被压太低
        bridge_440 = np.exp(-((temp - 440.0) / 9.0) ** 2)

        # 高温长时竞争项：只做极弱约束，避免错误引入系统性低估
        over_exposure = (
            np.clip(temp - p['critical_temp_regime_shift'], 0.0, None) / 22.0 *
            np.clip(time - p['critical_time_regime_shift'], 0.0, None) / 16.0
        )
        over_exposure = np.clip(over_exposure, 0.0, 1.0)

        tensile = (
            p['as_cast_tensile']
            + 98.0 * time_gain_fast
            + 20.0 * time_gain_slow
            + 58.0 * temp_gain
            + 62.0 * temp_peak
            + 26.0 * focus_470_12
            + 10.0 * bridge_440 * np.tanh(logt)
            - 5.0 * over_exposure
        )

        yield_strength = (
            p['as_cast_yield']
            + 74.0 * time_gain_fast
            + 15.0 * time_gain_slow
            + 45.0 * temp_gain
            + 52.0 * temp_peak
            + 18.0 * focus_470_12
            + 7.0 * bridge_440 * np.tanh(logt)
            - 5.5 * over_exposure
        )

        strain = (
            p['as_cast_strain']
            + 1.9 * time_gain_fast
            + 0.7 * time_gain_slow
            + 1.4 * temp_gain
            + 1.5 * temp_peak
            + 1.0 * focus_470_12
            + 0.30 * bridge_440 * np.tanh(logt)
            - 0.30 * over_exposure
        )

        return float(strain), float(tensile), float(yield_strength)

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        p = DOMAIN_PARAMS

        temp_k = temp + 273.15
        time_clip = np.clip(time, 1e-6, None)
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ========= 1) 基础少量主特征 =========
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_sq_centered'] = ((temp - p['opt_temp_center']) / 20.0) ** 2

        # ========= 2) 物理机制区间检测 =========
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

        # ========= 3) 动力学等效特征 =========
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius'] = arrhenius
        cols['time_x_arrhenius'] = time * arrhenius
        cols['logtime_x_arrhenius'] = log_time * arrhenius
        cols['thermal_exposure_index'] = temp_k * log_time
        cols['inv_temp_k'] = 1.0 / temp_k
        cols['larson_miller'] = temp_k * (
            p['larson_miller_C'] + np.log10(time_clip)
        )
        cols['log_time_over_tempk'] = log_time / temp_k

        # ========= 4) 外推/边界距离 =========
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

        # ========= 5) 外推热点局部特征 =========
        cols['temp_rel_440'] = temp - 440.0
        cols['temp_rel_470'] = temp - 470.0
        cols['is_440_band'] = (np.abs(temp - 440.0) <= 5.0).astype(float)
        cols['is_470_band'] = (np.abs(temp - 470.0) <= 5.0).astype(float)

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

        # 仅保留一个桥接增强项，避免过度扩维
        cols['focus440_bridge_with_time'] = (
            np.exp(-((temp - 440.0) / 9.0) ** 2) *
            np.tanh(log_time) *
            (1.0 + np.abs(time - 12.0) / 12.0)
        )

        # ========= 6) baseline 预测及其派生特征 =========
        baseline = np.array([self.physics_baseline(t, tm) for t, tm in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        cols['baseline_strength_gap'] = baseline[:, 1] - baseline[:, 2]
        cols['baseline_strength_ratio'] = baseline[:, 2] / np.clip(baseline[:, 1], 1e-6, None)

        # 保留一个强塑耦合项，但不再堆太多 coupling 特征
        cols['baseline_strain_strength_coupling'] = baseline[:, 0] * baseline[:, 2]

        # 相对机制边界位置：利于学习“相对 baseline 的残差修正”
        cols['temp_relative_to_boundary'] = (temp - p['critical_temp_regime_shift']) / 20.0
        cols['time_relative_to_boundary'] = log_time - np.log1p(p['critical_time_regime_shift'])
        cols['boundary_interaction'] = (
            cols['temp_relative_to_boundary'] * cols['time_relative_to_boundary']
        )

        # ========= 7) 少量分段特征 =========
        cols['temp_piece_low'] = np.clip(450.0 - temp, 0, None)
        cols['temp_piece_midhigh'] = np.clip(temp - 450.0, 0, None)
        cols['time_piece_12minus'] = np.clip(12.0 - time, 0, None)
        cols['time_piece_12plus'] = np.clip(time - 12.0, 0, None)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names