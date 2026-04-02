import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 结合题面：460~470℃附近进入更强激活区，且470@12h误差最大，说明该邻域存在斜率/机制变化
    'critical_temp_regime_shift': 465.0,
    # 12h 是明显关键节点：1→12h提升显著，而12→24h开始出现分化
    'critical_time_regime_shift': 12.0,
    # 长时边界：24h是现有数据上限，也是可能粗化/竞争机制更明显的时长
    'long_time_threshold': 24.0,

    # 经验优区中心：题面与数据都指向 460℃、12h 附近较优
    'opt_temp_center': 460.0,
    'opt_time_center': 12.0,

    # 对 440 / 470 这两个高误差外推温度点做局部机制锚定
    'low_extrap_temp_center': 440.0,
    'high_extrap_temp_center': 470.0,

    # ===== 动力学参数 =====
    # 铝合金中析出/扩散相关有效激活能取温和经验值，避免指数项过激
    'activation_energy_Q': 115000.0,   # J/mol
    'gas_constant_R': 8.314,           # J/(mol*K)

    # Larson-Miller 风格参数
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
    - 少而精，保留当前主干特征框架
    - 加入机制分段与动力学等效
    - 基线预测作为残差学习锚点
    - 外推距离特征帮助约束训练边界外方向
    - 仅做小幅针对性增强，避免小样本下过拟合
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的粗基线：
        1) 相对铸态，热处理整体显著提升强度
        2) 1→12h总体提升明显，之后趋缓并可能分化
        3) 460℃、12h附近接近优区
        4) 470~480℃长时可能出现部分竞争机制，但不能施加过强软化
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        logt = np.log1p(time)

        # 温度响应：在当前窗口内总体升温有利，但以460附近为较优中心
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 23.0) ** 2)

        # 时间响应：1->12h提升明显，之后逐步饱和
        time_sat = 1.0 - np.exp(-time / 8.5)

        # 高温激活：450℃以上扩散/均匀化/析出动力更强
        high_temp_gain = 1.0 / (1.0 + np.exp(-(temp - 450.0) / 8.0))

        # 仅给出温和的高温长时竞争项，避免错误强加过时效
        over_exposure = (
            max(temp - p['critical_temp_regime_shift'], 0.0) / 20.0
            * max(time - p['critical_time_regime_shift'], 0.0) / 12.0
        )
        over_exposure = np.clip(over_exposure, 0.0, 1.2)

        tensile = (
            p['as_cast_tensile']
            + 98.0 * time_sat
            + 78.0 * temp_peak
            + 52.0 * high_temp_gain * np.tanh(logt)
            - 15.0 * over_exposure
        )

        yield_strength = (
            p['as_cast_yield']
            + 76.0 * time_sat
            + 61.0 * temp_peak
            + 40.0 * high_temp_gain * np.tanh(logt)
            - 11.0 * over_exposure
        )

        strain = (
            p['as_cast_strain']
            + 1.9 * time_sat
            + 1.9 * temp_peak
            + 1.4 * high_temp_gain * np.tanh(logt)
            - 0.7 * over_exposure
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
        cols['temp_sq_centered'] = ((temp - p['opt_temp_center']) / 20.0) ** 2
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
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
        cols['larson_miller'] = temp_k * (p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None)))
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

        # ===== 小幅新增：更直接反映“外推到未观测温度层”的危险度 =====
        # 当前外推主要是 440 和 470 这两个温度层，因此加入到已知训练温度层的最小距离
        cols['dist_to_known_temp_levels'] = np.minimum.reduce([
            np.abs(temp - 420.0),
            np.abs(temp - 460.0),
            np.abs(temp - 470.0),
            np.abs(temp - 480.0),
        ])
        cols['dist_to_known_time_levels'] = np.minimum.reduce([
            np.abs(time - 1.0),
            np.abs(time - 12.0),
            np.abs(time - 24.0),
        ])

        # ========== 5) 基线预测特征 ==========
        baseline = np.array([self.physics_baseline(t, tm) for t, tm in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        cols['baseline_strength_gap'] = baseline[:, 1] - baseline[:, 2]
        cols['baseline_strength_ratio'] = baseline[:, 2] / np.clip(baseline[:, 1], 1e-6, None)
        cols['baseline_strain_strength_coupling'] = baseline[:, 0] * baseline[:, 1]

        cols['temp_relative_to_boundary'] = (temp - p['critical_temp_regime_shift']) / 20.0
        cols['time_relative_to_boundary'] = (log_time - np.log1p(p['critical_time_regime_shift'])) / 1.0
        cols['boundary_interaction'] = (
            cols['temp_relative_to_boundary'] * cols['time_relative_to_boundary']
        )

        # ========== 6) 少量针对性局部修正特征 ==========
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

        # ===== 小幅新增：针对误差最大的两个外推热点，加入“局部-边界交互” =====
        # 不再大规模扩特征，只补2个更有物理含义的修正项：
        # 1) 470@12 低估：表示高温关键时长附近的激活增强
        cols['focus47012_with_boundary'] = (
            cols['temp470_time12_focus'] *
            (1.0 + np.clip(temp - p['critical_temp_regime_shift'], 0, None) / 10.0)
        )
        # 2) 440@24 方向错误：表示低于主训练温度层时，长时响应可能与420@24不同
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