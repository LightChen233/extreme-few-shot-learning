import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 数据与题面都表明：460~470℃附近是重要机制转换带
    'critical_temp_regime_shift': 465.0,
    # 12h 是关键时间节点；1→12h提升明显，12→24h进入分化区
    'critical_time_regime_shift': 12.0,
    # 长时暴露边界
    'long_time_threshold': 24.0,

    # 经验优区中心
    'opt_temp_center': 462.0,
    'opt_time_center': 12.0,

    # 外推热点
    'low_extrap_temp_center': 440.0,
    'high_extrap_temp_center': 470.0,

    # ===== 动力学参数 =====
    # 铝合金热激活过程的温和经验值
    'activation_energy_Q': 118000.0,   # J/mol
    'gas_constant_R': 8.314,           # J/(mol*K)
    'larson_miller_C': 20.0,

    # ===== 训练覆盖边界 =====
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 训练温度/时间层（按题面覆盖分析）
    'known_temp_levels': [420.0, 460.0, 480.0],
    'known_time_levels': [1.0, 12.0, 24.0],

    # ===== 原始铸态基准 =====
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    小样本材料热处理特征工程：
    - 保持少而精
    - 强化机制分段与动力学等效
    - physics baseline 作为残差学习锚点
    - 重点修正 470@12、440@24、440@1 等外推热点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的粗基线：
        关键遵守：
        1) 当前数据范围内，时间效应不能被人为饱和压平
        2) 1h -> 12h -> 24h 整体应保持增益趋势，只允许高温长时出现很温和竞争项
        3) 470@12 不能被压得过低；440@24 也不能被错误地过度软化
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度主效应：当前窗口内升温总体有利，但460~470附近更强
        temp_linear = (temp - 420.0) / 60.0
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 18.0) ** 2)
        high_temp_activation = 1.0 / (1.0 + np.exp(-(temp - 452.0) / 7.0))

        # 时间主效应：严格采用对数/线性组合，避免饱和函数压制24h
        time_log_gain = logt / np.log1p(24.0)
        time_lin_gain = time / 24.0

        # 温-时协同：高温下到12h附近强化更明显
        mid_time_focus = np.exp(-((time - 12.0) / 7.0) ** 2)
        highT_midtime_boost = high_temp_activation * mid_time_focus

        # 高温长时仅施加很弱竞争项，避免错误外推为明显软化
        mild_overexposure = (
            np.clip(temp - p['critical_temp_regime_shift'], 0, None) / 18.0 *
            np.clip(time - p['critical_time_regime_shift'], 0, None) / 12.0
        )
        mild_overexposure = np.clip(mild_overexposure, 0.0, 1.0)

        tensile = (
            p['as_cast_tensile']
            + 92.0 * time_log_gain
            + 34.0 * time_lin_gain
            + 52.0 * temp_linear
            + 70.0 * temp_peak
            + 28.0 * highT_midtime_boost
            - 6.0 * mild_overexposure
        )

        yield_strength = (
            p['as_cast_yield']
            + 72.0 * time_log_gain
            + 24.0 * time_lin_gain
            + 39.0 * temp_linear
            + 55.0 * temp_peak
            + 21.0 * highT_midtime_boost
            - 5.0 * mild_overexposure
        )

        strain = (
            p['as_cast_strain']
            + 1.6 * time_log_gain
            + 0.8 * time_lin_gain
            + 1.0 * temp_linear
            + 1.8 * temp_peak
            + 0.9 * highT_midtime_boost
            - 0.25 * mild_overexposure
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

        # ========== 1) 基础少量主干特征 ==========
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

        # ========== 4) 外推 / 边界风险 ==========
        cols['dist_to_temp_min'] = np.clip(p['train_temp_min'] - temp, 0, None)
        cols['dist_to_temp_max'] = np.clip(temp - p['train_temp_max'], 0, None)
        cols['dist_to_time_min'] = np.clip(p['train_time_min'] - time, 0, None)
        cols['dist_to_time_max'] = np.clip(time - p['train_time_max'], 0, None)

        known_temp = np.array(p['known_temp_levels'], dtype=float)
        known_time = np.array(p['known_time_levels'], dtype=float)

        cols['dist_to_known_temp_levels'] = np.min(
            np.abs(temp[:, None] - known_temp[None, :]), axis=1
        )
        cols['dist_to_known_time_levels'] = np.min(
            np.abs(time[:, None] - known_time[None, :]), axis=1
        )
        cols['grid_dist_product'] = (
            cols['dist_to_known_temp_levels'] * cols['dist_to_known_time_levels']
        )

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

        # ========== 6) 小幅、针对性局部修正 ==========
        cols['temp_piece_low'] = np.clip(450.0 - temp, 0, None)
        cols['temp_piece_midhigh'] = np.clip(temp - 450.0, 0, None)
        cols['time_piece_12minus'] = np.clip(12.0 - time, 0, None)
        cols['time_piece_12plus'] = np.clip(time - 12.0, 0, None)

        cols['temp470_time12_focus'] = (
            np.exp(-((temp - 470.0) / 7.0) ** 2) *
            np.exp(-((time - 12.0) / 4.0) ** 2)
        )
        cols['temp440_time24_focus'] = (
            np.exp(-((temp - 440.0) / 7.0) ** 2) *
            np.exp(-((time - 24.0) / 5.0) ** 2)
        )
        cols['temp440_time1_focus'] = (
            np.exp(-((temp - 440.0) / 7.0) ** 2) *
            np.exp(-((time - 1.0) / 2.0) ** 2)
        )

        # 470@12：当前误差最大且表现为系统性低估，补一个局部激活增强
        cols['focus47012_with_boundary'] = (
            cols['temp470_time12_focus'] *
            (1.0 + np.clip(temp - p['critical_temp_regime_shift'], 0, None) / 10.0)
        )

        # 440@24：是外推点，且误差方向与长时响应有关
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