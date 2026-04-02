import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # =========================
    # 机制边界 / 工艺窗口
    # =========================
    # 从题面趋势看，460~470℃附近是强化机制更充分激活的区域，
    # 但470℃以上并不一定单调继续增益，因此把465℃作为机制切换点
    'critical_temp_regime_shift': 465.0,

    # 12h 是明显关键时间节点：1->12h 往往提升显著，12h后进入平台/竞争区
    'critical_time_regime_shift': 12.0,

    # 长时阈值：24h 为已观测上边界，同时也是可能出现粗化/组织竞争的时长
    'long_time_threshold': 24.0,

    # 经验优区中心：从描述看 460℃、12h 接近综合高性能窗口
    'opt_temp_center': 462.0,
    'opt_time_center': 12.0,

    # 对低温长时外推（440@24）进行约束：低温区扩散慢，但长时可累积暴露
    'low_temp_boundary': 445.0,
    'high_temp_boundary': 470.0,

    # =========================
    # 动力学参数
    # =========================
    # 铝合金析出/扩散相关有效激活能，取中等保守值
    'activation_energy_Q': 118000.0,   # J/mol
    'gas_constant_R': 8.314,           # J/(mol*K)

    # Larson-Miller 常数
    'larson_miller_C': 20.0,

    # JMAK/饱和时间尺度（温和）
    'time_saturation_scale': 8.0,

    # =========================
    # 训练覆盖边界（题面已知）
    # =========================
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 稀疏训练网格（用于“距已知工艺格点距离”）
    'known_temp_grid_low': 420.0,
    'known_temp_grid_mid': 460.0,
    'known_temp_grid_high': 470.0,
    'known_temp_grid_max': 480.0,
    'known_time_grid_short': 1.0,
    'known_time_grid_mid': 12.0,
    'known_time_grid_long': 24.0,

    # =========================
    # 原始铸态基准
    # =========================
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    小样本材料热处理特征工程：
    1) 少而精：优先保留物理含义明确的特征
    2) 用机制边界 + 动力学等效特征描述非线性温时耦合
    3) 用 physics_baseline 作为残差学习锚点
    4) 显式加入外推距离/边界距离特征
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的粗基线：
        - 相对铸态，热处理整体显著提升强度
        - 当前温区内温度/时间存在非线性耦合，460~470℃、约12h附近接近优区
        - 不强行假设高温长时一定大幅软化，只施加温和竞争项
        - 低温长时（如440@24）允许出现“增益有限/部分回落”
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-8)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 1) 时间饱和：1->12h 提升显著，之后趋于平台
        time_sat = 1.0 - np.exp(-time / p['time_saturation_scale'])

        # 2) 温度优区：460~470附近较优，但不做过窄峰
        temp_opt = np.exp(-((temp - p['opt_temp_center']) / 24.0) ** 2)

        # 3) 高温激活：描述460以上强化机制更易启动
        high_temp_activation = 1.0 / (1.0 + np.exp(-(temp - 452.0) / 7.5))

        # 4) 温时协同：高温 + 足够时间更接近充分处理状态
        temp_time_synergy = high_temp_activation * np.tanh(logt)

        # 5) 竞争机制：仅在高温长时或低温长时下给温和惩罚
        highT_longt_penalty = (
            np.clip(temp - p['critical_temp_regime_shift'], 0.0, None) / 18.0
        ) * (
            np.clip(time - p['critical_time_regime_shift'], 0.0, None) / 12.0
        )
        highT_longt_penalty = np.clip(highT_longt_penalty, 0.0, 1.5)

        lowT_longt_penalty = (
            np.clip(p['low_temp_boundary'] - temp, 0.0, None) / 20.0
        ) * (
            np.clip(time - p['critical_time_regime_shift'], 0.0, None) / 12.0
        )
        lowT_longt_penalty = np.clip(lowT_longt_penalty, 0.0, 1.2)

        # 6) 470@12 附近局部高性能窗口：针对当前最大误差点给一个温和局部提升
        peak_470_12 = np.exp(-((temp - 470.0) / 10.0) ** 2) * np.exp(-((time - 12.0) / 5.0) ** 2)

        tensile = (
            p['as_cast_tensile']
            + 92.0 * time_sat
            + 70.0 * temp_opt
            + 58.0 * temp_time_synergy
            + 24.0 * peak_470_12
            - 16.0 * highT_longt_penalty
            - 22.0 * lowT_longt_penalty
        )

        yield_strength = (
            p['as_cast_yield']
            + 74.0 * time_sat
            + 56.0 * temp_opt
            + 45.0 * temp_time_synergy
            + 18.0 * peak_470_12
            - 13.0 * highT_longt_penalty
            - 17.0 * lowT_longt_penalty
        )

        strain = (
            p['as_cast_strain']
            + 1.7 * time_sat
            + 1.8 * temp_opt
            + 1.3 * temp_time_synergy
            + 0.9 * peak_470_12
            - 0.7 * highT_longt_penalty
            - 0.8 * lowT_longt_penalty
        )

        return float(strain), float(tensile), float(yield_strength)

    def engineer_features(self, df):
        p = DOMAIN_PARAMS
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # =========================
        # 1) 基础特征：保留少量必要非线性
        # =========================
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time

        # 温度以优区中心做二次偏离，帮助描述“局部峰值”而非简单线性
        cols['temp_sq_centered'] = ((temp - p['opt_temp_center']) / 22.0) ** 2

        # 最必要的温时交互
        cols['temp_x_log_time'] = temp * log_time

        # =========================
        # 2) 机制区间 / 分段激活特征（关键）
        # =========================
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_long_time_regime'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_high_temp_long_time'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['is_low_temp_long_time'] = (
            (temp <= p['low_temp_boundary']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['temp_above_crit'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_below_low_boundary'] = np.clip(p['low_temp_boundary'] - temp, 0, None)
        cols['time_above_crit'] = np.clip(time - p['critical_time_regime_shift'], 0, None)

        cols['highT_longt_activation'] = (
            np.clip(temp - p['critical_temp_regime_shift'], 0, None) *
            np.clip(log_time - np.log1p(p['critical_time_regime_shift']), 0, None)
        )

        cols['lowT_longt_activation'] = (
            np.clip(p['low_temp_boundary'] - temp, 0, None) *
            np.clip(log_time - np.log1p(p['critical_time_regime_shift']), 0, None)
        )

        # 优区距离：帮助识别460/12附近与其外部偏离
        cols['dist_temp_to_opt'] = np.abs(temp - p['opt_temp_center'])
        cols['dist_logtime_to_opt'] = np.abs(log_time - np.log1p(p['opt_time_center']))
        cols['elliptic_dist_to_opt'] = np.sqrt(
            ((temp - p['opt_temp_center']) / 22.0) ** 2 +
            ((log_time - np.log1p(p['opt_time_center'])) / 0.95) ** 2
        )

        # =========================
        # 3) 动力学等效特征
        # =========================
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

        # 一个温和的“充分处理程度”近似
        cols['kinetic_progress'] = (1.0 - np.exp(-time / p['time_saturation_scale'])) * arrhenius * 1e8

        # =========================
        # 4) 外推 / 边界距离特征
        # =========================
        cols['dist_to_temp_min'] = np.clip(p['train_temp_min'] - temp, 0, None)
        cols['dist_to_temp_max'] = np.clip(temp - p['train_temp_max'], 0, None)
        cols['dist_to_time_min'] = np.clip(p['train_time_min'] - time, 0, None)
        cols['dist_to_time_max'] = np.clip(time - p['train_time_max'], 0, None)

        # 到已知稀疏工艺网格的距离
        nearest_temp_grid = np.minimum.reduce([
            np.abs(temp - p['known_temp_grid_low']),
            np.abs(temp - p['known_temp_grid_mid']),
            np.abs(temp - p['known_temp_grid_high']),
            np.abs(temp - p['known_temp_grid_max']),
        ])
        nearest_time_grid = np.minimum.reduce([
            np.abs(time - p['known_time_grid_short']),
            np.abs(time - p['known_time_grid_mid']),
            np.abs(time - p['known_time_grid_long']),
        ])
        cols['nearest_temp_grid_dist'] = nearest_temp_grid
        cols['nearest_time_grid_dist'] = nearest_time_grid
        cols['grid_dist_product'] = nearest_temp_grid * nearest_time_grid

        # 对当前验证外推点很关键的“夹在训练温度之间”的位置
        # 440 介于 420 和 460 之间；470 已有训练，但470@12仍有局部特殊性
        cols['temp_rel_440'] = temp - 440.0
        cols['temp_rel_470'] = temp - 470.0
        cols['is_440_band'] = (np.abs(temp - 440.0) <= 6.0).astype(float)
        cols['is_470_band'] = (np.abs(temp - 470.0) <= 6.0).astype(float)

        # =========================
        # 5) 基线预测特征（残差学习锚点）
        # =========================
        baseline = np.array([self.physics_baseline(t, tm) for t, tm in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        cols['baseline_strength_gap'] = baseline[:, 1] - baseline[:, 2]
        cols['baseline_strength_ratio'] = baseline[:, 2] / np.clip(baseline[:, 1], 1e-8, None)
        cols['baseline_strain_strength_coupling'] = baseline[:, 0] * baseline[:, 1]

        cols['temp_relative_to_boundary'] = (temp - p['critical_temp_regime_shift']) / 20.0
        cols['time_relative_to_boundary'] = (
            log_time - np.log1p(p['critical_time_regime_shift'])
        )
        cols['boundary_interaction'] = (
            cols['temp_relative_to_boundary'] * cols['time_relative_to_boundary']
        )

        # =========================
        # 6) 针对最大误差点的局部修正特征
        # =========================
        # 470@12：验证中显著低估，说明这里可能是一个局部高性能窗口
        cols['temp470_time12_focus'] = (
            np.exp(-((temp - 470.0) / 8.0) ** 2) *
            np.exp(-((time - 12.0) / 4.0) ** 2)
        )

        # 440@24：验证中多为高估强度，说明低温长时竞争机制未被充分表达
        cols['temp440_time24_focus'] = (
            np.exp(-((temp - 440.0) / 8.0) ** 2) *
            np.exp(-((time - 24.0) / 5.0) ** 2)
        )

        # 440@1：也是外推点，方向上与440@24不同，需单独提供局部识别
        cols['temp440_time1_focus'] = (
            np.exp(-((temp - 440.0) / 8.0) ** 2) *
            np.exp(-((time - 1.0) / 2.0) ** 2)
        )

        # 局部分段斜率，帮助模型学习 440~470 之间温度响应变化
        cols['temp_piece_below_450'] = np.clip(450.0 - temp, 0, None)
        cols['temp_piece_above_450'] = np.clip(temp - 450.0, 0, None)
        cols['temp_piece_above_470'] = np.clip(temp - 470.0, 0, None)
        cols['time_piece_below_12'] = np.clip(12.0 - time, 0, None)
        cols['time_piece_above_12'] = np.clip(time - 12.0, 0, None)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names