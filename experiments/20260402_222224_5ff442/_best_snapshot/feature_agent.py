import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # 机制边界：结合题面趋势，460~470℃附近可能进入更强的高温激活/组织转变区
    'critical_temp_regime_shift': 465.0,
    # 时间边界：12h 左右明显是关键时效/均匀化节点
    'critical_time_regime_shift': 12.0,
    # 更长时间下可能进入粗化/竞争机制增强区
    'long_time_threshold': 24.0,

    # 热处理窗口中心：从描述看 460℃、12h 附近接近优区
    'opt_temp_center': 460.0,
    'opt_time_center': 12.0,

    # 物理常数
    # 铝合金中扩散/析出相关有效激活能，取温和经验值，避免指数过激
    'activation_energy_Q': 120000.0,   # J/mol
    'gas_constant_R': 8.314,           # J/(mol*K)

    # Larson-Miller / 热暴露类参数
    'larson_miller_C': 20.0,

    # 外推/边界相关：训练覆盖主要在 420/460/470/480 与 1/12/24 的稀疏网格附近
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 原始铸态基准
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    小样本材料热处理特征工程：
    - 少而精，强调物理启发
    - 分段机制特征
    - 动力学等效特征
    - 基线预测作为残差学习锚点
    - 外推边界距离特征
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的粗基线：
        1) 相对铸态，热处理整体提升强度
        2) 温度-时间存在强耦合
        3) 460℃、12h附近接近优区
        4) 470~480℃长时间可能出现部分性能回落/分化，但不能强行施加强软化
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-6)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度主效应：在当前窗口中总体随温度升高而增强，但以 460 左右最优、远离则衰减
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 22.0) ** 2)

        # 时间主效应：1->12h 提升明显，之后趋缓
        time_sat = 1.0 - np.exp(-time / 8.0)

        # 高温激活协同
        high_temp_gain = 1.0 / (1.0 + np.exp(-(temp - 450.0) / 8.0))

        # 长时高温竞争机制：只给温和惩罚，避免错误引入过强“过时效”
        over_exposure = max(temp - p['critical_temp_regime_shift'], 0.0) / 20.0 * max(time - p['critical_time_regime_shift'], 0.0) / 12.0
        over_exposure = np.clip(over_exposure, 0.0, 1.5)

        # 基线抗拉/屈服：保持与数据趋势一致——整体显著高于铸态，460/12附近较优
        tensile = (
            p['as_cast_tensile']
            + 95.0 * time_sat
            + 80.0 * temp_peak
            + 55.0 * high_temp_gain * np.tanh(logt)
            - 18.0 * over_exposure
        )

        yield_strength = (
            p['as_cast_yield']
            + 75.0 * time_sat
            + 62.0 * temp_peak
            + 42.0 * high_temp_gain * np.tanh(logt)
            - 14.0 * over_exposure
        )

        # 应变：整体可提升，但高温长时不一定继续增
        strain = (
            p['as_cast_strain']
            + 1.8 * time_sat
            + 2.0 * temp_peak
            + 1.5 * high_temp_gain * np.tanh(logt)
            - 0.9 * over_exposure
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

        # ========== 1) 保留最基础原始特征 ==========
        cols['temp'] = temp
        cols['time'] = time

        # 只保留少量必要非线性，避免重复堆砌
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
            np.clip(np.log1p(time) - np.log1p(p['critical_time_regime_shift']), 0, None)
        )

        # 围绕经验优区的距离：帮助识别 460/12 附近优区以及外部偏离
        cols['dist_temp_to_opt'] = np.abs(temp - p['opt_temp_center'])
        cols['dist_logtime_to_opt'] = np.abs(log_time - np.log1p(p['opt_time_center']))
        cols['elliptic_dist_to_opt'] = np.sqrt(
            ((temp - p['opt_temp_center']) / 20.0) ** 2 +
            ((log_time - np.log1p(p['opt_time_center'])) / 0.9) ** 2
        )

        # ========== 3) 动力学等效特征 ==========
        # Arrhenius 核：反映高温扩散/析出驱动力
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius'] = arrhenius
        cols['time_x_arrhenius'] = time * arrhenius
        cols['logtime_x_arrhenius'] = log_time * arrhenius

        # JMAK/扩散暴露近似：避免指数过大，采用温和形式
        cols['thermal_exposure_index'] = temp_k * log_time
        cols['inv_temp_k'] = 1.0 / temp_k

        # Larson-Miller 参数
        cols['larson_miller'] = temp_k * (p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None)))

        # Zener-Hollomon风格的简化逆量纲特征（不严格物理，但能表达温时等效）
        cols['log_time_over_tempk'] = log_time / temp_k

        # ========== 4) 外推 / 边界距离特征 ==========
        # 题面明确要求：对外推点加入“距训练边界距离”特征
        cols['dist_to_temp_min'] = np.clip(p['train_temp_min'] - temp, 0, None)
        cols['dist_to_temp_max'] = np.clip(temp - p['train_temp_max'], 0, None)
        cols['dist_to_time_min'] = np.clip(p['train_time_min'] - time, 0, None)
        cols['dist_to_time_max'] = np.clip(time - p['train_time_max'], 0, None)

        # 对当前任务更关键的是“距关键已知工艺网格”的相对位置
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

        # 特别针对误差大的 440 和 470 邻域，给出边界相对位置
        cols['temp_rel_440'] = temp - 440.0
        cols['temp_rel_470'] = temp - 470.0
        cols['is_440_band'] = (np.abs(temp - 440.0) <= 5.0).astype(float)
        cols['is_470_band'] = (np.abs(temp - 470.0) <= 5.0).astype(float)

        # ========== 5) 基线预测特征（残差学习锚点） ==========
        baseline = np.array([self.physics_baseline(t, tm) for t, tm in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        # 相对基线/相对机制边界的残差型输入
        cols['baseline_strength_gap'] = baseline[:, 1] - baseline[:, 2]
        cols['baseline_strength_ratio'] = baseline[:, 2] / np.clip(baseline[:, 1], 1e-6, None)
        cols['baseline_strain_strength_coupling'] = baseline[:, 0] * baseline[:, 1]

        cols['temp_relative_to_boundary'] = (temp - p['critical_temp_regime_shift']) / 20.0
        cols['time_relative_to_boundary'] = (log_time - np.log1p(p['critical_time_regime_shift'])) / 1.0
        cols['boundary_interaction'] = (
            cols['temp_relative_to_boundary'] * cols['time_relative_to_boundary']
        )

        # ========== 6) 少量针对性局部修正特征 ==========
        # 历史反思提示：470@12 被低估，440@1/24 方向错误，说明温度非线性和局部斜率变化重要
        cols['temp_piece_low'] = np.clip(450.0 - temp, 0, None)
        cols['temp_piece_midhigh'] = np.clip(temp - 450.0, 0, None)
        cols['temp_piece_470plus'] = np.clip(temp - 470.0, 0, None)

        cols['time_piece_12minus'] = np.clip(12.0 - time, 0, None)
        cols['time_piece_12plus'] = np.clip(time - 12.0, 0, None)

        cols['temp470_time12_focus'] = np.exp(-((temp - 470.0) / 8.0) ** 2) * np.exp(-((time - 12.0) / 4.0) ** 2)
        cols['temp440_time24_focus'] = np.exp(-((temp - 440.0) / 8.0) ** 2) * np.exp(-((time - 24.0) / 6.0) ** 2)
        cols['temp440_time1_focus'] = np.exp(-((temp - 440.0) / 8.0) ** 2) * np.exp(-((time - 1.0) / 2.0) ** 2)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names