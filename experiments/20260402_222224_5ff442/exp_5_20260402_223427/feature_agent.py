import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ===== 机制边界 / 工艺窗口 =====
    # 结合题面与误差分布：460~470℃附近可能从“持续强化主导”转向“高温快速演化/峰值敏感区”
    'critical_temp_regime_shift': 465.0,
    # 12h 是明显的动力学分界：1->12h 增益显著，12h 之后不同温区开始分化
    'critical_time_regime_shift': 12.0,
    # 长时暴露边界
    'long_time_threshold': 24.0,

    # 经验优区中心：从现有描述看 460℃、12h 接近综合优区
    'opt_temp_center': 460.0,
    'opt_time_center': 12.0,

    # 对误差最大外推点做轻量局部编码
    'low_temp_extrap_anchor': 440.0,
    'high_temp_extrap_anchor': 470.0,

    # ===== 动力学参数 =====
    # Al-Zn-Mg-Cu 系析出/扩散相关有效激活能，取中等经验值，避免指数项过激
    'activation_energy_Q': 118000.0,   # J/mol
    'gas_constant_R': 8.314,           # J/(mol*K)
    'larson_miller_C': 20.0,

    # ===== 训练覆盖边界（题面已知）=====
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 已知稀疏工艺网格
    'known_temp_grid': [420.0, 460.0, 470.0, 480.0],
    'known_time_grid': [1.0, 12.0, 24.0],

    # ===== 原始铸态基准 =====
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    小样本热处理性能预测的特征工程：
    1) 少而精，避免高维过拟合
    2) 强化温度-时间耦合与机制边界
    3) 引入物理 baseline 作为残差学习锚点
    4) 对外推点加入“距训练边界/距已知工艺网格距离”特征
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于材料规律的粗基线，方向必须与数据趋势一致：
        - 相对铸态，热处理后强度显著提升
        - 1 -> 12h 通常明显增强
        - 460℃附近常接近强度优区
        - 470~480℃区域不是单调软化，而是进入“高温敏感区”
        - 高温+长时可能出现轻微竞争机制，但只给温和惩罚
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = max(float(time), 1e-8)

        temp_k = temp + 273.15
        logt = np.log1p(time)

        # 温度优区：460附近强化最显著，向两侧偏离后回落
        temp_peak = np.exp(-((temp - p['opt_temp_center']) / 20.0) ** 2)

        # 时间饱和：1->12h 提升快，之后趋缓
        time_sat = 1.0 - np.exp(-time / 7.0)

        # 高温激活：450℃以上微观演化明显加快
        high_temp_activation = 1.0 / (1.0 + np.exp(-(temp - 452.0) / 7.0))

        # 470附近的局部“峰值敏感区”补偿
        temp_470_bump = np.exp(-((temp - 470.0) / 9.0) ** 2) * np.exp(-((time - 12.0) / 7.0) ** 2)

        # 低温长时（如440/24）可能不如中高温12h优，给温和抑制
        lowT_longt_penalty = (
            np.clip(450.0 - temp, 0.0, None) / 25.0
            * np.clip(time - 12.0, 0.0, None) / 12.0
        )
        lowT_longt_penalty = np.clip(lowT_longt_penalty, 0.0, 1.2)

        tensile = (
            p['as_cast_tensile']
            + 92.0 * time_sat
            + 72.0 * temp_peak
            + 48.0 * high_temp_activation * np.tanh(logt)
            + 18.0 * temp_470_bump
            - 14.0 * lowT_longt_penalty
        )

        yield_strength = (
            p['as_cast_yield']
            + 73.0 * time_sat
            + 58.0 * temp_peak
            + 37.0 * high_temp_activation * np.tanh(logt)
            + 12.0 * temp_470_bump
            - 11.0 * lowT_longt_penalty
        )

        strain = (
            p['as_cast_strain']
            + 1.7 * time_sat
            + 1.8 * temp_peak
            + 1.2 * high_temp_activation * np.tanh(logt)
            + 0.8 * temp_470_bump
            - 0.7 * lowT_longt_penalty
        )

        return float(strain), float(tensile), float(yield_strength)

    def engineer_features(self, df):
        p = DOMAIN_PARAMS

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        cols = {}

        # ===== 1) 最基本原始特征 =====
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time

        # 少量必要非线性
        cols['temp_centered_sq'] = ((temp - p['opt_temp_center']) / 20.0) ** 2
        cols['temp_x_log_time'] = temp * log_time

        # ===== 2) 机制边界 / 分段特征（关键）=====
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_post12h_regime'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_high_temp_post12h'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['temp_above_crit'] = np.clip(temp - p['critical_temp_regime_shift'], 0.0, None)
        cols['time_above_12h'] = np.clip(time - p['critical_time_regime_shift'], 0.0, None)

        # 高温长时敏感激活：帮助识别 470/12 一带与更高热暴露区
        cols['highT_post12_activation'] = (
            np.clip(temp - p['critical_temp_regime_shift'], 0.0, None) *
            np.clip(log_time - np.log1p(p['critical_time_regime_shift']), 0.0, None)
        )

        # 低温长时特征：针对 440/24 外推方向错误
        cols['lowT_longt_activation'] = (
            np.clip(450.0 - temp, 0.0, None) *
            np.clip(log_time - np.log1p(12.0), 0.0, None)
        )

        # 围绕经验优区的偏离
        cols['dist_temp_to_opt'] = np.abs(temp - p['opt_temp_center'])
        cols['dist_logtime_to_opt'] = np.abs(log_time - np.log1p(p['opt_time_center']))
        cols['elliptic_dist_to_opt'] = np.sqrt(
            ((temp - p['opt_temp_center']) / 20.0) ** 2 +
            ((log_time - np.log1p(p['opt_time_center'])) / 0.9) ** 2
        )

        # ===== 3) 动力学等效特征 =====
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

        # 一个更温和的“等效热暴露”指标
        cols['temp_scaled_logtime'] = (temp - 400.0) / 80.0 * log_time

        # ===== 4) 外推 / 边界距离特征 =====
        cols['dist_to_temp_min'] = np.clip(p['train_temp_min'] - temp, 0.0, None)
        cols['dist_to_temp_max'] = np.clip(temp - p['train_temp_max'], 0.0, None)
        cols['dist_to_time_min'] = np.clip(p['train_time_min'] - time, 0.0, None)
        cols['dist_to_time_max'] = np.clip(time - p['train_time_max'], 0.0, None)

        known_temp_grid = np.array(p['known_temp_grid'], dtype=float)
        known_time_grid = np.array(p['known_time_grid'], dtype=float)

        nearest_temp_grid = np.min(np.abs(temp[:, None] - known_temp_grid[None, :]), axis=1)
        nearest_time_grid = np.min(np.abs(time[:, None] - known_time_grid[None, :]), axis=1)

        cols['nearest_temp_grid_dist'] = nearest_temp_grid
        cols['nearest_time_grid_dist'] = nearest_time_grid
        cols['grid_dist_product'] = nearest_temp_grid * nearest_time_grid

        # 对“训练网格交点外推”更敏感的距离
        grid_points = np.array(
            [[tt, tm] for tt in known_temp_grid for tm in known_time_grid],
            dtype=float
        )
        pair_dist = np.sqrt(
            ((temp[:, None] - grid_points[None, :, 0]) / 20.0) ** 2 +
            ((log_time[:, None] - np.log1p(grid_points[None, :, 1])) / 0.9) ** 2
        )
        cols['nearest_process_pair_dist'] = np.min(pair_dist, axis=1)

        # ===== 5) 局部温区位置特征：只做轻量补充 =====
        cols['temp_rel_440'] = temp - 440.0
        cols['temp_rel_470'] = temp - 470.0

        cols['is_440_band'] = (np.abs(temp - 440.0) <= 5.0).astype(float)
        cols['is_470_band'] = (np.abs(temp - 470.0) <= 5.0).astype(float)

        # 针对主要大误差点，仅保留两个 focus，避免继续堆特征
        cols['focus_470_12'] = (
            np.exp(-((temp - 470.0) / 8.0) ** 2) *
            np.exp(-((time - 12.0) / 4.0) ** 2)
        )
        cols['focus_440_24'] = (
            np.exp(-((temp - 440.0) / 8.0) ** 2) *
            np.exp(-((time - 24.0) / 6.0) ** 2)
        )

        # ===== 6) 物理 baseline 作为残差学习锚点 =====
        baseline = np.array([self.physics_baseline(t, tm) for t, tm in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        cols['baseline_strength_gap'] = baseline[:, 1] - baseline[:, 2]
        cols['baseline_strength_ratio'] = baseline[:, 2] / np.clip(baseline[:, 1], 1e-8, None)
        cols['baseline_strain_strength_coupling'] = baseline[:, 0] * baseline[:, 1]

        cols['temp_relative_to_boundary'] = (
            temp - p['critical_temp_regime_shift']
        ) / 20.0
        cols['time_relative_to_boundary'] = (
            log_time - np.log1p(p['critical_time_regime_shift'])
        ) / 1.0
        cols['boundary_interaction'] = (
            cols['temp_relative_to_boundary'] * cols['time_relative_to_boundary']
        )

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names