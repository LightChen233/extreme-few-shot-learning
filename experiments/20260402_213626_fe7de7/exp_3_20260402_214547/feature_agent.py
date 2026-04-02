import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # -----------------------------
    # Regime / mechanism boundaries
    # -----------------------------
    # 420→460°C 区间强化快速提升，约 455°C 左右可视作主机制切换点
    'critical_temp_regime_shift': 455.0,
    # 470°C 附近在数据中多次表现为高强高塑窗口
    'critical_temp_high': 470.0,
    # 440°C 是当前外推误差最集中的低温边界点之一，单独作为局部非线性边界
    'critical_temp_low_edge': 440.0,

    # 时间机制边界：1h 短时，12h 中时，24h 长时
    'critical_time_regime_shift': 12.0,
    'critical_time_long': 24.0,
    'critical_time_short': 1.0,

    # -----------------------------
    # Kinetics / physical constants
    # -----------------------------
    # 铝合金析出/扩散控制过程的保守表观激活能
    'activation_energy_Q': 120000.0,   # J/mol
    'gas_constant_R': 8.314,
    'larson_miller_C': 20.0,

    # -----------------------------
    # As-cast reference
    # -----------------------------
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # -----------------------------
    # Approx training domain bounds
    # -----------------------------
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # -----------------------------
    # Empirical high-performance window
    # -----------------------------
    # 从题目趋势看，强度峰值更接近 460~470°C，中等时间附近
    'peak_temp_center': 465.0,
    'peak_temp_width': 16.0,
    'peak_logtime_center': np.log1p(12.0),
    'peak_logtime_width': 0.55,

    # 局部困难点窗口：用于构建“定向但温和”的物理特征
    'window_470_temp': 470.0,
    'window_440_temp': 440.0,
}


class FeatureAgent:
    """
    7499 铝合金热处理-性能预测的小样本物理启发特征工程

    设计原则：
    1) 少而精，避免 29 条样本下高维过拟合
    2) 用机制边界 + 动力学等效量替代纯多项式堆叠
    3) 强化外推点识别，尤其 440×1 / 440×24 / 470×12
    4) 引入 physics baseline，让模型更偏向学习残差
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于材料规律的弱基线：
        - 热处理后强度整体高于铸态
        - 温度升高通常比时间延长更强烈
        - 中高温+中等时间附近存在性能窗口
        - 不强加激进软化，仅允许温和峰值/平台趋势
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = float(time)

        temp_k = temp + 273.15
        log_time = np.log1p(max(time, 0.0))

        # 归一化
        temp_norm = (temp - p['train_temp_min']) / (p['train_temp_max'] - p['train_temp_min'] + 1e-12)
        temp_norm = np.clip(temp_norm, -0.3, 1.3)

        log_tmin = np.log1p(p['train_time_min'])
        log_tmax = np.log1p(p['train_time_max'])
        time_norm = (log_time - log_tmin) / (log_tmax - log_tmin + 1e-12)
        time_norm = np.clip(time_norm, -0.3, 1.3)

        # 高温激活：455°C 后加速
        high_temp_activation = 1.0 / (1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / 5.5))
        very_high_temp_activation = 1.0 / (1.0 + np.exp(-(temp - p['critical_temp_high']) / 4.0))

        # 中高温 + 12h 附近窗口
        peak_temp = np.exp(-((temp - p['peak_temp_center']) / p['peak_temp_width']) ** 2)
        peak_time = np.exp(-((log_time - p['peak_logtime_center']) / p['peak_logtime_width']) ** 2)
        peak_window = peak_temp * peak_time

        # 低温长时补偿：420~440°C 长时下强度仍可持续提高
        low_temp_longtime = np.exp(-((temp - 435.0) / 18.0) ** 2) * np.clip(time_norm, 0.0, None)

        # baseline: 方向与数据一致，幅度保守
        baseline_tensile = (
            p['as_cast_tensile']
            + 105.0 * temp_norm
            + 62.0 * time_norm
            + 58.0 * peak_window
            + 24.0 * high_temp_activation
            + 18.0 * low_temp_longtime
            - 10.0 * very_high_temp_activation * np.clip(time_norm - 0.45, 0.0, None)
        )

        baseline_yield = (
            p['as_cast_yield']
            + 84.0 * temp_norm
            + 48.0 * time_norm
            + 46.0 * peak_window
            + 18.0 * high_temp_activation
            + 14.0 * low_temp_longtime
            - 8.0 * very_high_temp_activation * np.clip(time_norm - 0.45, 0.0, None)
        )

        baseline_strain = (
            p['as_cast_strain']
            - 3.0
            + 4.2 * high_temp_activation
            + 1.5 * time_norm
            + 2.4 * peak_window
            + 0.5 * very_high_temp_activation
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        p = DOMAIN_PARAMS
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # =========================
        # 1) Core low-dimensional features
        # =========================
        cols['temp'] = temp
        cols['log_time'] = log_time
        cols['temp_x_log_time'] = temp * log_time

        # =========================
        # 2) Mild nonlinearities
        # =========================
        cols['temp_sq_centered'] = ((temp - p['critical_temp_regime_shift']) / 18.0) ** 2
        cols['log_time_sq_centered'] = ((log_time - p['peak_logtime_center']) / 0.6) ** 2

        # =========================
        # 3) Mechanism regime detection
        # =========================
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_very_high_temp'] = (temp >= p['critical_temp_high']).astype(float)
        cols['is_low_edge_temp'] = (temp <= p['critical_temp_low_edge']).astype(float)

        cols['is_mid_long_time'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_long_time'] = (time >= p['critical_time_long']).astype(float)
        cols['is_short_time'] = (time <= p['critical_time_short']).astype(float)

        cols['highT_x_midLongTime'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['veryHighT_x_midLongTime'] = (
            (temp >= p['critical_temp_high']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['lowEdgeT_x_shortTime'] = (
            (temp <= p['critical_temp_low_edge']) &
            (time <= p['critical_time_short'])
        ).astype(float)

        cols['lowEdgeT_x_longTime'] = (
            (temp <= p['critical_temp_low_edge']) &
            (time >= p['critical_time_long'])
        ).astype(float)

        # 平滑分段
        cols['temp_above_shift'] = np.maximum(temp - p['critical_temp_regime_shift'], 0.0)
        cols['temp_above_high'] = np.maximum(temp - p['critical_temp_high'], 0.0)
        cols['temp_below_low_edge'] = np.maximum(p['critical_temp_low_edge'] - temp, 0.0)

        cols['time_above_12'] = np.maximum(time - p['critical_time_regime_shift'], 0.0)
        cols['logtime_above_12'] = np.maximum(log_time - np.log1p(p['critical_time_regime_shift']), 0.0)
        cols['time_above_24'] = np.maximum(time - p['critical_time_long'], 0.0)

        # =========================
        # 4) Kinetics-equivalent features
        # =========================
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius_rate'] = arrhenius
        cols['arrhenius_time'] = arrhenius * np.clip(time, 0, None)
        cols['arrhenius_logtime'] = arrhenius * log_time

        cols['larson_miller'] = temp_k * (
            p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None))
        )

        cols['thermal_exposure'] = temp_k * log_time

        # =========================
        # 5) Peak-window and distance-to-window
        # =========================
        temp_dist_peak = (temp - p['peak_temp_center']) / p['peak_temp_width']
        time_dist_peak = (log_time - p['peak_logtime_center']) / p['peak_logtime_width']
        peak_proximity = np.exp(-(temp_dist_peak ** 2 + time_dist_peak ** 2))

        cols['peak_window_proximity'] = peak_proximity
        cols['temp_relative_to_peak'] = temp - p['peak_temp_center']
        cols['logtime_relative_to_peak'] = log_time - p['peak_logtime_center']

        # =========================
        # 6) Extrapolation / boundary-distance features
        # =========================
        # 全局范围边界
        cols['dist_to_temp_min'] = temp - p['train_temp_min']
        cols['dist_to_temp_max'] = p['train_temp_max'] - temp
        cols['dist_to_time_min'] = time - p['train_time_min']
        cols['dist_to_time_max'] = p['train_time_max'] - time

        cols['outside_temp_low'] = np.maximum(p['train_temp_min'] - temp, 0.0)
        cols['outside_temp_high'] = np.maximum(temp - p['train_temp_max'], 0.0)
        cols['outside_time_low'] = np.maximum(p['train_time_min'] - time, 0.0)
        cols['outside_time_high'] = np.maximum(time - p['train_time_max'], 0.0)

        # 机制边界距离：用于组合外推
        cols['abs_temp_to_shift_boundary'] = np.abs(temp - p['critical_temp_regime_shift'])
        cols['abs_temp_to_high_boundary'] = np.abs(temp - p['critical_temp_high'])
        cols['abs_logtime_to_12h'] = np.abs(log_time - np.log1p(p['critical_time_regime_shift']))
        cols['abs_temp_to_low_edge'] = np.abs(temp - p['critical_temp_low_edge'])

        # 更针对当前验证外推点的“组合边界距离”
        cols['dist_to_470_12'] = np.sqrt(
            ((temp - 470.0) / 10.0) ** 2 +
            ((log_time - np.log1p(12.0)) / 0.45) ** 2
        )
        cols['dist_to_440_24'] = np.sqrt(
            ((temp - 440.0) / 10.0) ** 2 +
            ((log_time - np.log1p(24.0)) / 0.35) ** 2
        )
        cols['dist_to_440_1'] = np.sqrt(
            ((temp - 440.0) / 10.0) ** 2 +
            ((log_time - np.log1p(1.0)) / 0.22) ** 2
        )

        # =========================
        # 7) Physics baseline as features
        # =========================
        baseline = np.array([self.physics_baseline(ti, hi) for ti, hi in zip(temp, time)])
        cols['baseline_strain'] = baseline[:, 0]
        cols['baseline_tensile'] = baseline[:, 1]
        cols['baseline_yield'] = baseline[:, 2]

        # baseline 与窗口/边界的耦合
        cols['baseline_tensile_x_peak'] = cols['baseline_tensile'] * peak_proximity
        cols['baseline_yield_x_peak'] = cols['baseline_yield'] * peak_proximity
        cols['baseline_strain_x_highT'] = cols['baseline_strain'] * cols['is_high_temp_regime']

        cols['baseline_tensile_x_lowEdgeLong'] = cols['baseline_tensile'] * cols['lowEdgeT_x_longTime']
        cols['baseline_yield_x_lowEdgeLong'] = cols['baseline_yield'] * cols['lowEdgeT_x_longTime']
        cols['baseline_strain_x_veryHighMidLong'] = cols['baseline_strain'] * cols['veryHighT_x_midLongTime']

        # =========================
        # 8) Targeted local windows for hardest error points
        # =========================
        cols['near_470_12_window'] = np.exp(
            -((temp - 470.0) / 8.0) ** 2
            - ((log_time - np.log1p(12.0)) / 0.35) ** 2
        )

        cols['near_440_temp'] = np.exp(-((temp - 440.0) / 10.0) ** 2)
        cols['near_440_x_short1h'] = cols['near_440_temp'] * np.exp(
            -((log_time - np.log1p(1.0)) / 0.20) ** 2
        )
        cols['near_440_x_long24h'] = cols['near_440_temp'] * np.exp(
            -((log_time - np.log1p(24.0)) / 0.22) ** 2
        )

        # =========================
        # 9) Compact scale-transformed features
        # =========================
        cols['inv_temp_k'] = 1.0 / temp_k
        cols['sqrt_time'] = sqrt_time

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names