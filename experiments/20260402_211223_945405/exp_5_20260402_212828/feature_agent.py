import pandas as pd
import numpy as np


# 7499 铝合金：小样本（29条）热处理特征工程
# 设计思路：
# 1) 维持“少而精”，避免继续膨胀特征维度；
# 2) 最大误差集中在外推组合 470/12，其次 440/24、440/1；
# 3) 因此外推特征要强调：
#    - 机制边界（高温/中时效/长时效）
#    - 组合缺口（训练中缺少该 temp-time 组合）
#    - 距训练组合边界/热点的距离
# 4) physics_baseline 必须保持与当前统计一致：
#    在本数据覆盖的局部窗口内，更高温度/更长时间总体上对应更高强度，
#    但在 440°C 附近时间响应可能非单调，470/12 附近可能有局部强化峰。


DOMAIN_PARAMS = {
    # -------- 机制边界 / regime boundaries --------
    'critical_temp_regime_shift': 455.0,   # 低温区(≈440)到高温强化区(≈460-470)的切换
    'secondary_temp_boundary': 470.0,      # 高温端重点边界
    'critical_time_regime_shift': 12.0,    # 中时效阶段边界
    'secondary_time_boundary': 24.0,       # 长时效边界
    'short_time_boundary': 1.0,            # 短时边界

    # -------- 动力学参数（铝合金析出/扩散量级的温和估计）--------
    'activation_energy_Q': 90000.0,        # J/mol
    'gas_constant_R': 8.314,
    'larson_miller_C': 20.0,

    # -------- 训练域范围 --------
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # -------- 外推/缺口组合锚点 --------
    'missing_temp_low': 440.0,
    'missing_temp_high': 470.0,
    'missing_time_short': 1.0,
    'missing_time_mid': 12.0,
    'missing_time_long': 24.0,

    # -------- 原始凝固态平均性能 --------
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # -------- 在当前数据窗内允许的温和增益幅度 --------
    'max_strength_gain_tensile': 102.0,
    'max_strength_gain_yield': 78.0,
    'max_strain_gain': 1.6,

    # -------- 重点局部区宽度 --------
    'zone_temp_sigma': 6.0,
    'zone_time_sigma_mid': 3.0,
    'zone_time_sigma_long': 4.0,
    'zone_time_sigma_short': 0.9,

    # -------- 距离缩放（因为 time 数值尺度远小于 temp）--------
    'time_distance_scale_mid': 2.0,
    'time_distance_scale_long': 1.25,
    'time_distance_scale_short': 4.0,

    # -------- logistic 平滑宽度 --------
    'temp_sigmoid_width': 3.8,
    'time_sigmoid_width_mid': 1.8,
    'time_sigmoid_width_long': 2.2,
}


class FeatureAgent:
    """
    面向小样本热处理问题的物理引导特征工程：
    1) 基础 process 特征；
    2) 动力学等效特征（Arrhenius / Larson-Miller）；
    3) 机制边界检测特征；
    4) 外推边界/组合缺口距离；
    5) 基于物理先验的 baseline，让模型学习残差修正。
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于世界知识与当前数据统计一致性的温和基线：

        - 在当前局部窗口内（约 440–470°C, 1–24h），整体强化随温度/时间推进；
        - 不强行写入全局“过时效软化”；
        - 但允许：
          1) 470/12 附近存在局部强化峰；
          2) 440°C 下 1h 与 24h 处存在时间响应不稳定/非单调。

        返回:
            baseline_strain, baseline_tensile, baseline_yield
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = max(float(time), 1e-8)
        T_k = temp + 273.15

        # 局部归一化（以实际误差最关注区为中心）
        temp_norm = (temp - 440.0) / 30.0
        temp_norm = np.clip(temp_norm, -0.8, 1.4)

        time_norm = np.log1p(time) / np.log1p(24.0)
        time_norm = np.clip(time_norm, 0.0, 1.2)

        # 相对 Arrhenius 推进度（相对 440°C）
        T_ref = 440.0 + 273.15
        arr_rel = np.exp(
            -(p['activation_energy_Q'] / p['gas_constant_R']) * (1.0 / T_k - 1.0 / T_ref)
        )
        arr_prog = np.clip((arr_rel - 1.0) / 2.5, -0.2, 1.8)

        # 机制激活
        high_temp = 1.0 / (
            1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / p['temp_sigmoid_width'])
        )
        stage_12 = 1.0 / (
            1.0 + np.exp(-(time - p['critical_time_regime_shift']) / p['time_sigmoid_width_mid'])
        )
        stage_24 = 1.0 / (
            1.0 + np.exp(-(time - p['secondary_time_boundary']) / p['time_sigmoid_width_long'])
        )

        # 主导热暴露推进量
        exposure = (
            0.36 * temp_norm +
            0.22 * time_norm +
            0.24 * arr_prog +
            0.10 * high_temp +
            0.08 * stage_12
        )
        exposure = np.clip(exposure, -0.15, 1.40)

        # 重点区域局部机制
        zone_470_12 = np.exp(-((temp - 470.0) / 6.5) ** 2 - ((time - 12.0) / 3.2) ** 2)
        zone_440_24 = np.exp(-((temp - 440.0) / 6.5) ** 2 - ((time - 24.0) / 4.2) ** 2)
        zone_440_1 = np.exp(-((temp - 440.0) / 6.5) ** 2 - ((time - 1.0) / 1.0) ** 2)

        # 470/12 是最强低估区：对强度 baseline 做更明确但仍温和的抬升
        local_boost_strength = 0.18 * zone_470_12 + 0.05 * zone_440_24 + 0.02 * zone_440_1
        local_boost_yield = 0.15 * zone_470_12 + 0.07 * zone_440_24 + 0.01 * zone_440_1
        local_boost_strain = 0.05 * zone_470_12 - 0.02 * zone_440_24 + 0.02 * zone_440_1

        baseline_tensile = (
            p['as_cast_tensile'] +
            p['max_strength_gain_tensile'] * (exposure + local_boost_strength)
        )
        baseline_yield = (
            p['as_cast_yield'] +
            p['max_strength_gain_yield'] * (exposure + local_boost_yield)
        )

        # 应变只做弱响应，避免牺牲强度拟合稳定性
        baseline_strain = (
            p['as_cast_strain']
            + 0.42 * time_norm
            - 0.10 * temp_norm
            + 0.14 * arr_prog
            + 0.08 * stage_12
            - 0.03 * stage_24
            + local_boost_strain
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        cols = {}
        p = DOMAIN_PARAMS

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        time_clip = np.clip(time, 1e-8, None)
        T_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ================= 1) 基础特征 =================
        cols['temp'] = temp
        cols['time'] = time
        cols['log1p_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_x_logtime'] = temp * log_time
        cols['temp_centered_sq'] = ((temp - p['critical_temp_regime_shift']) / 20.0) ** 2

        # ================= 2) 动力学等效特征 =================
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['arrhenius'] = arrhenius
        cols['log_arrhenius'] = np.log(np.clip(arrhenius, 1e-300, None))

        T_ref = 440.0 + 273.15
        arr_rel = np.exp(
            -(p['activation_energy_Q'] / p['gas_constant_R']) * (1.0 / T_k - 1.0 / T_ref)
        )
        arr_prog = np.clip((arr_rel - 1.0) / 2.5, -0.2, 1.8)

        cols['arrhenius_relative'] = arr_rel
        cols['arrhenius_time'] = arr_rel * time_clip
        cols['arrhenius_logtime'] = arr_rel * log_time
        cols['arrhenius_progress'] = arr_prog

        cols['larson_miller'] = T_k * (
            p['larson_miller_C'] + np.log10(np.clip(time_clip, 1e-8, None))
        )
        cols['thermal_dose'] = T_k * log_time
        cols['zener_like'] = 1.0 / np.clip(arr_rel * time_clip, 1e-8, None)

        # ================= 3) 机制区间检测特征（关键） =================
        cols['temp_above_regime'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_below_regime'] = np.clip(p['critical_temp_regime_shift'] - temp, 0, None)
        cols['time_above_regime'] = np.clip(time - p['critical_time_regime_shift'], 0, None)
        cols['time_below_regime'] = np.clip(p['critical_time_regime_shift'] - time, 0, None)

        cols['is_high_temp'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_very_high_temp'] = (temp >= p['secondary_temp_boundary']).astype(float)
        cols['is_mid_long_time'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_long_time'] = (time >= p['secondary_time_boundary']).astype(float)
        cols['is_short_time'] = (time <= 1.5).astype(float)

        cols['is_high_temp_midlong_time'] = (
            (temp >= p['critical_temp_regime_shift']) & (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['is_very_high_temp_12h_zone'] = (
            (temp >= 468.0) & (time >= 8.0) & (time <= 16.0)
        ).astype(float)

        cols['is_440_long_time_zone'] = (
            (temp <= 442.0) & (time >= 20.0)
        ).astype(float)

        cols['is_440_short_time_zone'] = (
            (temp <= 442.0) & (time <= 1.5)
        ).astype(float)

        cols['sig_temp_regime'] = 1.0 / (
            1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / 4.0)
        )
        cols['sig_temp_470'] = 1.0 / (
            1.0 + np.exp(-(temp - p['secondary_temp_boundary']) / 2.4)
        )
        cols['sig_time_12'] = 1.0 / (
            1.0 + np.exp(-(time - p['critical_time_regime_shift']) / 1.9)
        )
        cols['sig_time_24'] = 1.0 / (
            1.0 + np.exp(-(time - p['secondary_time_boundary']) / 2.2)
        )

        # 新增但很克制：中时效峰值提示（服务 470/12）
        cols['mid_time_window'] = np.exp(-((time - 12.0) / 5.0) ** 2)
        cols['highT_midtime_synergy'] = cols['sig_temp_470'] * cols['mid_time_window']

        # ================= 4) 重点误差区局部特征 =================
        sT = p['zone_temp_sigma']
        s12 = p['zone_time_sigma_mid']
        s24 = p['zone_time_sigma_long']
        s1 = p['zone_time_sigma_short']

        cols['near_470'] = np.exp(-((temp - 470.0) / sT) ** 2)
        cols['near_440'] = np.exp(-((temp - 440.0) / sT) ** 2)
        cols['near_12h'] = np.exp(-((time - 12.0) / s12) ** 2)
        cols['near_24h'] = np.exp(-((time - 24.0) / s24) ** 2)
        cols['near_1h'] = np.exp(-((time - 1.0) / s1) ** 2)

        cols['zone_470_12'] = cols['near_470'] * cols['near_12h']
        cols['zone_440_24'] = cols['near_440'] * cols['near_24h']
        cols['zone_440_1'] = cols['near_440'] * cols['near_1h']

        cols['u_time_edge_at_440'] = cols['near_440'] * np.maximum(cols['near_1h'], cols['near_24h'])
        cols['mid_stage_at_highT'] = cols['near_12h'] * cols['sig_temp_regime']

        # ================= 5) 外推 / 边界 / 组合缺口特征 =================
        cols['dist_temp_to_train_min'] = temp - p['train_temp_min']
        cols['dist_temp_to_train_max'] = p['train_temp_max'] - temp
        cols['dist_time_to_train_min'] = time - p['train_time_min']
        cols['dist_time_to_train_max'] = p['train_time_max'] - time

        cols['inside_train_temp_range'] = (
            (temp >= p['train_temp_min']) & (temp <= p['train_temp_max'])
        ).astype(float)
        cols['inside_train_time_range'] = (
            (time >= p['train_time_min']) & (time <= p['train_time_max'])
        ).astype(float)

        cols['temp_rel_440'] = temp - p['missing_temp_low']
        cols['temp_rel_470'] = temp - p['missing_temp_high']
        cols['time_rel_1'] = time - p['missing_time_short']
        cols['time_rel_12'] = time - p['missing_time_mid']
        cols['time_rel_24'] = time - p['missing_time_long']

        cols['dist_to_470_12'] = np.sqrt(
            (temp - 470.0) ** 2 + ((time - 12.0) * p['time_distance_scale_mid']) ** 2
        )
        cols['dist_to_440_24'] = np.sqrt(
            (temp - 440.0) ** 2 + ((time - 24.0) * p['time_distance_scale_long']) ** 2
        )
        cols['dist_to_440_1'] = np.sqrt(
            (temp - 440.0) ** 2 + ((time - 1.0) * p['time_distance_scale_short']) ** 2
        )

        cols['inv_dist_470_12'] = 1.0 / (1.0 + cols['dist_to_470_12'])
        cols['inv_dist_440_24'] = 1.0 / (1.0 + cols['dist_to_440_24'])
        cols['inv_dist_440_1'] = 1.0 / (1.0 + cols['dist_to_440_1'])

        cols['combo_gap_470_12'] = (
            cols['zone_470_12'] * cols['inside_train_temp_range'] * cols['inside_train_time_range']
        )
        cols['combo_gap_440_24'] = (
            cols['zone_440_24'] * cols['inside_train_temp_range'] * cols['inside_train_time_range']
        )
        cols['combo_gap_440_1'] = (
            cols['zone_440_1'] * cols['inside_train_temp_range'] * cols['inside_train_time_range']
        )

        temp_span = max(p['train_temp_max'] - p['train_temp_min'], 1e-8)
        time_span = max(p['train_time_max'] - p['train_time_min'], 1e-8)

        edge_temp_risk = np.minimum(
            (temp - p['train_temp_min']) / temp_span,
            (p['train_temp_max'] - temp) / temp_span
        )
        edge_time_risk = np.minimum(
            (time - p['train_time_min']) / time_span,
            (p['train_time_max'] - time) / time_span
        )
        cols['edge_risk_temp'] = 1.0 - np.clip(edge_temp_risk * 2.0, 0.0, 1.0)
        cols['edge_risk_time'] = 1.0 - np.clip(edge_time_risk * 2.0, 0.0, 1.0)
        cols['edge_risk_joint'] = cols['edge_risk_temp'] * cols['edge_risk_time']

        # 新增一个更直接的“组合外推风险”特征，仍只围绕高误差点
        cols['extrapolation_risk_470_12'] = cols['edge_risk_joint'] * cols['zone_470_12']
        cols['extrapolation_risk_440_edges'] = cols['edge_risk_joint'] * np.maximum(
            cols['zone_440_24'], cols['zone_440_1']
        )

        # ================= 6) 物理基线特征 =================
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []

        for t, tm in zip(temp, time):
            bs, bt, by = self.physics_baseline(t, tm)
            baseline_strain.append(bs)
            baseline_tensile.append(bt)
            baseline_yield.append(by)

        baseline_strain = np.array(baseline_strain)
        baseline_tensile = np.array(baseline_tensile)
        baseline_yield = np.array(baseline_yield)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # ================= 7) 相对基线 / 残差型输入 =================
        cols['baseline_tensile_x_highT'] = baseline_tensile * cols['is_high_temp']
        cols['baseline_yield_x_midlong'] = baseline_yield * cols['is_mid_long_time']

        cols['baseline_tensile_x_zone470_12'] = baseline_tensile * cols['zone_470_12']
        cols['baseline_yield_x_zone470_12'] = baseline_yield * cols['zone_470_12']
        cols['baseline_yield_x_zone440_24'] = baseline_yield * cols['zone_440_24']
        cols['baseline_strain_x_zone440_1'] = baseline_strain * cols['zone_440_1']

        cols['baseline_tensile_x_temp_rel_regime'] = (
            baseline_tensile * (temp - p['critical_temp_regime_shift']) / 10.0
        )
        cols['baseline_yield_x_time_rel_regime'] = (
            baseline_yield * (time - p['critical_time_regime_shift']) / 12.0
        )
        cols['baseline_strain_x_time_rel_regime'] = (
            baseline_strain * (time - p['critical_time_regime_shift']) / 12.0
        )

        cols['baseline_tensile_x_combo470_12'] = baseline_tensile * cols['combo_gap_470_12']
        cols['baseline_yield_x_combo470_12'] = baseline_yield * cols['combo_gap_470_12']
        cols['baseline_yield_x_combo440_24'] = baseline_yield * cols['combo_gap_440_24']

        # 新增极少量、有明确物理含义的残差引导
        cols['baseline_tensile_x_highT_midtime'] = baseline_tensile * cols['highT_midtime_synergy']
        cols['baseline_yield_x_highT_midtime'] = baseline_yield * cols['highT_midtime_synergy']

        # ================= 8) 少量局部归一化特征 =================
        cols['temp_norm_local'] = (temp - 440.0) / 30.0
        cols['time_norm_local'] = log_time / np.log1p(24.0)
        cols['combined_exposure_simple'] = (
            0.40 * cols['temp_norm_local'] +
            0.28 * cols['time_norm_local'] +
            0.32 * arr_prog
        )

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names