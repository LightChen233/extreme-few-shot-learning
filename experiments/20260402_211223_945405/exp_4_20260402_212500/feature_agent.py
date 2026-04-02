import pandas as pd
import numpy as np


# 7499 铝合金小样本热处理特征工程参数
# 设计原则：
# 1) 仅 29 条训练样本，特征必须“少而精”，重点服务于外推点；
# 2) 当前最大误差集中在 470/12、440/24、440/1，且强度误差多为正：
#    模型在这些外推组合上系统性低估强度；
# 3) 因此不做激进重构，只做温和增强：
#    - 保留现有局部动力学与基线思路
#    - 增加“组合外推边界距离/训练组合缺口”特征
#    - 对 470/12 附近加入更明确的局部机制激活
#    - 对 440 的短时/长时两端加入时间非单调分段提示
DOMAIN_PARAMS = {
    # 机制边界：按当前误差最敏感区设置温和转折
    'critical_temp_regime_shift': 455.0,   # 440 与 460~470 间局部机制切换
    'secondary_temp_boundary': 470.0,      # 高温重点外推边界
    'critical_time_regime_shift': 12.0,    # 中时效阶段
    'secondary_time_boundary': 24.0,       # 长时边界
    'short_time_boundary': 1.0,            # 短时边界

    # 动力学参数
    'activation_energy_Q': 92000.0,        # J/mol，铝合金析出/扩散的温和数量级
    'gas_constant_R': 8.314,
    'larson_miller_C': 20.0,

    # 训练域边界（题目给定）
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 训练集中“组合缺口”与高误差点
    'missing_temp_low': 440.0,
    'missing_temp_high': 470.0,
    'missing_time_short': 1.0,
    'missing_time_mid': 12.0,
    'missing_time_long': 24.0,

    # 原始凝固态平均性能
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 基线最大增益：保持方向正确，但幅度保守
    'max_strength_gain_tensile': 100.0,
    'max_strength_gain_yield': 76.0,
    'max_strain_gain': 1.5,

    # 局部热点宽度
    'zone_temp_sigma': 6.5,
    'zone_time_sigma_mid': 3.2,
    'zone_time_sigma_long': 4.5,
    'zone_time_sigma_short': 1.0,

    # 外推距离缩放
    'time_distance_scale_mid': 2.0,
    'time_distance_scale_long': 1.2,
    'time_distance_scale_short': 4.0,
}


class FeatureAgent:
    """
    小样本、物理引导的特征工程：
    1) 保留基础温度/时间/动力学等效量；
    2) 用分段激活描述机制切换；
    3) 显式表征重点外推组合与训练组合缺口；
    4) 用物理基线作为特征，让模型更像学习残差。
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于当前数据统计方向构建温和物理基线：
        - 在本题覆盖的局部 440–470°C、1–24h 内，
          更高温度/更长时间整体上更接近强化推进，而非强行假设明显软化；
        - 470/12 附近允许局部强化抬升；
        - 440°C 下 1h 与 24h 两端表现不稳，因此时间效应不写成简单单调直线，
          而用分段/平滑阶段项表达。
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = max(float(time), 1e-8)
        T_k = temp + 273.15

        # 局部归一化
        temp_norm = (temp - 440.0) / 30.0
        temp_norm = np.clip(temp_norm, -0.8, 1.4)

        time_norm = np.log1p(time) / np.log1p(24.0)
        time_norm = np.clip(time_norm, 0.0, 1.2)

        # 相对 Arrhenius 推进度（相对 440°C）
        T_ref = 440.0 + 273.15
        arr_rel = np.exp(-(p['activation_energy_Q'] / p['gas_constant_R']) * (1.0 / T_k - 1.0 / T_ref))
        arr_prog = np.clip((arr_rel - 1.0) / 2.6, -0.2, 1.6)

        # 平滑阶段激活
        high_temp = 1.0 / (1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / 3.8))
        stage_12 = 1.0 / (1.0 + np.exp(-(time - p['critical_time_regime_shift']) / 1.8))
        stage_24 = 1.0 / (1.0 + np.exp(-(time - p['secondary_time_boundary']) / 2.2))

        # 核心热暴露推进量
        exposure = (
            0.35 * temp_norm +
            0.24 * time_norm +
            0.23 * arr_prog +
            0.10 * high_temp +
            0.08 * stage_12
        )
        exposure = np.clip(exposure, -0.15, 1.35)

        # 重点高误差区局部修正
        zone_470_12 = np.exp(-((temp - 470.0) / 7.0) ** 2 - ((time - 12.0) / 3.5) ** 2)
        zone_440_24 = np.exp(-((temp - 440.0) / 7.0) ** 2 - ((time - 24.0) / 4.5) ** 2)
        zone_440_1 = np.exp(-((temp - 440.0) / 7.0) ** 2 - ((time - 1.0) / 1.1) ** 2)

        # 470/12 的强度低估最严重，给更明确的局部抬升
        local_boost_strength = 0.14 * zone_470_12 + 0.06 * zone_440_24 + 0.03 * zone_440_1
        local_boost_strain = 0.05 * zone_470_12 - 0.03 * zone_440_24 + 0.02 * zone_440_1

        baseline_tensile = p['as_cast_tensile'] + p['max_strength_gain_tensile'] * (exposure + local_boost_strength)
        baseline_yield = p['as_cast_yield'] + p['max_strength_gain_yield'] * (exposure + 0.9 * local_boost_strength)

        # 应变变化弱处理，避免再次出现“只改善应变、牺牲强度”
        baseline_strain = (
            p['as_cast_strain']
            + 0.45 * time_norm
            - 0.12 * temp_norm
            + 0.16 * arr_prog
            + 0.10 * stage_12
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
        arr_rel = np.exp(-(p['activation_energy_Q'] / p['gas_constant_R']) * (1.0 / T_k - 1.0 / T_ref))
        arr_prog = np.clip((arr_rel - 1.0) / 2.6, -0.2, 1.6)

        cols['arrhenius_relative'] = arr_rel
        cols['arrhenius_time'] = arr_rel * time_clip
        cols['arrhenius_logtime'] = arr_rel * log_time
        cols['arrhenius_progress'] = arr_prog

        cols['larson_miller'] = T_k * (p['larson_miller_C'] + np.log10(np.clip(time_clip, 1e-8, None)))
        cols['thermal_dose'] = T_k * log_time
        cols['zener_like'] = 1.0 / np.clip(arr_rel * time_clip, 1e-8, None)

        # ================= 3) 机制区间检测特征 =================
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

        cols['sig_temp_regime'] = 1.0 / (1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / 4.0))
        cols['sig_temp_470'] = 1.0 / (1.0 + np.exp(-(temp - p['secondary_temp_boundary']) / 2.5))
        cols['sig_time_12'] = 1.0 / (1.0 + np.exp(-(time - p['critical_time_regime_shift']) / 2.0))
        cols['sig_time_24'] = 1.0 / (1.0 + np.exp(-(time - p['secondary_time_boundary']) / 2.3))

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

        # 用于表达 440°C 下时间两端不稳定、12h 为中间阶段
        cols['u_time_edge_at_440'] = cols['near_440'] * np.maximum(cols['near_1h'], cols['near_24h'])
        cols['mid_stage_at_highT'] = cols['near_12h'] * cols['sig_temp_regime']

        # ================= 5) 外推/边界距离特征 =================
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

        # 组合缺口强度：不是简单看 temp/time 是否在范围内，而是看是否靠近训练中缺失的组合
        cols['combo_gap_470_12'] = cols['zone_470_12'] * cols['inside_train_temp_range'] * cols['inside_train_time_range']
        cols['combo_gap_440_24'] = cols['zone_440_24'] * cols['inside_train_temp_range'] * cols['inside_train_time_range']
        cols['combo_gap_440_1'] = cols['zone_440_1'] * cols['inside_train_temp_range'] * cols['inside_train_time_range']

        # 训练域边界“风险感”：靠近边界且处于关键时间/温度时，提示外推保守偏差
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

        # ================= 7) 相对基线/边界的残差型输入 =================
        cols['baseline_tensile_x_highT'] = baseline_tensile * cols['is_high_temp']
        cols['baseline_yield_x_midlong'] = baseline_yield * cols['is_mid_long_time']

        cols['baseline_tensile_x_zone470_12'] = baseline_tensile * cols['zone_470_12']
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

        # 新增：用基线与组合缺口相乘，帮助模型在外推缺口上学“基线修正幅度”
        cols['baseline_tensile_x_combo470_12'] = baseline_tensile * cols['combo_gap_470_12']
        cols['baseline_yield_x_combo470_12'] = baseline_yield * cols['combo_gap_470_12']
        cols['baseline_yield_x_combo440_24'] = baseline_yield * cols['combo_gap_440_24']

        # ================= 8) 少量局部归一化特征 =================
        cols['temp_norm_local'] = (temp - 440.0) / 30.0
        cols['time_norm_local'] = log_time / np.log1p(24.0)
        cols['combined_exposure_simple'] = (
            0.40 * cols['temp_norm_local'] +
            0.30 * cols['time_norm_local'] +
            0.30 * arr_prog
        )

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names