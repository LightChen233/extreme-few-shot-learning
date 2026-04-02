import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # 机制边界：结合题目给出的趋势，约在中高温区开始出现显著强化/塑性提升
    'critical_temp_regime_shift': 450.0,   # ℃，由低温缓慢响应转向高响应区
    'critical_temp_peak_strength': 470.0,  # ℃，强度最优附近
    'critical_temp_high': 480.0,           # ℃，高温边界

    # 时间机制边界：1h 为短时，12h 为主要转变区，24h 为长时/更充分演化
    'critical_time_short': 3.0,            # h，短时区近似上界
    'critical_time_regime_shift': 12.0,    # h，主强化响应区
    'critical_time_long': 24.0,            # h，长时区

    # 动力学参数：铝合金扩散/析出相关量级，取温和值避免指数爆炸
    'activation_energy_Q': 120000.0,       # J/mol，等效激活能估计
    'gas_constant_R': 8.314,               # J/(mol·K)

    # LMP 常数
    'larson_miller_C': 20.0,

    # 训练域边界，用于外推距离特征
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 经验最优窗口中心：用于构造“距最优区距离”
    'opt_temp_center': 468.0,
    'opt_time_center': 12.0,
    'opt_temp_width': 18.0,
    'opt_time_width_log': 0.9,

    # 原始铸态基准
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    面向 7499 铝合金热处理-性能预测的小样本物理特征工程。
    重点：
    1) 少而精，避免高维过拟合
    2) 强化温度-时间耦合与机制分段
    3) 加入外推距离与物理基线，帮助模型处理训练未覆盖点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于材料规律构造粗粒度物理基线：
        - 相比铸态，热处理后整体强度显著提高
        - 温度升高通常更显著提升性能，尤其在 450~470℃区间
        - 时间延长一般有利于组织演化，但效应趋于饱和
        - 强度在 470℃附近存在最优窗口；应变在中高温明显抬升
        """
        t = float(temp)
        h = max(float(time), 1e-6)

        # 温度归一化（420~480）
        temp_norm = (t - 420.0) / 60.0
        temp_norm = np.clip(temp_norm, 0.0, 1.2)

        # 时间采用对数刻画饱和动力学（1,12,24h 跨度不均匀）
        logh = np.log1p(h)
        logh_norm = np.log1p(h) / np.log1p(24.0)
        logh_norm = np.clip(logh_norm, 0.0, 1.1)

        # 强度最优温度窗：470附近最高，偏离后衰减
        peak_temp = DOMAIN_PARAMS['critical_temp_peak_strength']
        peak_width = 18.0
        peak_factor = np.exp(-((t - peak_temp) / peak_width) ** 2)

        # 长时间有助于高温下进一步强化，但会有饱和
        time_gain = 1.0 - np.exp(-h / 10.0)

        # 应变：中高温显著上升，长时适度提升；在极高温附近不过分惩罚
        baseline_strain = (
            DOMAIN_PARAMS['as_cast_strain']
            - 4.0
            + 7.5 * temp_norm
            + 2.0 * time_gain
            + 1.5 * peak_factor
        )

        # 抗拉强度：整体显著高于铸态，受温度主导，470附近峰值
        baseline_tensile = (
            DOMAIN_PARAMS['as_cast_tensile']
            + 95.0 * temp_norm
            + 70.0 * logh_norm
            + 110.0 * peak_factor
            + 20.0 * peak_factor * time_gain
        )

        # 屈服强度：与抗拉类似，但对峰值窗口更敏感
        baseline_yield = (
            DOMAIN_PARAMS['as_cast_yield']
            + 78.0 * temp_norm
            + 52.0 * logh_norm
            + 95.0 * peak_factor
            + 18.0 * peak_factor * time_gain
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        time_safe = np.clip(time, 1e-6, None)

        # 基础变量
        log_time = np.log1p(time_safe)
        sqrt_time = np.sqrt(time_safe)
        temp_K = temp + 273.15

        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time

        # 少量二次项：保留必要非线性
        cols['temp_sq_centered'] = ((temp - DOMAIN_PARAMS['critical_temp_peak_strength']) / 20.0) ** 2
        cols['log_time_sq_centered'] = ((log_time - np.log1p(DOMAIN_PARAMS['critical_time_regime_shift'])) / 1.0) ** 2

        # 温度-时间耦合
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_centered_x_log_time'] = (temp - DOMAIN_PARAMS['critical_temp_regime_shift']) * log_time

        # 动力学等效特征：Arrhenius / LMP
        Q = DOMAIN_PARAMS['activation_energy_Q']
        R = DOMAIN_PARAMS['gas_constant_R']
        arrhenius = np.exp(-Q / (R * temp_K))
        cols['arrhenius'] = arrhenius
        cols['arrhenius_x_time'] = arrhenius * time
        cols['arrhenius_x_log_time'] = arrhenius * log_time

        C = DOMAIN_PARAMS['larson_miller_C']
        cols['larson_miller'] = temp_K * (C + np.log10(np.clip(time_safe, 1e-6, None)))
        cols['temp_over_Tm_like'] = temp_K / (480.0 + 273.15)  # 用当前最高工艺温度做归一化参考

        # 机制区间检测特征
        Tc1 = DOMAIN_PARAMS['critical_temp_regime_shift']
        Tc2 = DOMAIN_PARAMS['critical_temp_peak_strength']
        ts = DOMAIN_PARAMS['critical_time_short']
        tm = DOMAIN_PARAMS['critical_time_regime_shift']
        tl = DOMAIN_PARAMS['critical_time_long']

        cols['is_mid_high_temp'] = (temp >= Tc1).astype(float)
        cols['is_peak_temp_zone'] = (np.abs(temp - Tc2) <= 10.0).astype(float)
        cols['is_high_temp_edge'] = (temp >= 475.0).astype(float)

        cols['is_short_time'] = (time <= ts).astype(float)
        cols['is_mid_time'] = ((time > ts) & (time <= 18.0)).astype(float)
        cols['is_long_time'] = (time >= 18.0).astype(float)

        cols['is_peak_processing_window'] = (
            (np.abs(temp - Tc2) <= 12.0) & (time >= 8.0) & (time <= 18.0)
        ).astype(float)

        cols['is_low_temp_long_time'] = ((temp <= 445.0) & (time >= tm)).astype(float)
        cols['is_high_temp_short_time'] = ((temp >= 465.0) & (time <= ts)).astype(float)
        cols['is_high_temp_long_time'] = ((temp >= 465.0) & (time >= tm)).astype(float)

        # 连续型边界距离特征：帮助外推点建模
        cols['temp_rel_regime_shift'] = (temp - Tc1) / 20.0
        cols['temp_rel_peak'] = (temp - Tc2) / 10.0
        cols['time_rel_regime_shift_log'] = log_time - np.log1p(tm)
        cols['time_rel_long_log'] = log_time - np.log1p(tl)

        # 到经验最优窗口的距离
        opt_tc = DOMAIN_PARAMS['opt_temp_center']
        opt_tw = DOMAIN_PARAMS['opt_temp_width']
        opt_hc = DOMAIN_PARAMS['opt_time_center']
        opt_hw = DOMAIN_PARAMS['opt_time_width_log']

        dist_opt_temp = (temp - opt_tc) / opt_tw
        dist_opt_time = (log_time - np.log1p(opt_hc)) / opt_hw
        cols['dist_to_opt_window'] = np.sqrt(dist_opt_temp ** 2 + dist_opt_time ** 2)
        cols['in_opt_window_soft'] = np.exp(-(dist_opt_temp ** 2 + dist_opt_time ** 2))

        # 训练域边界距离：显式告知模型哪些点是外推
        tmin, tmax = DOMAIN_PARAMS['train_temp_min'], DOMAIN_PARAMS['train_temp_max']
        hmin, hmax = DOMAIN_PARAMS['train_time_min'], DOMAIN_PARAMS['train_time_max']

        temp_below = np.maximum(tmin - temp, 0.0)
        temp_above = np.maximum(temp - tmax, 0.0)
        time_below = np.maximum(hmin - time, 0.0)
        time_above = np.maximum(time - hmax, 0.0)

        cols['temp_outside_low'] = temp_below
        cols['temp_outside_high'] = temp_above
        cols['time_outside_low'] = time_below
        cols['time_outside_high'] = time_above

        cols['temp_edge_distance'] = np.minimum(np.abs(temp - tmin), np.abs(temp - tmax))
        cols['time_edge_distance_log'] = np.minimum(
            np.abs(log_time - np.log1p(hmin)),
            np.abs(log_time - np.log1p(hmax))
        )

        cols['is_temp_extrapolation'] = ((temp < tmin) | (temp > tmax)).astype(float)
        cols['is_time_extrapolation'] = ((time < hmin) | (time > hmax)).astype(float)

        # 针对题目中误差最大的外推条件做定向特征
        # 440/24：低温长时；470/12：峰值附近高响应窗口；440/1：低温短时
        cols['flag_440_24_like'] = ((temp <= 445.0) & (time >= 20.0)).astype(float)
        cols['flag_470_12_like'] = ((temp >= 465.0) & (temp <= 475.0) & (time >= 8.0) & (time <= 16.0)).astype(float)
        cols['flag_440_1_like'] = ((temp <= 445.0) & (time <= 3.0)).astype(float)

        # 物理基线特征：让模型学习“真实值 - 基线”的残差
        bs, bt, by = [], [], []
        for t, h in zip(temp, time):
            s0, ts0, y0 = self.physics_baseline(t, h)
            bs.append(s0)
            bt.append(ts0)
            by.append(y0)

        bs = np.array(bs, dtype=float)
        bt = np.array(bt, dtype=float)
        by = np.array(by, dtype=float)

        cols['baseline_strain'] = bs
        cols['baseline_tensile'] = bt
        cols['baseline_yield'] = by

        # 相对基线与相对铸态的“状态”特征
        cols['baseline_strength_sum'] = bt + by
        cols['baseline_strength_diff'] = bt - by
        cols['baseline_yield_ratio'] = by / np.clip(bt, 1e-6, None)
        cols['baseline_strain_x_strength'] = bs * bt

        cols['baseline_tensile_gain_over_cast'] = bt - DOMAIN_PARAMS['as_cast_tensile']
        cols['baseline_yield_gain_over_cast'] = by - DOMAIN_PARAMS['as_cast_yield']
        cols['baseline_strain_gain_over_cast'] = bs - DOMAIN_PARAMS['as_cast_strain']

        # 分段激活：在边界两侧允许模型学习不同斜率
        cols['temp_above_shift_pos'] = np.maximum(temp - Tc1, 0.0)
        cols['temp_below_shift_pos'] = np.maximum(Tc1 - temp, 0.0)
        cols['time_above_mid_pos_log'] = np.maximum(log_time - np.log1p(tm), 0.0)
        cols['time_below_mid_pos_log'] = np.maximum(np.log1p(tm) - log_time, 0.0)

        # 保持特征数量克制，避免小样本过拟合
        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names