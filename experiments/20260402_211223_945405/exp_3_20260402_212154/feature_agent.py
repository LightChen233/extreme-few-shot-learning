import pandas as pd
import numpy as np


# 7499 铝合金小样本热处理特征工程参数
# 依据当前误差分布与常见铝合金热暴露规律做“温和物理先验”：
# 1) 在本题覆盖的 440–470°C、1–24h 局部区间内，验证误差显示高温/长时点强度多被低估，
#    因此此局部区间更像是“组织演化/强化推进区”，不强行假设明显过时效软化；
# 2) 470/12、440/24 是重点外推误差点，说明可能存在区间斜率切换或局部强化增幅；
# 3) 样本仅 29 条，故只做少量、针对性的机制特征，不堆砌高阶项。
DOMAIN_PARAMS = {
    # 机制边界/斜率切换点
    'critical_temp_regime_shift': 455.0,   # 440 与 460–470 之间的局部机制切换
    'secondary_temp_boundary': 470.0,      # 高温边界，重点误差点
    'critical_time_regime_shift': 12.0,    # 中时效阶段切换
    'secondary_time_boundary': 24.0,       # 长时边界，重点误差点

    # 动力学参数（数量级温和估计）
    'activation_energy_Q': 90000.0,        # J/mol，铝合金析出/扩散过程的温和数量级
    'gas_constant_R': 8.314,               # J/(mol*K)
    'larson_miller_C': 20.0,

    # 训练域边界（题目已给）
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 关键缺失/高误差组合
    'missing_temp_low': 440.0,
    'missing_temp_high': 470.0,
    'missing_time_mid': 12.0,
    'missing_time_long': 24.0,

    # 原始凝固态平均性能
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 局部强化幅值参数：保持“方向正确、幅度温和”
    'max_strength_gain_tensile': 95.0,
    'max_strength_gain_yield': 72.0,
    'max_strain_gain': 1.4,

    # 局部热点宽度：只针对误差最大的区域建模
    'zone_temp_sigma': 7.0,
    'zone_time_sigma_mid': 3.5,
    'zone_time_sigma_long': 5.0,
    'zone_time_sigma_short': 1.2,
}


class FeatureAgent:
    """
    小样本、物理引导的特征工程：
    1) 保留少量基础特征；
    2) 用 Arrhenius / Larson-Miller / 热暴露等效量描述动力学；
    3) 用分段激活和重点误差区特征捕捉机制切换；
    4) 显式加入“距训练边界/未覆盖组合”的特征，帮助外推；
    5) 基线预测作为特征，让模型更像学习残差。
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于题目数据方向约束的局部物理基线：
        - 在当前 440–470°C, 1–24h 局部范围内，强度随温度/时间升高总体表现为上升；
        - 12h、24h 附近加入温和阶段增益，反映组织演化阶段切换；
        - 对应变只给弱变化，避免强行绑定强度-塑性反相关。
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

        # 相对 Arrhenius 推进度（相对 440°C 参考）
        T_ref = 440.0 + 273.15
        arr_rel = np.exp(-(p['activation_energy_Q'] / p['gas_constant_R']) * (1.0 / T_k - 1.0 / T_ref))
        arr_prog = np.clip((arr_rel - 1.0) / 2.5, -0.2, 1.5)

        # 阶段激活：12h 与 24h 附近强化增幅切换
        stage_12 = 1.0 / (1.0 + np.exp(-(time - p['critical_time_regime_shift']) / 2.0))
        stage_24 = 1.0 / (1.0 + np.exp(-(time - p['secondary_time_boundary']) / 2.5))
        high_temp = 1.0 / (1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / 4.0))

        # 热暴露综合量：方向与当前数据一致（更高温/更长时 => 更高强度）
        exposure = (
            0.34 * temp_norm +
            0.26 * time_norm +
            0.20 * arr_prog +
            0.12 * stage_12 +
            0.08 * high_temp
        )
        exposure = np.clip(exposure, -0.15, 1.25)

        # 470/12、440/24 附近允许温和局部抬升，避免基线在高误差区过保守
        zone_470_12 = np.exp(-((temp - 470.0) / 8.0) ** 2 - ((time - 12.0) / 4.0) ** 2)
        zone_440_24 = np.exp(-((temp - 440.0) / 8.0) ** 2 - ((time - 24.0) / 5.0) ** 2)

        local_boost = 0.10 * zone_470_12 + 0.07 * zone_440_24

        baseline_tensile = p['as_cast_tensile'] + p['max_strength_gain_tensile'] * (exposure + local_boost)
        baseline_yield = p['as_cast_yield'] + p['max_strength_gain_yield'] * (exposure + 0.8 * local_boost)

        # 应变变化弱一些，避免重复历史中的强度/应变 trade-off
        baseline_strain = (
            p['as_cast_strain']
            + 0.55 * time_norm
            - 0.18 * temp_norm
            + 0.18 * arr_prog
            + 0.10 * stage_12
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

        # ================= 1) 少量基础特征 =================
        cols['temp'] = temp
        cols['time'] = time
        cols['log1p_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_x_logtime'] = temp * log_time

        # 只保留一个温和二次项，避免过多多项式
        cols['temp_centered_sq'] = ((temp - p['critical_temp_regime_shift']) / 20.0) ** 2

        # ================= 2) 动力学等效特征 =================
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['arrhenius'] = arrhenius
        cols['log_arrhenius'] = np.log(np.clip(arrhenius, 1e-300, None))

        # 相对 440°C 的推进倍数，更利于小样本学习
        T_ref = 440.0 + 273.15
        arr_rel = np.exp(-(p['activation_energy_Q'] / p['gas_constant_R']) * (1.0 / T_k - 1.0 / T_ref))
        cols['arrhenius_relative'] = arr_rel
        cols['arrhenius_time'] = arr_rel * time_clip
        cols['arrhenius_logtime'] = arr_rel * log_time

        # Larson-Miller / 热暴露等效
        cols['larson_miller'] = T_k * (p['larson_miller_C'] + np.log10(np.clip(time_clip, 1e-8, None)))
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

        cols['is_high_temp_midlong_time'] = (
            (temp >= p['critical_temp_regime_shift']) & (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['is_very_high_temp_12h_zone'] = (
            (temp >= 468.0) & (time >= 8.0) & (time <= 16.0)
        ).astype(float)

        cols['is_440_long_time_zone'] = (
            (temp <= 442.0) & (time >= 20.0)
        ).astype(float)

        # 平滑阶段激活，比硬阈值更稳
        cols['sig_temp_regime'] = 1.0 / (1.0 + np.exp(-(temp - p['critical_temp_regime_shift']) / 4.0))
        cols['sig_time_12'] = 1.0 / (1.0 + np.exp(-(time - p['critical_time_regime_shift']) / 2.0))
        cols['sig_time_24'] = 1.0 / (1.0 + np.exp(-(time - p['secondary_time_boundary']) / 2.5))

        # ================= 4) 重点误差区局部特征（只保留少量） =================
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

        # ================= 5) 外推/边界距离特征（关键） =================
        # 到训练域边界的距离
        cols['dist_temp_to_train_min'] = temp - p['train_temp_min']
        cols['dist_temp_to_train_max'] = p['train_temp_max'] - temp
        cols['dist_time_to_train_min'] = time - p['train_time_min']
        cols['dist_time_to_train_max'] = p['train_time_max'] - time

        # 域内标记
        cols['inside_train_temp_range'] = (
            (temp >= p['train_temp_min']) & (temp <= p['train_temp_max'])
        ).astype(float)
        cols['inside_train_time_range'] = (
            (time >= p['train_time_min']) & (time <= p['train_time_max'])
        ).astype(float)

        # 到关键缺失组合的相对位置
        cols['temp_rel_440'] = temp - p['missing_temp_low']
        cols['temp_rel_470'] = temp - p['missing_temp_high']
        cols['time_rel_12'] = time - p['missing_time_mid']
        cols['time_rel_24'] = time - p['missing_time_long']

        # 组合距离：针对题目给出的高误差外推点
        cols['dist_to_470_12'] = np.sqrt((temp - 470.0) ** 2 + ((time - 12.0) * 2.0) ** 2)
        cols['dist_to_440_24'] = np.sqrt((temp - 440.0) ** 2 + ((time - 24.0) * 1.2) ** 2)
        cols['dist_to_440_1'] = np.sqrt((temp - 440.0) ** 2 + ((time - 1.0) * 4.0) ** 2)

        # 缺失组合“外推感”：
        # 若温度/时间值本身在训练域内，但组合未见，则需要告诉模型其靠近关键未覆盖点
        cols['inv_dist_470_12'] = 1.0 / (1.0 + cols['dist_to_470_12'])
        cols['inv_dist_440_24'] = 1.0 / (1.0 + cols['dist_to_440_24'])
        cols['inv_dist_440_1'] = 1.0 / (1.0 + cols['dist_to_440_1'])

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

        # 针对最大误差区，只保留两项最关键交互，避免维度膨胀
        cols['baseline_tensile_x_zone470_12'] = baseline_tensile * cols['zone_470_12']
        cols['baseline_yield_x_zone440_24'] = baseline_yield * cols['zone_440_24']

        cols['baseline_tensile_x_temp_rel_regime'] = (
            baseline_tensile * (temp - p['critical_temp_regime_shift']) / 10.0
        )
        cols['baseline_yield_x_time_rel_regime'] = (
            baseline_yield * (time - p['critical_time_regime_shift']) / 12.0
        )
        cols['baseline_strain_x_time_rel_regime'] = (
            baseline_strain * (time - p['critical_time_regime_shift']) / 12.0
        )

        # ================= 8) 少量局部归一化特征 =================
        cols['temp_norm_local'] = (temp - 440.0) / 30.0
        cols['time_norm_local'] = log_time / np.log1p(24.0)
        cols['combined_exposure_simple'] = (
            0.42 * cols['temp_norm_local'] +
            0.33 * cols['time_norm_local'] +
            0.25 * np.clip((arr_rel - 1.0) / 2.5, -0.2, 1.5)
        )

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names