import pandas as pd
import numpy as np


# 7499铝合金在当前给定区间更像“固溶/均匀化热暴露”主导，而非典型低温时效。
# 因此物理上应采用：温度升高、时间延长 -> 整体固溶/均匀化程度提高 -> 强度总体升高；
# 但在约460~470°C、12h附近可能出现机制切换/敏感区，表现为局部非线性和样本误差集中。
DOMAIN_PARAMS = {
    # 机制边界：结合题目误差集中区与给定趋势做出的物理先验
    'critical_temp_regime_shift': 460.0,     # 高温强化/快速动力学启动区
    'secondary_temp_boundary': 470.0,        # 更强敏感区，误差集中
    'critical_time_regime_shift': 12.0,      # 中长时处理的机制分界
    'long_time_boundary': 24.0,              # 长时端点

    # 动力学参数：铝合金中溶质扩散/相溶解的量级估计，用于构造相对动力学特征
    'activation_energy_Q': 135000.0,         # J/mol，数量级合理即可，主要用于特征映射
    'gas_constant_R': 8.314,                 # J/mol/K

    # Larson-Miller 类参数常数
    'lmp_C': 20.0,

    # 原始铸态基准性能
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 当前工艺窗口，用于归一化和相对位置特征
    'temp_min': 420.0,
    'temp_max': 480.0,
    'time_min': 1.0,
    'time_max': 24.0,
}


class FeatureAgent:
    """
    面向小样本材料工艺预测的物理启发特征工程。
    重点：
    1. 少而精，避免特征爆炸；
    2. 显式编码温度-时间耦合；
    3. 加入机制边界特征；
    4. 加入 physics baseline 作为残差学习锚点。
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的近似基线：
        - 在420–480°C、1–24h范围内，升温/延时总体提升强度；
        - 应变整体改善但非线性更强；
        - 在460°C以上和12h以上进入更强热激活区；
        - 在470°C、12h附近加入轻微的“敏感窗口”增益，帮助针对当前最大误差区建模残差。
        """
        t = float(temp)
        h = float(time)

        p = DOMAIN_PARAMS
        Tn = (t - p['temp_min']) / (p['temp_max'] - p['temp_min'])
        Tn = np.clip(Tn, 0.0, 1.0)

        ln_time = np.log1p(max(h, 0.0))
        ln_time_max = np.log1p(p['time_max'])
        Hn = ln_time / ln_time_max
        Hn = np.clip(Hn, 0.0, 1.0)

        # 等效热暴露：温度权重大于时间，符合数据趋势
        dose = 0.65 * Tn + 0.35 * Hn + 0.25 * Tn * Hn

        # 机制分段增益
        high_temp_gain = max(t - p['critical_temp_regime_shift'], 0.0) / (p['temp_max'] - p['critical_temp_regime_shift'])
        mid_long_time_gain = max(h - p['critical_time_regime_shift'], 0.0) / max(p['time_max'] - p['critical_time_regime_shift'], 1e-6)

        # 对470°C/12h附近的敏感区做平滑激活，而不是硬编码峰值
        temp_window = np.exp(-((t - 470.0) / 8.0) ** 2)
        time_window = np.exp(-((h - 12.0) / 6.0) ** 2)
        sensitive_window = temp_window * time_window

        # 强度：相对铸态显著提高，且高温/长时增强更明显
        baseline_tensile = (
            p['as_cast_tensile']
            + 170.0 * dose
            + 55.0 * high_temp_gain
            + 22.0 * mid_long_time_gain
            + 18.0 * sensitive_window
        )

        baseline_yield = (
            p['as_cast_yield']
            + 125.0 * dose
            + 40.0 * high_temp_gain
            + 18.0 * mid_long_time_gain
            + 10.0 * sensitive_window
        )

        # 应变：整体可改善，但幅度小于强度，且在中高温+适中时间更佳
        baseline_strain = (
            p['as_cast_strain']
            - 3.0
            + 3.5 * dose
            + 1.4 * sensitive_window
            + 0.4 * high_temp_gain
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        cols = {}

        # 只围绕 temp / time 做紧凑型物理特征，避免小样本过拟合
        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        p = DOMAIN_PARAMS
        T_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))

        # 基础归一化
        temp_norm = (temp - p['temp_min']) / (p['temp_max'] - p['temp_min'])
        temp_norm = np.clip(temp_norm, 0.0, 1.0)

        time_norm = log_time / np.log1p(p['time_max'])
        time_norm = np.clip(time_norm, 0.0, 1.0)

        cols['temp'] = temp
        cols['time'] = time
        cols['temp_norm'] = temp_norm
        cols['log_time'] = log_time
        cols['time_norm'] = time_norm

        # 少量非线性项
        cols['temp_sq_centered'] = ((temp - 450.0) / 30.0) ** 2
        cols['log_time_sq'] = log_time ** 2
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_norm_x_time_norm'] = temp_norm * time_norm

        # 动力学等效特征
        # 1) Arrhenius因子：扩散/溶解过程热激活尺度
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['arrhenius'] = arrhenius
        cols['time_x_arrhenius'] = time * arrhenius
        cols['log_time_x_arrhenius'] = log_time * arrhenius

        # 2) Larson-Miller-like parameter
        lmp = T_k * (p['lmp_C'] + np.log10(np.clip(time, 1e-6, None)))
        cols['lmp'] = lmp

        # 3) 等效热暴露/热剂量
        cols['thermal_dose_linear'] = temp * time
        cols['thermal_dose_log'] = temp * log_time
        cols['thermal_dose_excess'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None) * log_time

        # 机制边界检测特征（关键）
        cols['is_high_temp'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_very_high_temp'] = (temp >= p['secondary_temp_boundary']).astype(float)
        cols['is_mid_long_time'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_long_time'] = (time >= p['long_time_boundary']).astype(float)

        cols['is_high_temp_and_mid_long'] = (
            (temp >= p['critical_temp_regime_shift']) & (time >= p['critical_time_regime_shift'])
        ).astype(float)
        cols['is_very_high_temp_and_mid_long'] = (
            (temp >= p['secondary_temp_boundary']) & (time >= p['critical_time_regime_shift'])
        ).astype(float)

        # 相对边界距离：比纯布尔更平滑
        cols['temp_rel_critical'] = temp - p['critical_temp_regime_shift']
        cols['temp_rel_secondary'] = temp - p['secondary_temp_boundary']
        cols['time_rel_critical'] = time - p['critical_time_regime_shift']
        cols['temp_excess_pos'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['time_excess_pos'] = np.clip(time - p['critical_time_regime_shift'], 0, None)
        cols['regime_excess_product'] = cols['temp_excess_pos'] * cols['time_excess_pos']

        # 对误差集中区 470°C / 12h 附近做平滑窗口，不用过多局部特征
        cols['window_470_12'] = np.exp(-((temp - 470.0) / 8.0) ** 2 - ((time - 12.0) / 6.0) ** 2)
        cols['window_460_12'] = np.exp(-((temp - 460.0) / 8.0) ** 2 - ((time - 12.0) / 6.0) ** 2)
        cols['window_440_1'] = np.exp(-((temp - 440.0) / 8.0) ** 2 - ((time - 1.0) / 2.0) ** 2)
        cols['window_440_24'] = np.exp(-((temp - 440.0) / 8.0) ** 2 - ((time - 24.0) / 4.0) ** 2)

        # 物理基线预测值作为特征：让模型重点学习残差
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []
        for t, h in zip(temp, time):
            bs, bt, by = self.physics_baseline(t, h)
            baseline_strain.append(bs)
            baseline_tensile.append(bt)
            baseline_yield.append(by)

        baseline_strain = np.array(baseline_strain, dtype=float)
        baseline_tensile = np.array(baseline_tensile, dtype=float)
        baseline_yield = np.array(baseline_yield, dtype=float)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 相对铸态提升量：体现“处理增益”
        cols['baseline_strain_gain_vs_cast'] = baseline_strain - p['as_cast_strain']
        cols['baseline_tensile_gain_vs_cast'] = baseline_tensile - p['as_cast_tensile']
        cols['baseline_yield_gain_vs_cast'] = baseline_yield - p['as_cast_yield']

        # 基线归一化关系特征
        cols['baseline_yield_tensile_ratio'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)
        cols['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names