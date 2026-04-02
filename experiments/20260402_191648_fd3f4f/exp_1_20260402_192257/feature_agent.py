import pandas as pd
import numpy as np


# 7499 铝合金在 420–480°C、1–24h 区间内更接近“固溶/均匀化增强”主导区，
# 数据统计显示：升温、延时总体提升强度，但在约 460–470°C、12h 左右可能出现机制切换/最优窗口。
DOMAIN_PARAMS = {
    # 机制边界 / regime shift
    'critical_temp_regime_shift': 460.0,      # 高响应区起点
    'secondary_temp_peak_window': 470.0,      # 可能的最优窗口中心
    'critical_time_regime_shift': 12.0,       # 短时/长时机制切换
    'long_time_threshold': 24.0,              # 长时饱和边界
    'low_temp_threshold': 440.0,              # 低温固溶不足边界

    # 动力学参数（经验量级，作为特征构造用，不追求严格材料常数精确性）
    'activation_energy_Q': 135000.0,          # J/mol，铝合金溶解/扩散过程量级
    'gas_constant_R': 8.314,                  # J/mol/K
    'larson_miller_C': 20.0,                  # 常见经验常数

    # 原始铸态基准
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 当前工艺窗口，用于归一化与相对位置特征
    'temp_min': 420.0,
    'temp_max': 480.0,
    'time_min': 1.0,
    'time_max': 24.0,
}


class FeatureAgent:
    """
    面向小样本（29条）的物理启发特征工程：
    1) 少而精，避免过多无约束高维项
    2) 强化温度-时间耦合、机制分段、动力学等效量
    3) 使用 physics_baseline 作为残差学习锚点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的近似基线：
        - 在当前区间内，升温/延时总体提高强度
        - 应变整体改善，但在高温长时可能不再单调增大
        - 460–470°C、~12h 附近存在较优窗口
        """
        t = float(temp)
        h = float(time)

        p = DOMAIN_PARAMS
        Tn = (t - p['temp_min']) / (p['temp_max'] - p['temp_min'])   # 0~1
        Hn = np.log1p(h) / np.log1p(p['time_max'])                   # 0~1，体现时间边际递减

        # 等效处理程度：温度权重大于时间，符合当前数据趋势
        dose = 0.65 * Tn + 0.35 * Hn

        # 最优窗口增强项：470°C/12h 附近常见较高性能，但不做过强假设
        temp_peak = np.exp(-((t - p['secondary_temp_peak_window']) / 12.0) ** 2)
        time_peak = np.exp(-((np.log1p(h) - np.log1p(p['critical_time_regime_shift'])) / 0.7) ** 2)
        peak_boost = temp_peak * time_peak

        # 高温长时下塑性可能回落；强度则总体仍维持高位/小幅上升
        hot_long = max(0.0, Tn - 0.75) * max(0.0, Hn - 0.75)

        # 强度：相对铸态显著提升，方向与数据一致
        baseline_tensile = (
            p['as_cast_tensile']
            + 250.0 * dose
            + 22.0 * peak_boost
        )

        baseline_yield = (
            p['as_cast_yield']
            + 165.0 * dose
            + 16.0 * peak_boost
        )

        # 应变：总体改善，但高温长时可能从峰值回落
        baseline_strain = (
            p['as_cast_strain']
            - 2.5
            + 7.5 * dose
            + 2.0 * peak_boost
            - 3.0 * hot_long
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        cols = {}
        p = DOMAIN_PARAMS

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        T_K = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ---------- 基础低维特征 ----------
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_centered'] = temp - p['critical_temp_regime_shift']
        cols['time_centered'] = time - p['critical_time_regime_shift']

        # 小样本下保留必要二次项，不做全量堆砌
        cols['temp_sq_scaled'] = ((temp - 450.0) / 30.0) ** 2
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['log_time_sq'] = log_time ** 2

        # ---------- 温度-时间耦合 / 等效热处理量 ----------
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_x_time_scaled'] = temp * time / 100.0
        cols['dose_linear'] = (
            0.65 * (temp - p['temp_min']) / (p['temp_max'] - p['temp_min'])
            + 0.35 * log_time / np.log1p(p['time_max'])
        )
        cols['dose_relative_to_460C_12h'] = (
            ((temp - p['critical_temp_regime_shift']) / 10.0)
            * (log_time - np.log1p(p['critical_time_regime_shift']))
        )

        # ---------- 动力学等效特征 ----------
        # Arrhenius 型速率项
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_K))
        cols['arrhenius_rate'] = arrhenius
        cols['arrhenius_dose'] = time * arrhenius
        cols['log_arrhenius_dose'] = np.log1p(cols['arrhenius_dose'])

        # Larson-Miller Parameter（尽管更常用于蠕变，但可作热暴露等效量）
        cols['larson_miller'] = T_K * (p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None)))

        # 扩散/热激活常见简化形式
        cols['inv_temp_K'] = 1.0 / T_K
        cols['log_time_over_T'] = log_time / T_K
        cols['time_over_T'] = time / T_K

        # ---------- 机制区间检测特征（关键） ----------
        cols['is_low_temp'] = (temp <= p['low_temp_threshold']).astype(float)
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_peak_temp_regime'] = (temp >= 470.0).astype(float)
        cols['is_long_time_regime'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_very_long_time'] = (time >= 24.0).astype(float)

        cols['is_lowT_shortt'] = ((temp <= 440.0) & (time <= 1.0)).astype(float)
        cols['is_lowT_longt'] = ((temp <= 440.0) & (time >= 12.0)).astype(float)
        cols['is_highT_shortt'] = ((temp >= 460.0) & (time <= 1.0)).astype(float)
        cols['is_highT_longt'] = ((temp >= 460.0) & (time >= 12.0)).astype(float)
        cols['is_peak_window_470_12'] = ((temp >= 470.0) & (time >= 12.0)).astype(float)

        # 分段激活：帮助模型表达机制切换后的斜率变化
        cols['temp_above_460'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_above_470'] = np.clip(temp - 470.0, 0, None)
        cols['time_above_12'] = np.clip(time - p['critical_time_regime_shift'], 0, None)
        cols['time_above_24'] = np.clip(time - p['long_time_threshold'], 0, None)

        cols['highT_longt_synergy'] = cols['temp_above_460'] * np.clip(log_time - np.log1p(12.0), 0, None)
        cols['lowT_penalty_index'] = np.clip(440.0 - temp, 0, None) * (1.0 / (1.0 + time))

        # ---------- 平滑窗口特征 ----------
        temp_peak = np.exp(-((temp - p['secondary_temp_peak_window']) / 10.0) ** 2)
        time_peak = np.exp(-((log_time - np.log1p(p['critical_time_regime_shift'])) / 0.6) ** 2)
        cols['peak_temp_proximity'] = temp_peak
        cols['peak_time_proximity'] = time_peak
        cols['peak_window_proximity'] = temp_peak * time_peak

        # ---------- 相对边界位置特征 ----------
        cols['temp_relative_to_boundary'] = (temp - p['critical_temp_regime_shift']) / 20.0
        cols['time_relative_to_boundary'] = (log_time - np.log1p(p['critical_time_regime_shift'])) / np.log1p(p['time_max'])
        cols['distance_from_peak_window'] = np.sqrt(
            ((temp - p['secondary_temp_peak_window']) / 10.0) ** 2 +
            ((log_time - np.log1p(p['critical_time_regime_shift'])) / 0.6) ** 2
        )

        # ---------- 基线预测特征：让模型学习残差 ----------
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []
        for t, h in zip(temp, time):
            s0, ts0, ys0 = self.physics_baseline(t, h)
            baseline_strain.append(s0)
            baseline_tensile.append(ts0)
            baseline_yield.append(ys0)

        baseline_strain = np.array(baseline_strain, dtype=float)
        baseline_tensile = np.array(baseline_tensile, dtype=float)
        baseline_yield = np.array(baseline_yield, dtype=float)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 基线派生：表示相对铸态提升幅度与输出间结构关系
        cols['baseline_tensile_gain_over_cast'] = baseline_tensile - p['as_cast_tensile']
        cols['baseline_yield_gain_over_cast'] = baseline_yield - p['as_cast_yield']
        cols['baseline_strain_gain_over_cast'] = baseline_strain - p['as_cast_strain']
        cols['baseline_yield_tensile_ratio'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)
        cols['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield

        # 用工艺输入对基线进行门控，利于学习误差最大的特殊工艺点
        cols['baseline_tensile_x_peak'] = baseline_tensile * cols['peak_window_proximity']
        cols['baseline_yield_x_highTlongt'] = baseline_yield * cols['is_highT_longt']
        cols['baseline_strain_x_hotlong'] = baseline_strain * cols['is_peak_window_470_12']

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names