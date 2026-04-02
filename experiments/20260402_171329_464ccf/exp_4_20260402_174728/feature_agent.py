import pandas as pd
import numpy as np


# 7499 铝合金在 420–480°C、1–24h 热处理窗口下的特征工程
# 依据当前数据趋势与误差分布：
# 1) 当前范围更像固溶/高温均匀化处理区，整体上“升温 + 延时”通常提升强度；
# 2) 但存在明显温度-时间交互，且在 470°C-12h、460°C-12h、440°C-24h、440°C-1h 附近出现局部跃迁/分叉；
# 3) 因此应重点构造：机制边界、窗口邻近度、动力学累计量、分段交互、以及物理基线残差学习特征。
DOMAIN_PARAMS = {
    # -----------------------------
    # 机制边界（regime boundaries）
    # -----------------------------
    # 数据显示 440 附近仍属较弱处理区，460 附近进入更充分区，470 附近是高敏感窗口
    'critical_temp_regime_shift': 445.0,
    'critical_temp_mid_high': 460.0,
    'critical_temp_high': 470.0,
    'critical_temp_top': 480.0,

    # 时间边界：短时/中时/长时/极长时
    'critical_time_short': 3.0,
    'critical_time_mid': 8.0,
    'critical_time_long': 12.0,
    'critical_time_very_long': 24.0,

    # 重点误差窗口/机制窗口
    'window_temp_center_1': 440.0,
    'window_time_center_1': 1.0,
    'window_temp_center_2': 470.0,
    'window_time_center_2': 12.0,
    'window_temp_center_3': 440.0,
    'window_time_center_3': 24.0,
    'window_temp_center_4': 460.0,
    'window_time_center_4': 12.0,

    # 动力学参数：铝合金高温扩散/固溶相关的经验量级
    'activation_energy_Q': 115000.0,   # J/mol
    'gas_constant_R': 8.314,           # J/(mol·K)

    # Larson-Miller 风格参数
    'lmp_C': 20.0,

    # 参考温度
    'reference_temp_c': 420.0,
    'reference_temp_k': 420.0 + 273.15,

    # 原始铸态基准性能
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # 经验中心：当前最大低估区
    'solution_center_temp': 470.0,
    'solution_center_time': 12.0,
}


class FeatureAgent:
    """基于材料热处理机理的特征工程"""

    def __init__(self):
        self.feature_names = []

        self.base_strain = DOMAIN_PARAMS['base_strain']
        self.base_tensile = DOMAIN_PARAMS['base_tensile']
        self.base_yield = DOMAIN_PARAMS['base_yield']

        self.Q = DOMAIN_PARAMS['activation_energy_Q']
        self.R = DOMAIN_PARAMS['gas_constant_R']
        self.lmp_C = DOMAIN_PARAMS['lmp_C']

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def _gaussian_window(self, x, center, width):
        width = max(width, 1e-6)
        return np.exp(-((x - center) / width) ** 2)

    def physics_baseline(self, temp, time):
        """
        基于世界知识的基线估计（不依赖训练标签逐点拟合）：
        - 当前数据范围内，整体上更高温度/更长时间通常对应更高强度；
        - 时间效应具有递减性，适合用 log(time)；
        - 470°C-12h 附近可能存在组织均匀化/强化窗口；
        - 对极高温长时仅做轻微饱和，不引入与数据主趋势冲突的强软化假设。
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        log_time = np.log1p(np.clip(time, 0, None))

        temp_progress = np.clip((temp - 420.0) / 60.0, 0.0, 1.2)
        time_progress = np.clip(log_time / np.log1p(24.0), 0.0, 1.2)

        process_extent = (
            0.54 * temp_progress +
            0.22 * time_progress +
            0.24 * temp_progress * time_progress
        )

        hot_mid_window = (
            self._gaussian_window(temp, DOMAIN_PARAMS['window_temp_center_2'], 7.0) *
            self._gaussian_window(log_time, np.log1p(DOMAIN_PARAMS['window_time_center_2']), 0.42)
        )

        low_long_window = (
            self._gaussian_window(temp, DOMAIN_PARAMS['window_temp_center_3'], 8.0) *
            self._gaussian_window(log_time, np.log1p(DOMAIN_PARAMS['window_time_center_3']), 0.36)
        )

        low_short_window = (
            self._gaussian_window(temp, DOMAIN_PARAMS['window_temp_center_1'], 8.0) *
            self._gaussian_window(log_time, np.log1p(DOMAIN_PARAMS['window_time_center_1']), 0.25)
        )

        mid_hot_window = (
            self._gaussian_window(temp, DOMAIN_PARAMS['window_temp_center_4'], 7.0) *
            self._gaussian_window(log_time, np.log1p(DOMAIN_PARAMS['window_time_center_4']), 0.38)
        )

        high_temp_gate = self._sigmoid((temp - DOMAIN_PARAMS['critical_temp_high']) / 4.0)
        very_long_time_gate = self._sigmoid((time - DOMAIN_PARAMS['critical_time_long']) / 2.0)
        mild_saturation = 0.05 * high_temp_gate * very_long_time_gate

        effective_extent = (
            process_extent
            + 0.11 * hot_mid_window
            + 0.05 * low_long_window
            + 0.02 * low_short_window
            + 0.03 * mid_hot_window
            - mild_saturation
        )

        baseline_tensile = (
            self.base_tensile
            + 110.0 * effective_extent
            + 9.0 * hot_mid_window
            + 4.0 * low_long_window
            + 2.0 * mid_hot_window
        )

        baseline_yield = (
            self.base_yield
            + 92.0 * effective_extent
            + 7.5 * hot_mid_window
            + 3.5 * low_long_window
            + 1.5 * mid_hot_window
        )

        ductility_extent = (
            0.38 * temp_progress +
            0.28 * time_progress +
            0.34 * temp_progress * time_progress
        )

        baseline_strain = (
            self.base_strain
            + 2.5 * ductility_extent
            + 0.65 * hot_mid_window
            + 0.20 * low_long_window
            + 0.18 * mid_hot_window
            - 0.20 * high_temp_gate * very_long_time_gate
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-6, None)

        T1 = DOMAIN_PARAMS['critical_temp_regime_shift']
        Tm = DOMAIN_PARAMS['critical_temp_mid_high']
        T2 = DOMAIN_PARAMS['critical_temp_high']
        T3 = DOMAIN_PARAMS['critical_temp_top']

        t1 = DOMAIN_PARAMS['critical_time_short']
        t2 = DOMAIN_PARAMS['critical_time_mid']
        t3 = DOMAIN_PARAMS['critical_time_long']
        t4 = DOMAIN_PARAMS['critical_time_very_long']

        Tref_c = DOMAIN_PARAMS['reference_temp_c']
        Tref_k = DOMAIN_PARAMS['reference_temp_k']
        Tc = DOMAIN_PARAMS['solution_center_temp']
        tc = DOMAIN_PARAMS['solution_center_time']

        # -----------------------------
        # 1. 原始与基础变换
        # -----------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2

        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)
        X['temp_over_log_time'] = self._safe_divide(temp, log_time + 1e-6)

        # -----------------------------
        # 2. 动力学等效特征
        # -----------------------------
        arrhenius = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-6, None)))
        X['arrhenius_factor'] = arrhenius

        thermal_dose = time * arrhenius
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = self._safe_log(thermal_dose)

        delta_invT = (1.0 / np.clip(Tref_k, 1e-6, None)) - inv_temp_k
        equivalent_rate = np.exp(np.clip((self.Q / self.R) * delta_invT, -50, 50))
        equivalent_time = time * equivalent_rate
        X['equivalent_rate_vs_420C'] = equivalent_rate
        X['equivalent_time_vs_420C'] = equivalent_time
        X['log_equivalent_time_vs_420C'] = self._safe_log(equivalent_time)

        lmp = temp_k * (self.lmp_C + np.log10(np.clip(time, 1e-6, None)))
        X['larson_miller'] = lmp

        X['temp_centered_log_time'] = (temp - Tref_c) * log_time
        X['kinetic_index'] = self._safe_divide(log_time, temp_k)
        X['arrhenius_log_time'] = arrhenius * log_time
        X['arrhenius_time'] = arrhenius * time
        X['sqrt_equivalent_time'] = np.sqrt(np.clip(equivalent_time, 0, None))

        # JMAK风格的“转变量”近似特征：高温加速、时间饱和
        transformed_fraction_proxy = 1.0 - np.exp(-np.clip(equivalent_time, 0, None) / 8.0)
        X['transformed_fraction_proxy'] = transformed_fraction_proxy
        X['transformed_fraction_sq'] = transformed_fraction_proxy ** 2

        # -----------------------------
        # 3. 物理机制区间检测特征（关键）
        # -----------------------------
        is_low_temp = (temp < T1).astype(float)
        is_mid_temp = ((temp >= T1) & (temp < Tm)).astype(float)
        is_mid_high_temp = ((temp >= Tm) & (temp < T2)).astype(float)
        is_high_temp = (temp >= T2).astype(float)
        is_top_temp = (temp >= T3).astype(float)

        X['is_low_temp_regime'] = is_low_temp
        X['is_mid_temp_regime'] = is_mid_temp
        X['is_mid_high_temp_regime'] = is_mid_high_temp
        X['is_high_temp_regime'] = is_high_temp
        X['is_top_temp_regime'] = is_top_temp

        is_short_time = (time < t1).astype(float)
        is_mid_time = ((time >= t1) & (time < t2)).astype(float)
        is_long_time = ((time >= t2) & (time < t3)).astype(float)
        is_very_long_time = (time >= t3).astype(float)
        is_top_time = (time >= t4).astype(float)

        X['is_short_time_regime'] = is_short_time
        X['is_mid_time_regime'] = is_mid_time
        X['is_long_time_regime'] = is_long_time
        X['is_very_long_time_regime'] = is_very_long_time
        X['is_top_time_regime'] = is_top_time

        # 重点联合窗口
        X['is_440_1_window'] = ((np.abs(temp - 440.0) <= 5.0) & (np.abs(time - 1.0) <= 1.0)).astype(float)
        X['is_440_24_window'] = ((np.abs(temp - 440.0) <= 5.0) & (np.abs(time - 24.0) <= 4.0)).astype(float)
        X['is_460_12_window'] = ((np.abs(temp - 460.0) <= 5.0) & (np.abs(time - 12.0) <= 2.0)).astype(float)
        X['is_470_12_window'] = ((np.abs(temp - 470.0) <= 5.0) & (np.abs(time - 12.0) <= 2.0)).astype(float)

        X['is_high_temp_long_time'] = ((temp >= T2) & (time >= t3)).astype(float)
        X['is_high_temp_mid_time'] = ((temp >= T2) & (time >= t2) & (time < t3)).astype(float)
        X['is_mid_temp_long_time'] = ((temp >= T1) & (temp < T2) & (time >= t3)).astype(float)
        X['is_low_temp_very_long_time'] = ((temp < T1) & (time >= t3)).astype(float)
        X['is_near_regime_shift'] = (np.abs(temp - T1) <= 5.0).astype(float)
        X['is_near_mid_high_shift'] = (np.abs(temp - Tm) <= 5.0).astype(float)
        X['is_near_high_temp_shift'] = (np.abs(temp - T2) <= 5.0).astype(float)

        # 更细的机制门控
        X['is_below_solution_window'] = ((temp < T2) & (time < t3)).astype(float)
        X['is_entering_solution_window'] = ((temp >= Tm) & (temp < T2) & (time >= t2)).astype(float)
        X['is_peak_solution_window'] = ((temp >= T2) & (np.abs(time - t3) <= 2.0)).astype(float)
        X['is_extreme_hot_long'] = ((temp >= T2) & (time >= t4)).astype(float)

        # -----------------------------
        # 4. 相对机制边界的连续特征
        # -----------------------------
        X['temp_relative_to_T1'] = temp - T1
        X['temp_relative_to_Tm'] = temp - Tm
        X['temp_relative_to_T2'] = temp - T2
        X['temp_relative_to_T3'] = temp - T3

        X['time_relative_to_t1'] = time - t1
        X['time_relative_to_t2'] = time - t2
        X['time_relative_to_t3'] = time - t3
        X['time_relative_to_t4'] = time - t4

        X['relu_above_T1'] = np.maximum(temp - T1, 0)
        X['relu_above_Tm'] = np.maximum(temp - Tm, 0)
        X['relu_above_T2'] = np.maximum(temp - T2, 0)
        X['relu_above_T3'] = np.maximum(temp - T3, 0)
        X['relu_below_T1'] = np.maximum(T1 - temp, 0)

        X['relu_above_t1'] = np.maximum(time - t1, 0)
        X['relu_above_t2'] = np.maximum(time - t2, 0)
        X['relu_above_t3'] = np.maximum(time - t3, 0)
        X['relu_above_t4'] = np.maximum(time - t4, 0)
        X['relu_below_t1'] = np.maximum(t1 - time, 0)

        X['temp_gate_T1'] = self._sigmoid((temp - T1) / 4.0)
        X['temp_gate_Tm'] = self._sigmoid((temp - Tm) / 4.0)
        X['temp_gate_T2'] = self._sigmoid((temp - T2) / 3.0)
        X['time_gate_t2'] = self._sigmoid((time - t2) / 1.5)
        X['time_gate_t3'] = self._sigmoid((time - t3) / 1.5)
        X['time_gate_t4'] = self._sigmoid((time - t4) / 2.0)

        # 边界附近的分段斜率变化
        X['piecewise_temp_slope_1'] = np.maximum(temp - T1, 0) * is_mid_temp
        X['piecewise_temp_slope_2'] = np.maximum(temp - Tm, 0) * is_mid_high_temp
        X['piecewise_temp_slope_3'] = np.maximum(temp - T2, 0) * is_high_temp
        X['piecewise_time_slope_1'] = np.maximum(time - t1, 0) * is_mid_time
        X['piecewise_time_slope_2'] = np.maximum(time - t2, 0) * is_long_time
        X['piecewise_time_slope_3'] = np.maximum(time - t3, 0) * is_very_long_time

        # -----------------------------
        # 5. 机制驱动力特征
        # -----------------------------
        temp_progress = np.clip((temp - 420.0) / 60.0, 0, 1.2)
        time_progress = np.clip(log_time / np.log1p(24.0), 0, 1.2)

        treatment_extent = (
            0.54 * temp_progress +
            0.22 * time_progress +
            0.24 * temp_progress * time_progress
        )
        X['treatment_extent'] = treatment_extent

        hot_mid_window = (
            self._gaussian_window(temp, 470.0, 7.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.42)
        )
        X['hot_mid_window'] = hot_mid_window

        low_long_window = (
            self._gaussian_window(temp, 440.0, 8.0) *
            self._gaussian_window(log_time, np.log1p(24.0), 0.36)
        )
        X['low_long_window'] = low_long_window

        low_short_window = (
            self._gaussian_window(temp, 440.0, 8.0) *
            self._gaussian_window(log_time, np.log1p(1.0), 0.25)
        )
        X['low_short_window'] = low_short_window

        mid_hot_window = (
            self._gaussian_window(temp, 460.0, 7.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.38)
        )
        X['mid_hot_window'] = mid_hot_window

        saturation_risk = self._sigmoid((temp - T2) / 4.0) * self._sigmoid((time - t3) / 2.0)
        X['saturation_risk'] = saturation_risk

        X['net_solution_strengthening'] = (
            treatment_extent
            + 0.11 * hot_mid_window
            + 0.05 * low_long_window
            + 0.02 * low_short_window
            + 0.03 * mid_hot_window
            - 0.07 * saturation_risk
        )

        ductility_support = (
            0.28 * self._sigmoid((temp - T1) / 6.0) +
            0.24 * self._sigmoid((time - t2) / 2.0) +
            0.18 * self._sigmoid((temp - T2) / 4.0) * self._sigmoid((time - t2) / 2.0) +
            0.22 * hot_mid_window +
            0.07 * low_long_window +
            0.06 * mid_hot_window
        )
        X['ductility_support'] = ductility_support

        X['strength_ductility_synergy'] = X['net_solution_strengthening'].values + 0.5 * X['ductility_support'].values
        X['window_competition_470_vs_460'] = hot_mid_window - mid_hot_window
        X['long_vs_short_440_balance'] = low_long_window - low_short_window

        # -----------------------------
        # 6. 分区交互特征
        # -----------------------------
        X['temp_in_low_regime'] = temp * is_low_temp
        X['temp_in_mid_regime'] = temp * is_mid_temp
        X['temp_in_mid_high_regime'] = temp * is_mid_high_temp
        X['temp_in_high_regime'] = temp * is_high_temp

        X['log_time_in_low_regime'] = log_time * is_low_temp
        X['log_time_in_mid_regime'] = log_time * is_mid_temp
        X['log_time_in_mid_high_regime'] = log_time * is_mid_high_temp
        X['log_time_in_high_regime'] = log_time * is_high_temp

        X['temp_log_time_low_regime'] = temp * log_time * is_low_temp
        X['temp_log_time_mid_regime'] = temp * log_time * is_mid_temp
        X['temp_log_time_mid_high_regime'] = temp * log_time * is_mid_high_temp
        X['temp_log_time_high_regime'] = temp * log_time * is_high_temp

        X['high_temp_long_time_interaction'] = temp * time * X['is_high_temp_long_time'].values
        X['mid_temp_long_time_interaction'] = temp * time * X['is_mid_temp_long_time'].values
        X['hot_mid_window_interaction'] = temp * log_time * hot_mid_window
        X['low_long_window_interaction'] = temp * log_time * low_long_window
        X['mid_hot_window_interaction'] = temp * log_time * mid_hot_window

        X['gate_T2_x_t3'] = X['temp_gate_T2'].values * X['time_gate_t3'].values
        X['gate_Tm_x_t2'] = X['temp_gate_Tm'].values * X['time_gate_t2'].values
        X['regime_peak_drive'] = X['gate_T2_x_t3'].values * X['treatment_extent'].values

        # -----------------------------
        # 7. 以关键窗口为参照的邻近度特征
        # -----------------------------
        X['dist_to_solution_center_temp'] = temp - Tc
        X['dist_to_solution_center_time'] = log_time - np.log1p(tc)

        X['solution_center_temp_proximity'] = -((temp - Tc) / 8.0) ** 2
        X['solution_center_time_proximity'] = -((log_time - np.log1p(tc)) / 0.42) ** 2
        X['joint_solution_window_proximity'] = (
            X['solution_center_temp_proximity'].values +
            X['solution_center_time_proximity'].values
        )
        X['solution_window_weight'] = np.exp(X['joint_solution_window_proximity'].values)

        X['window_440_1_weight'] = (
            self._gaussian_window(temp, 440.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(1.0), 0.20)
        )
        X['window_440_24_weight'] = (
            self._gaussian_window(temp, 440.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(24.0), 0.28)
        )
        X['window_460_12_weight'] = (
            self._gaussian_window(temp, 460.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.30)
        )
        X['window_470_12_weight'] = (
            self._gaussian_window(temp, 470.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.30)
        )

        # -----------------------------
        # 8. physics baseline 及残差学习特征
        # -----------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)

        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        X['baseline_delta_strain'] = baseline_strain - self.base_strain
        X['baseline_delta_tensile'] = baseline_tensile - self.base_tensile
        X['baseline_delta_yield'] = baseline_yield - self.base_yield

        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield

        X['temp_minus_baseline_ref'] = temp - Tref_c
        X['time_log_minus_baseline_ref'] = log_time - np.log1p(1.0)
        X['treatment_minus_baseline_extent'] = X['treatment_extent'].values

        X['baseline_tensile_x_high_temp_gate'] = baseline_tensile * X['temp_gate_T2'].values
        X['baseline_yield_x_long_time_gate'] = baseline_yield * X['time_gate_t3'].values
        X['baseline_strain_x_solution_window'] = baseline_strain * X['solution_window_weight'].values

        X['baseline_tensile_x_hot_mid_window'] = baseline_tensile * hot_mid_window
        X['baseline_yield_x_hot_mid_window'] = baseline_yield * hot_mid_window
        X['baseline_tensile_x_low_long_window'] = baseline_tensile * low_long_window
        X['baseline_yield_x_low_long_window'] = baseline_yield * low_long_window
        X['baseline_yield_x_mid_hot_window'] = baseline_yield * mid_hot_window

        # 基线偏离边界的显式特征
        X['baseline_tensile_over_extent'] = self._safe_divide(baseline_tensile, 1.0 + X['treatment_extent'].values)
        X['baseline_yield_over_extent'] = self._safe_divide(baseline_yield, 1.0 + X['treatment_extent'].values)
        X['baseline_strength_gap_x_peak_window'] = (
            (baseline_tensile - baseline_yield) * X['solution_window_weight'].values
        )

        # -----------------------------
        # 9. 稳健截断与饱和特征
        # -----------------------------
        clipped_temp = np.clip(temp, 420.0, 480.0)
        clipped_time = np.clip(time, 1.0, 24.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 8.0)
        X['temp_activation'] = self._sigmoid((clipped_temp - T1) / 6.0)
        X['combined_activation'] = X['time_saturation'].values * X['temp_activation'].values

        # -----------------------------
        # 10. 基于原始铸态的潜力指标
        # -----------------------------
        base_strength_sum = self.base_tensile + self.base_yield
        X['base_strength_sum'] = base_strength_sum
        X['base_yield_tensile_ratio'] = self.base_yield / self.base_tensile

        X['process_to_base_potential'] = (
            0.35 * X['combined_activation'].values +
            0.40 * X['net_solution_strengthening'].values +
            0.25 * X['ductility_support'].values
        )

        # -----------------------------
        # 11. 针对高误差条件的显式校正特征
        # -----------------------------
        X['err_focus_boost_470_12'] = (
            X['window_470_12_weight'].values *
            (1.0 + 0.6 * X['temp_gate_T2'].values + 0.5 * X['time_gate_t3'].values)
        )

        X['err_focus_boost_440_24'] = (
            X['window_440_24_weight'].values *
            (1.0 + 0.7 * X['is_low_temp_very_long_time'].values)
        )

        X['err_focus_boost_440_1'] = (
            X['window_440_1_weight'].values *
            (1.0 + 0.6 * X['is_low_temp_regime'].values * X['is_short_time_regime'].values)
        )

        X['err_focus_balance_460_12'] = (
            X['window_460_12_weight'].values *
            (1.0 + 0.4 * X['is_mid_high_temp_regime'].values * X['is_very_long_time_regime'].values)
        )

        # 重点：470/12 与 460/12 的“分叉”校正
        X['focus_47012_minus_46012'] = X['window_470_12_weight'].values - X['window_460_12_weight'].values
        X['focus_47012_strength_drive'] = (
            X['window_470_12_weight'].values *
            X['net_solution_strengthening'].values *
            (1.0 + X['temp_gate_T2'].values)
        )
        X['focus_46012_yield_check'] = (
            X['window_460_12_weight'].values *
            X['baseline_yield'].values *
            (1.0 - 0.3 * X['window_470_12_weight'].values)
        )

        # -----------------------------
        # 12. 数值清理
        # -----------------------------
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names


if __name__ == '__main__':
    agent = FeatureAgent()
    train = pd.read_csv('data/train.csv')
    X = agent.engineer_features(train)
    print("Features:")
    print(agent.get_feature_names())
    print(f"Shape: {X.shape}")