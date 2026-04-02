import pandas as pd
import numpy as np


# 7499 铝合金在 420–480°C、1–24h 热处理窗口下的特征工程
# 依据当前数据与误差分布：
# 1) 当前区间更接近高温固溶/均匀化主导区，而非典型低温人工时效区；
# 2) 数据统计表明：升温、延时总体提升强度，但在 460–470°C / 12h 附近存在明显“强化跃迁/峰值窗口”；
# 3) 440°C-1h、440°C-24h、460°C-12h、470°C-12h 是误差集中区，说明存在机制边界与窗口化响应；
# 4) 因此重点构造：机制阈值、动力学累计量、峰值窗口邻近度、分段交互、以及 physics baseline 残差学习特征。
DOMAIN_PARAMS = {
    # -----------------------------
    # 温度机制边界（经验）
    # -----------------------------
    # 420–440: 处理不足/固溶不充分区
    # 445–460: 进入明显强化区
    # 465–472: 强敏感峰值窗口
    # >=475: 高温充分处理区，可能出现轻微饱和但不预设强软化
    'critical_temp_regime_shift': 445.0,
    'critical_temp_mid_high': 458.0,
    'critical_temp_peak': 468.0,
    'critical_temp_high': 472.0,
    'critical_temp_top': 480.0,

    # -----------------------------
    # 时间机制边界（经验）
    # -----------------------------
    # 1h 附近：短时
    # 8–12h：中长时，常是动力学明显推进窗口
    # 24h：长时极限点
    'critical_time_short': 2.0,
    'critical_time_mid': 8.0,
    'critical_time_long': 12.0,
    'critical_time_very_long': 18.0,
    'critical_time_top': 24.0,

    # -----------------------------
    # 重点误差窗口 / 机制窗口
    # -----------------------------
    'window_temp_center_1': 440.0,
    'window_time_center_1': 1.0,

    'window_temp_center_2': 470.0,
    'window_time_center_2': 12.0,

    'window_temp_center_3': 440.0,
    'window_time_center_3': 24.0,

    'window_temp_center_4': 460.0,
    'window_time_center_4': 12.0,

    # 额外机制中心：470/12 与 460/12 的分叉带
    'window_temp_center_5': 465.0,
    'window_time_center_5': 12.0,

    # -----------------------------
    # 动力学参数
    # -----------------------------
    # Al-Zn-Mg-Cu 合金高温扩散/固溶相关经验量级
    'activation_energy_Q': 118000.0,  # J/mol
    'gas_constant_R': 8.314,          # J/(mol·K)

    # Larson-Miller 风格参数
    'lmp_C': 20.0,

    # Hollomon-Jaffe 风格参数
    'hj_C': 14.0,

    # -----------------------------
    # 参考温度
    # -----------------------------
    'reference_temp_c': 420.0,
    'reference_temp_k': 420.0 + 273.15,

    # -----------------------------
    # 原始铸态基准性能
    # -----------------------------
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # -----------------------------
    # 主强化窗口中心
    # -----------------------------
    'solution_center_temp': 468.0,
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
        self.hj_C = DOMAIN_PARAMS['hj_C']

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def _gaussian_window(self, x, center, width):
        width = max(width, 1e-8)
        return np.exp(-((x - center) / width) ** 2)

    def physics_baseline(self, temp, time):
        """
        基于领域知识的基线估计（不依赖训练标签逐点拟合）：
        - 当前数据范围内，整体上更高温度 / 更长时间通常对应更高强度；
        - 时间效应表现为前期快、后期趋缓，适合 log(time)；
        - 460–470°C / 12h 附近存在强化跃迁窗口；
        - 440°C / 1h 与 440°C / 24h 属于低温短时、低温长时两个边界窗口；
        - 仅对极高温长时引入轻微软饱和，不引入与数据主趋势冲突的强“过时效软化”。
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        log_time = np.log1p(np.clip(time, 0, None))

        temp_progress = np.clip((temp - 420.0) / 60.0, 0.0, 1.2)
        time_progress = np.clip(log_time / np.log1p(24.0), 0.0, 1.2)

        process_extent = (
            0.56 * temp_progress +
            0.18 * time_progress +
            0.26 * temp_progress * time_progress
        )

        peak_470_12 = (
            self._gaussian_window(temp, 470.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.28)
        )

        mid_460_12 = (
            self._gaussian_window(temp, 460.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.30)
        )

        low_440_24 = (
            self._gaussian_window(temp, 440.0, 7.0) *
            self._gaussian_window(log_time, np.log1p(24.0), 0.24)
        )

        low_440_1 = (
            self._gaussian_window(temp, 440.0, 7.0) *
            self._gaussian_window(log_time, np.log1p(1.0), 0.18)
        )

        transition_465_12 = (
            self._gaussian_window(temp, 465.0, 5.5) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.24)
        )

        high_temp_gate = self._sigmoid((temp - 472.0) / 3.5)
        long_time_gate = self._sigmoid((time - 12.0) / 1.8)
        very_long_gate = self._sigmoid((time - 20.0) / 2.0)

        mild_saturation = 0.035 * high_temp_gate * long_time_gate + 0.015 * high_temp_gate * very_long_gate

        effective_extent = (
            process_extent
            + 0.12 * peak_470_12
            + 0.06 * mid_460_12
            + 0.035 * low_440_24
            + 0.015 * low_440_1
            + 0.05 * transition_465_12
            - mild_saturation
        )

        baseline_tensile = (
            self.base_tensile
            + 108.0 * effective_extent
            + 16.0 * peak_470_12
            + 7.0 * mid_460_12
            + 3.5 * low_440_24
            - 1.5 * mild_saturation
        )

        baseline_yield = (
            self.base_yield
            + 86.0 * effective_extent
            + 12.5 * peak_470_12
            + 5.5 * mid_460_12
            + 2.5 * low_440_24
            - 1.2 * mild_saturation
        )

        ductility_extent = (
            0.36 * temp_progress +
            0.24 * time_progress +
            0.40 * temp_progress * time_progress
        )

        baseline_strain = (
            self.base_strain
            + 2.2 * ductility_extent
            + 0.85 * peak_470_12
            + 0.45 * mid_460_12
            + 0.20 * low_440_24
            - 0.16 * high_temp_gate * long_time_gate
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-8, None)

        T1 = DOMAIN_PARAMS['critical_temp_regime_shift']
        Tm = DOMAIN_PARAMS['critical_temp_mid_high']
        Tp = DOMAIN_PARAMS['critical_temp_peak']
        T2 = DOMAIN_PARAMS['critical_temp_high']
        T3 = DOMAIN_PARAMS['critical_temp_top']

        t1 = DOMAIN_PARAMS['critical_time_short']
        t2 = DOMAIN_PARAMS['critical_time_mid']
        t3 = DOMAIN_PARAMS['critical_time_long']
        t4 = DOMAIN_PARAMS['critical_time_very_long']
        t5 = DOMAIN_PARAMS['critical_time_top']

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
        X['sqrt_time_sq'] = sqrt_time ** 2

        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)
        X['temp_over_log_time'] = self._safe_divide(temp, log_time + 1e-6)

        # -----------------------------
        # 2. 动力学等效特征
        # -----------------------------
        arrhenius = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-8, None)))
        X['arrhenius_factor'] = arrhenius

        thermal_dose = time * arrhenius
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = self._safe_log(thermal_dose)

        delta_invT = (1.0 / np.clip(Tref_k, 1e-8, None)) - inv_temp_k
        equivalent_rate = np.exp(np.clip((self.Q / self.R) * delta_invT, -50, 50))
        equivalent_time = time * equivalent_rate

        X['equivalent_rate_vs_420C'] = equivalent_rate
        X['equivalent_time_vs_420C'] = equivalent_time
        X['log_equivalent_time_vs_420C'] = self._safe_log(equivalent_time)
        X['sqrt_equivalent_time'] = np.sqrt(np.clip(equivalent_time, 0, None))

        # Larson-Miller / Hollomon-Jaffe 风格
        lmp = temp_k * (self.lmp_C + np.log10(np.clip(time, 1e-6, None)))
        hj = temp * (self.hj_C + np.log10(np.clip(time, 1e-6, None)))

        X['larson_miller'] = lmp
        X['hollomon_jaffe'] = hj

        X['temp_centered_log_time'] = (temp - Tref_c) * log_time
        X['kinetic_index'] = self._safe_divide(log_time, temp_k)
        X['arrhenius_log_time'] = arrhenius * log_time
        X['arrhenius_time'] = arrhenius * time

        transformed_fraction_proxy = 1.0 - np.exp(-np.clip(equivalent_time, 0, None) / 8.0)
        X['transformed_fraction_proxy'] = transformed_fraction_proxy
        X['transformed_fraction_sq'] = transformed_fraction_proxy ** 2
        X['transformed_fraction_cube'] = transformed_fraction_proxy ** 3

        # -----------------------------
        # 3. 物理机制区间检测特征（关键）
        # -----------------------------
        is_low_temp = (temp < T1).astype(float)
        is_mid_temp = ((temp >= T1) & (temp < Tm)).astype(float)
        is_mid_high_temp = ((temp >= Tm) & (temp < Tp)).astype(float)
        is_peak_temp = ((temp >= Tp) & (temp < T2)).astype(float)
        is_high_temp = (temp >= T2).astype(float)
        is_top_temp = (temp >= T3).astype(float)

        X['is_low_temp_regime'] = is_low_temp
        X['is_mid_temp_regime'] = is_mid_temp
        X['is_mid_high_temp_regime'] = is_mid_high_temp
        X['is_peak_temp_regime'] = is_peak_temp
        X['is_high_temp_regime'] = is_high_temp
        X['is_top_temp_regime'] = is_top_temp

        is_short_time = (time <= t1).astype(float)
        is_mid_time = ((time > t1) & (time < t2)).astype(float)
        is_long_time = ((time >= t2) & (time < t3)).astype(float)
        is_peak_time = ((time >= t3) & (time < t4)).astype(float)
        is_very_long_time = (time >= t4).astype(float)
        is_top_time = (time >= t5).astype(float)

        X['is_short_time_regime'] = is_short_time
        X['is_mid_time_regime'] = is_mid_time
        X['is_long_time_regime'] = is_long_time
        X['is_peak_time_regime'] = is_peak_time
        X['is_very_long_time_regime'] = is_very_long_time
        X['is_top_time_regime'] = is_top_time

        # 重点联合窗口
        X['is_440_1_window'] = ((np.abs(temp - 440.0) <= 5.0) & (np.abs(time - 1.0) <= 0.8)).astype(float)
        X['is_440_24_window'] = ((np.abs(temp - 440.0) <= 5.0) & (np.abs(time - 24.0) <= 3.0)).astype(float)
        X['is_460_12_window'] = ((np.abs(temp - 460.0) <= 5.0) & (np.abs(time - 12.0) <= 1.5)).astype(float)
        X['is_470_12_window'] = ((np.abs(temp - 470.0) <= 5.0) & (np.abs(time - 12.0) <= 1.5)).astype(float)
        X['is_465_12_transition_window'] = ((np.abs(temp - 465.0) <= 4.0) & (np.abs(time - 12.0) <= 1.5)).astype(float)

        X['is_high_temp_long_time'] = ((temp >= T2) & (time >= t3)).astype(float)
        X['is_high_temp_peak_time'] = ((temp >= Tp) & (time >= t3) & (time < t4)).astype(float)
        X['is_mid_high_temp_long_time'] = ((temp >= Tm) & (temp < T2) & (time >= t3)).astype(float)
        X['is_low_temp_very_long_time'] = ((temp < T1) & (time >= t4)).astype(float)

        X['is_near_regime_shift'] = (np.abs(temp - T1) <= 4.0).astype(float)
        X['is_near_mid_high_shift'] = (np.abs(temp - Tm) <= 4.0).astype(float)
        X['is_near_peak_shift'] = (np.abs(temp - Tp) <= 3.0).astype(float)
        X['is_near_high_temp_shift'] = (np.abs(temp - T2) <= 3.0).astype(float)

        # 机制门控
        X['is_below_solution_window'] = ((temp < Tm) & (time < t3)).astype(float)
        X['is_entering_solution_window'] = ((temp >= Tm) & (temp < Tp) & (time >= t2)).astype(float)
        X['is_peak_solution_window'] = ((temp >= Tp) & (temp < T2) & (np.abs(time - t3) <= 1.5)).astype(float)
        X['is_hot_saturated_window'] = ((temp >= T2) & (time >= t4)).astype(float)

        # -----------------------------
        # 4. 相对机制边界的连续特征
        # -----------------------------
        X['temp_relative_to_T1'] = temp - T1
        X['temp_relative_to_Tm'] = temp - Tm
        X['temp_relative_to_Tp'] = temp - Tp
        X['temp_relative_to_T2'] = temp - T2
        X['temp_relative_to_T3'] = temp - T3

        X['time_relative_to_t1'] = time - t1
        X['time_relative_to_t2'] = time - t2
        X['time_relative_to_t3'] = time - t3
        X['time_relative_to_t4'] = time - t4
        X['time_relative_to_t5'] = time - t5

        X['relu_above_T1'] = np.maximum(temp - T1, 0)
        X['relu_above_Tm'] = np.maximum(temp - Tm, 0)
        X['relu_above_Tp'] = np.maximum(temp - Tp, 0)
        X['relu_above_T2'] = np.maximum(temp - T2, 0)
        X['relu_above_T3'] = np.maximum(temp - T3, 0)
        X['relu_below_T1'] = np.maximum(T1 - temp, 0)

        X['relu_above_t1'] = np.maximum(time - t1, 0)
        X['relu_above_t2'] = np.maximum(time - t2, 0)
        X['relu_above_t3'] = np.maximum(time - t3, 0)
        X['relu_above_t4'] = np.maximum(time - t4, 0)
        X['relu_above_t5'] = np.maximum(time - t5, 0)
        X['relu_below_t1'] = np.maximum(t1 - time, 0)

        X['temp_gate_T1'] = self._sigmoid((temp - T1) / 3.5)
        X['temp_gate_Tm'] = self._sigmoid((temp - Tm) / 3.0)
        X['temp_gate_Tp'] = self._sigmoid((temp - Tp) / 2.5)
        X['temp_gate_T2'] = self._sigmoid((temp - T2) / 2.5)

        X['time_gate_t2'] = self._sigmoid((time - t2) / 1.2)
        X['time_gate_t3'] = self._sigmoid((time - t3) / 1.0)
        X['time_gate_t4'] = self._sigmoid((time - t4) / 1.5)
        X['time_gate_t5'] = self._sigmoid((time - t5) / 1.5)

        X['piecewise_temp_slope_1'] = np.maximum(temp - T1, 0) * is_mid_temp
        X['piecewise_temp_slope_2'] = np.maximum(temp - Tm, 0) * is_mid_high_temp
        X['piecewise_temp_slope_3'] = np.maximum(temp - Tp, 0) * is_peak_temp
        X['piecewise_temp_slope_4'] = np.maximum(temp - T2, 0) * is_high_temp

        X['piecewise_time_slope_1'] = np.maximum(time - t1, 0) * is_mid_time
        X['piecewise_time_slope_2'] = np.maximum(time - t2, 0) * is_long_time
        X['piecewise_time_slope_3'] = np.maximum(time - t3, 0) * is_peak_time
        X['piecewise_time_slope_4'] = np.maximum(time - t4, 0) * is_very_long_time

        # -----------------------------
        # 5. 机制驱动力特征
        # -----------------------------
        temp_progress = np.clip((temp - 420.0) / 60.0, 0.0, 1.2)
        time_progress = np.clip(log_time / np.log1p(24.0), 0.0, 1.2)

        treatment_extent = (
            0.56 * temp_progress +
            0.18 * time_progress +
            0.26 * temp_progress * time_progress
        )
        X['treatment_extent'] = treatment_extent

        peak_470_12 = (
            self._gaussian_window(temp, 470.0, 5.5) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.24)
        )
        mid_460_12 = (
            self._gaussian_window(temp, 460.0, 5.5) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.26)
        )
        low_440_24 = (
            self._gaussian_window(temp, 440.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(24.0), 0.22)
        )
        low_440_1 = (
            self._gaussian_window(temp, 440.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(1.0), 0.16)
        )
        transition_465_12 = (
            self._gaussian_window(temp, 465.0, 4.5) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.22)
        )

        X['peak_470_12_window'] = peak_470_12
        X['mid_460_12_window'] = mid_460_12
        X['low_440_24_window'] = low_440_24
        X['low_440_1_window'] = low_440_1
        X['transition_465_12_window'] = transition_465_12

        saturation_risk = (
            self._sigmoid((temp - T2) / 3.0) *
            self._sigmoid((time - t4) / 1.8)
        )
        X['saturation_risk'] = saturation_risk

        X['net_solution_strengthening'] = (
            treatment_extent
            + 0.13 * peak_470_12
            + 0.07 * mid_460_12
            + 0.04 * transition_465_12
            + 0.03 * low_440_24
            + 0.01 * low_440_1
            - 0.06 * saturation_risk
        )

        ductility_support = (
            0.22 * X['temp_gate_T1'].values +
            0.18 * X['time_gate_t2'].values +
            0.18 * X['temp_gate_Tm'].values * X['time_gate_t2'].values +
            0.26 * peak_470_12 +
            0.14 * mid_460_12 +
            0.05 * low_440_24
            - 0.05 * saturation_risk
        )
        X['ductility_support'] = ductility_support

        X['strength_ductility_synergy'] = X['net_solution_strengthening'].values + 0.55 * X['ductility_support'].values
        X['window_competition_470_vs_460'] = peak_470_12 - mid_460_12
        X['transition_vs_peak_balance'] = transition_465_12 - peak_470_12
        X['long_vs_short_440_balance'] = low_440_24 - low_440_1

        # -----------------------------
        # 6. 分区交互特征
        # -----------------------------
        X['temp_in_low_regime'] = temp * is_low_temp
        X['temp_in_mid_regime'] = temp * is_mid_temp
        X['temp_in_mid_high_regime'] = temp * is_mid_high_temp
        X['temp_in_peak_regime'] = temp * is_peak_temp
        X['temp_in_high_regime'] = temp * is_high_temp

        X['log_time_in_low_regime'] = log_time * is_low_temp
        X['log_time_in_mid_regime'] = log_time * is_mid_temp
        X['log_time_in_mid_high_regime'] = log_time * is_mid_high_temp
        X['log_time_in_peak_regime'] = log_time * is_peak_temp
        X['log_time_in_high_regime'] = log_time * is_high_temp

        X['temp_log_time_low_regime'] = temp * log_time * is_low_temp
        X['temp_log_time_mid_regime'] = temp * log_time * is_mid_temp
        X['temp_log_time_mid_high_regime'] = temp * log_time * is_mid_high_temp
        X['temp_log_time_peak_regime'] = temp * log_time * is_peak_temp
        X['temp_log_time_high_regime'] = temp * log_time * is_high_temp

        X['high_temp_long_time_interaction'] = temp * time * X['is_high_temp_long_time'].values
        X['peak_temp_peak_time_interaction'] = temp * time * X['is_high_temp_peak_time'].values
        X['mid_high_temp_long_time_interaction'] = temp * time * X['is_mid_high_temp_long_time'].values

        X['peak_470_12_interaction'] = temp * log_time * peak_470_12
        X['mid_460_12_interaction'] = temp * log_time * mid_460_12
        X['transition_465_12_interaction'] = temp * log_time * transition_465_12
        X['low_440_24_interaction'] = temp * log_time * low_440_24

        X['gate_Tp_x_t3'] = X['temp_gate_Tp'].values * X['time_gate_t3'].values
        X['gate_T2_x_t4'] = X['temp_gate_T2'].values * X['time_gate_t4'].values
        X['gate_Tm_x_t2'] = X['temp_gate_Tm'].values * X['time_gate_t2'].values

        X['regime_peak_drive'] = X['gate_Tp_x_t3'].values * X['treatment_extent'].values
        X['regime_high_saturation_drive'] = X['gate_T2_x_t4'].values * X['treatment_extent'].values

        # -----------------------------
        # 7. 关键窗口邻近度特征
        # -----------------------------
        X['dist_to_solution_center_temp'] = temp - Tc
        X['dist_to_solution_center_time'] = log_time - np.log1p(tc)

        X['solution_center_temp_proximity'] = -((temp - Tc) / 7.0) ** 2
        X['solution_center_time_proximity'] = -((log_time - np.log1p(tc)) / 0.24) ** 2
        X['joint_solution_window_proximity'] = (
            X['solution_center_temp_proximity'].values +
            X['solution_center_time_proximity'].values
        )
        X['solution_window_weight'] = np.exp(X['joint_solution_window_proximity'].values)

        X['window_440_1_weight'] = (
            self._gaussian_window(temp, 440.0, 5.0) *
            self._gaussian_window(log_time, np.log1p(1.0), 0.14)
        )
        X['window_440_24_weight'] = (
            self._gaussian_window(temp, 440.0, 5.0) *
            self._gaussian_window(log_time, np.log1p(24.0), 0.18)
        )
        X['window_460_12_weight'] = (
            self._gaussian_window(temp, 460.0, 5.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.20)
        )
        X['window_470_12_weight'] = (
            self._gaussian_window(temp, 470.0, 5.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.20)
        )
        X['window_465_12_weight'] = (
            self._gaussian_window(temp, 465.0, 4.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.18)
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

        X['baseline_tensile_x_peak_gate'] = baseline_tensile * X['temp_gate_Tp'].values
        X['baseline_yield_x_peak_gate'] = baseline_yield * X['temp_gate_Tp'].values
        X['baseline_tensile_x_high_temp_gate'] = baseline_tensile * X['temp_gate_T2'].values
        X['baseline_yield_x_long_time_gate'] = baseline_yield * X['time_gate_t3'].values
        X['baseline_strain_x_solution_window'] = baseline_strain * X['solution_window_weight'].values

        X['baseline_tensile_x_470_12'] = baseline_tensile * peak_470_12
        X['baseline_yield_x_470_12'] = baseline_yield * peak_470_12
        X['baseline_tensile_x_460_12'] = baseline_tensile * mid_460_12
        X['baseline_yield_x_460_12'] = baseline_yield * mid_460_12
        X['baseline_tensile_x_440_24'] = baseline_tensile * low_440_24
        X['baseline_yield_x_440_24'] = baseline_yield * low_440_24

        X['baseline_tensile_over_extent'] = self._safe_divide(baseline_tensile, 1.0 + X['treatment_extent'].values)
        X['baseline_yield_over_extent'] = self._safe_divide(baseline_yield, 1.0 + X['treatment_extent'].values)
        X['baseline_strength_gap_x_peak_window'] = (
            (baseline_tensile - baseline_yield) * X['solution_window_weight'].values
        )

        # 基线相对机制边界
        X['baseline_tensile_relative_to_Tp'] = baseline_tensile * (temp - Tp)
        X['baseline_yield_relative_to_Tp'] = baseline_yield * (temp - Tp)
        X['baseline_tensile_relative_to_t3'] = baseline_tensile * (time - t3)
        X['baseline_yield_relative_to_t3'] = baseline_yield * (time - t3)

        # -----------------------------
        # 9. 稳健截断与饱和特征
        # -----------------------------
        clipped_temp = np.clip(temp, 420.0, 480.0)
        clipped_time = np.clip(time, 1.0, 24.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 8.0)
        X['temp_activation'] = self._sigmoid((clipped_temp - T1) / 5.0)
        X['peak_temp_activation'] = self._sigmoid((clipped_temp - Tp) / 2.5)
        X['combined_activation'] = X['time_saturation'].values * X['temp_activation'].values
        X['peak_combined_activation'] = X['time_saturation'].values * X['peak_temp_activation'].values

        # -----------------------------
        # 10. 基于原始铸态的潜力指标
        # -----------------------------
        base_strength_sum = self.base_tensile + self.base_yield
        X['base_strength_sum'] = base_strength_sum
        X['base_yield_tensile_ratio'] = self.base_yield / self.base_tensile

        X['process_to_base_potential'] = (
            0.34 * X['combined_activation'].values +
            0.46 * X['net_solution_strengthening'].values +
            0.20 * X['ductility_support'].values
        )

        X['peak_window_potential'] = (
            0.50 * X['window_470_12_weight'].values +
            0.30 * X['window_460_12_weight'].values +
            0.20 * X['window_465_12_weight'].values
        )

        # -----------------------------
        # 11. 针对高误差条件的显式校正特征
        # -----------------------------
        X['err_focus_boost_470_12'] = (
            X['window_470_12_weight'].values *
            (1.0 + 0.8 * X['temp_gate_Tp'].values + 0.6 * X['time_gate_t3'].values)
        )

        X['err_focus_boost_460_12'] = (
            X['window_460_12_weight'].values *
            (1.0 + 0.5 * X['temp_gate_Tm'].values + 0.5 * X['time_gate_t3'].values)
        )

        X['err_focus_boost_440_24'] = (
            X['window_440_24_weight'].values *
            (1.0 + 0.7 * X['is_low_temp_very_long_time'].values)
        )

        X['err_focus_boost_440_1'] = (
            X['window_440_1_weight'].values *
            (1.0 + 0.7 * X['is_low_temp_regime'].values * X['is_short_time_regime'].values)
        )

        # 470/12 与 460/12 的分叉校正
        X['focus_47012_minus_46012'] = X['window_470_12_weight'].values - X['window_460_12_weight'].values
        X['focus_47012_plus_46012'] = X['window_470_12_weight'].values + X['window_460_12_weight'].values

        X['focus_47012_strength_drive'] = (
            X['window_470_12_weight'].values *
            X['net_solution_strengthening'].values *
            (1.0 + X['temp_gate_Tp'].values)
        )

        X['focus_46012_yield_drive'] = (
            X['window_460_12_weight'].values *
            X['baseline_yield'].values *
            (1.0 + 0.5 * X['temp_gate_Tm'].values)
        )

        X['focus_transition_46512_drive'] = (
            X['window_465_12_weight'].values *
            X['transition_465_12_window'].values *
            X['net_solution_strengthening'].values
        )

        # 470/12 下模型系统低估强度/屈服，加入更显式的峰值强化提示
        X['peak_strength_hint'] = (
            X['window_470_12_weight'].values *
            X['baseline_tensile'].values *
            X['peak_combined_activation'].values
        )

        X['peak_yield_hint'] = (
            X['window_470_12_weight'].values *
            X['baseline_yield'].values *
            X['peak_combined_activation'].values
        )

        # 460/12 有时高估/低估混合，加入过渡带特征帮助区分
        X['mid_peak_branch_indicator'] = (
            X['window_460_12_weight'].values *
            (1.0 - X['window_470_12_weight'].values) *
            X['window_465_12_weight'].values
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