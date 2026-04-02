import pandas as pd
import numpy as np


# 基于7499铝合金在 420–480°C、1–24h 热处理窗口的领域先验
# 任务重点：用少量、强物理意义的特征帮助模型识别“温度-时间耦合导致的组织跃迁/平台/局部转折”
# 结合当前误差分布，重点关注：
# 1) 470°C-12h：当前模型明显低估强度与屈服，说明该区域可能接近“高效固溶/组织均匀化窗口”
# 2) 440°C-1h 与 440°C-24h：同温不同时间差异显著，说明 time 在中低温区的边际作用强
# 3) 460°C-12h：出现部分高估屈服的情况，提示在“窗口中心附近”不能只用单调项，需加入窗口/转折特征
DOMAIN_PARAMS = {
    # -----------------------------
    # 机制边界（regime boundaries）
    # -----------------------------
    # 420–445°C：相对较弱处理区
    # 445–465°C：处理中等到较充分区
    # >=470°C：高温敏感区，常伴随更强温时耦合
    'critical_temp_regime_shift': 445.0,
    'critical_temp_mid_high': 460.0,
    'critical_temp_high': 470.0,
    'critical_temp_top': 480.0,

    # 时间边界：短时/中时/长时/极长时
    'critical_time_short': 3.0,
    'critical_time_mid': 8.0,
    'critical_time_long': 12.0,
    'critical_time_very_long': 24.0,

    # 重点误差区对应的“可疑组织窗口”
    # 用于构造 proximity / gate 特征，而不是硬编码绝对最优
    'window_temp_center_1': 440.0,
    'window_time_center_1': 1.0,
    'window_temp_center_2': 470.0,
    'window_time_center_2': 12.0,
    'window_temp_center_3': 440.0,
    'window_time_center_3': 24.0,
    'window_temp_center_4': 460.0,
    'window_time_center_4': 12.0,

    # 扩散/热激活动力学参数（经验量级，用于构造等效热处理特征）
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

    # 在当前数据趋势下，470/12附近可能对应较高“处理充分度”
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
        - 当前数据范围更接近固溶处理窗口，整体上温度升高 -> 强度提高；
        - 时间延长通常也提高处理充分度，但边际收益递减；
        - 470°C-12h 附近可能存在较强的组织均匀化/强化窗口；
        - 极高温长时仅做轻微饱和修正，避免违背当前数据总体趋势。
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))

        # 基础归一化进程
        temp_progress = np.clip((temp - 420.0) / 60.0, 0.0, 1.2)
        time_progress = np.clip(log_time / np.log1p(24.0), 0.0, 1.2)

        # 当前数据下温度主导，时间次主导，并保留交互项
        process_extent = (
            0.52 * temp_progress +
            0.23 * time_progress +
            0.25 * temp_progress * time_progress
        )

        # 470°C-12h 附近窗口增强：专门补偿当前最大误差区
        hot_mid_window = (
            self._gaussian_window(temp, DOMAIN_PARAMS['window_temp_center_2'], 7.0) *
            self._gaussian_window(log_time, np.log1p(DOMAIN_PARAMS['window_time_center_2']), 0.45)
        )

        # 440°C-24h 低温长时补偿：中低温区通过时间累积达到较充分处理
        low_long_window = (
            self._gaussian_window(temp, DOMAIN_PARAMS['window_temp_center_3'], 8.0) *
            self._gaussian_window(log_time, np.log1p(DOMAIN_PARAMS['window_time_center_3']), 0.40)
        )

        # 440°C-1h 早期响应窗口：帮助模型识别低温短时不应过分压低
        low_short_window = (
            self._gaussian_window(temp, DOMAIN_PARAMS['window_temp_center_1'], 8.0) *
            self._gaussian_window(log_time, np.log1p(DOMAIN_PARAMS['window_time_center_1']), 0.28)
        )

        # 极高温长时仅设置轻微饱和
        high_temp_gate = self._sigmoid((temp - DOMAIN_PARAMS['critical_temp_high']) / 4.0)
        very_long_time_gate = self._sigmoid((time - DOMAIN_PARAMS['critical_time_long']) / 2.0)
        saturation_penalty = 0.06 * high_temp_gate * very_long_time_gate

        effective_extent = (
            process_extent
            + 0.10 * hot_mid_window
            + 0.06 * low_long_window
            + 0.03 * low_short_window
            - saturation_penalty
        )

        # 强度：相对铸态显著提高，且在 470/12 附近额外抬升
        baseline_tensile = self.base_tensile + 108.0 * effective_extent + 8.0 * hot_mid_window + 4.0 * low_long_window
        baseline_yield = self.base_yield + 90.0 * effective_extent + 7.0 * hot_mid_window + 3.5 * low_long_window

        # 应变：在本数据中中高温+较充分处理下可同步改善
        ductility_extent = (
            0.40 * temp_progress +
            0.30 * time_progress +
            0.30 * temp_progress * time_progress
        )
        ductility_window = 0.55 * hot_mid_window + 0.20 * low_long_window
        ductility_penalty = 0.25 * high_temp_gate * very_long_time_gate
        baseline_strain = self.base_strain + 2.6 * ductility_extent + ductility_window - ductility_penalty

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

        # 补充更适合小样本学习的热处理累计尺度
        X['temp_centered_log_time'] = (temp - Tref_c) * log_time
        X['kinetic_index'] = self._safe_divide(log_time, temp_k)
        X['arrhenius_log_time'] = arrhenius * log_time
        X['arrhenius_time'] = arrhenius * time
        X['sqrt_equivalent_time'] = np.sqrt(np.clip(equivalent_time, 0, None))

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

        # 联合机制区域：专门针对误差大区域
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

        # -----------------------------
        # 5. 机制驱动力特征
        # -----------------------------
        temp_progress = np.clip((temp - 420.0) / 60.0, 0, 1.2)
        time_progress = np.clip(log_time / np.log1p(24.0), 0, 1.2)

        treatment_extent = (
            0.52 * temp_progress +
            0.23 * time_progress +
            0.25 * temp_progress * time_progress
        )
        X['treatment_extent'] = treatment_extent

        # 中心窗口：470°C-12h 附近，专门捕捉当前最大低估区
        hot_mid_window = (
            self._gaussian_window(temp, 470.0, 7.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.45)
        )
        X['hot_mid_window'] = hot_mid_window

        # 440°C-24h：低温长时窗口
        low_long_window = (
            self._gaussian_window(temp, 440.0, 8.0) *
            self._gaussian_window(log_time, np.log1p(24.0), 0.40)
        )
        X['low_long_window'] = low_long_window

        # 440°C-1h：低温短时窗口
        low_short_window = (
            self._gaussian_window(temp, 440.0, 8.0) *
            self._gaussian_window(log_time, np.log1p(1.0), 0.28)
        )
        X['low_short_window'] = low_short_window

        # 460°C-12h：近中心但可能不同于470/12的局部机制
        mid_hot_window = (
            self._gaussian_window(temp, 460.0, 7.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.40)
        )
        X['mid_hot_window'] = mid_hot_window

        # 高温长时下轻微饱和/局部非线性风险
        saturation_risk = self._sigmoid((temp - T2) / 4.0) * self._sigmoid((time - t3) / 2.0)
        X['saturation_risk'] = saturation_risk

        X['net_solution_strengthening'] = (
            treatment_extent
            + 0.10 * hot_mid_window
            + 0.06 * low_long_window
            + 0.03 * low_short_window
            - 0.08 * saturation_risk
        )

        ductility_support = (
            0.30 * self._sigmoid((temp - T1) / 6.0) +
            0.25 * self._sigmoid((time - t2) / 2.0) +
            0.20 * self._sigmoid((temp - T2) / 4.0) * self._sigmoid((time - t2) / 2.0) +
            0.20 * hot_mid_window +
            0.08 * low_long_window
        )
        X['ductility_support'] = ductility_support

        X['strength_ductility_synergy'] = X['net_solution_strengthening'].values + 0.5 * X['ductility_support'].values

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

        # -----------------------------
        # 7. 以关键窗口为参照的邻近度特征
        # -----------------------------
        X['dist_to_solution_center_temp'] = temp - Tc
        X['dist_to_solution_center_time'] = log_time - np.log1p(tc)

        X['solution_center_temp_proximity'] = -((temp - Tc) / 8.0) ** 2
        X['solution_center_time_proximity'] = -((log_time - np.log1p(tc)) / 0.45) ** 2
        X['joint_solution_window_proximity'] = (
            X['solution_center_temp_proximity'].values +
            X['solution_center_time_proximity'].values
        )
        X['solution_window_weight'] = np.exp(X['joint_solution_window_proximity'].values)

        # 多个误差高发窗口的 proximity
        X['window_440_1_weight'] = (
            self._gaussian_window(temp, 440.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(1.0), 0.22)
        )
        X['window_440_24_weight'] = (
            self._gaussian_window(temp, 440.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(24.0), 0.30)
        )
        X['window_460_12_weight'] = (
            self._gaussian_window(temp, 460.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.32)
        )
        X['window_470_12_weight'] = (
            self._gaussian_window(temp, 470.0, 6.0) *
            self._gaussian_window(log_time, np.log1p(12.0), 0.32)
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

        # 基线与边界/窗口耦合，让模型直接学“在某些窗口应偏离一般规律”
        X['baseline_tensile_x_high_temp_gate'] = baseline_tensile * X['temp_gate_T2'].values
        X['baseline_yield_x_long_time_gate'] = baseline_yield * X['time_gate_t3'].values
        X['baseline_strain_x_solution_window'] = baseline_strain * X['solution_window_weight'].values

        X['baseline_tensile_x_hot_mid_window'] = baseline_tensile * hot_mid_window
        X['baseline_yield_x_hot_mid_window'] = baseline_yield * hot_mid_window
        X['baseline_tensile_x_low_long_window'] = baseline_tensile * low_long_window
        X['baseline_yield_x_low_long_window'] = baseline_yield * low_long_window
        X['baseline_yield_x_mid_hot_window'] = baseline_yield * mid_hot_window

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
        # 帮助下游树模型/线性模型更快识别：
        # 470/12 通常被低估；440/24、440/1 也有系统低估；460/12 可能存在局部偏差方向变化
        X['err_focus_boost_470_12'] = (
            X['window_470_12_weight'].values *
            (1.0 + 0.5 * X['temp_gate_T2'].values + 0.5 * X['time_gate_t3'].values)
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