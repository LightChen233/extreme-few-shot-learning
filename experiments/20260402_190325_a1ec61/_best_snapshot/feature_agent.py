import pandas as pd
import numpy as np


# -----------------------------
# Domain knowledge for 7499 Al alloy solution treatment
# -----------------------------
# 当前数据温度范围约为 420–480°C，按给定领域知识更接近“固溶处理”窗口，
# 总体规律应设为：温度升高/时间延长 -> 固溶更充分 -> 强度总体提升，
# 但高温长时区可能出现局部非单调（组织粗化、晶粒长大、局部过烧风险）。
DOMAIN_PARAMS = {
    # 机制边界（基于当前题目给出的温度范围，而不是旧代码中错误的 110/150）
    'critical_temp_regime_shift': 450.0,      # 低温固溶不足 -> 中高温充分固溶的转折
    'high_temp_risk_boundary': 470.0,         # 高温区，局部非单调/组织粗化风险开始增强
    'critical_time_regime_shift': 6.0,        # 短时 -> 中时的动力学转折
    'long_time_boundary': 12.0,               # 长时区
    'very_long_time_boundary': 24.0,          # 数据上界附近

    # 动力学参数（铝合金扩散/固溶过程的启发式量级）
    'activation_energy_Q': 135000.0,          # J/mol，启发式有效激活能
    'gas_constant_R': 8.314,                  # J/mol/K

    # 参考工艺中心（用于“接近最佳窗口”类特征；非硬约束）
    'reference_temp_center': 465.0,
    'reference_time_center': 12.0,

    # Larson-Miller 风格常数
    'lmp_C': 20.0,

    # 基准原始态性能
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,
}


class FeatureAgent:
    """面向 7499 铝合金温度-时间-性能预测的小样本物理启发特征工程"""

    def __init__(self):
        self.feature_names = []

        self.base_strain = DOMAIN_PARAMS['base_strain']
        self.base_tensile = DOMAIN_PARAMS['base_tensile']
        self.base_yield = DOMAIN_PARAMS['base_yield']

        self.Q = DOMAIN_PARAMS['activation_energy_Q']
        self.R = DOMAIN_PARAMS['gas_constant_R']

    def _safe_log(self, x):
        return np.log(np.clip(x, 1e-12, None))

    def _safe_log1p(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def physics_baseline(self, temp, time):
        """
        基于领域知识的基线预测：
        - 在 420–480°C / 1–24h 范围内，总体上升趋势应与数据统计一致
        - 低温区：时间敏感
        - 中高温区：温度和时间共同驱动性能提升
        - 高温长时区仅给出轻微软化/平台化，不反转总体趋势
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = self._safe_log1p(time)

        # 1) 温度激活：450°C附近开始明显进入更充分固溶区
        temp_activation = self._sigmoid((temp - DOMAIN_PARAMS['critical_temp_regime_shift']) / 10.0)

        # 2) 时间激活：前期快、后期趋缓
        time_activation = 1.0 - np.exp(-time / DOMAIN_PARAMS['critical_time_regime_shift'])

        # 3) 扩散/动力学等效项
        arrhenius = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-6, None)))
        kinetic_dose = time * arrhenius
        kinetic_norm = kinetic_dose / (np.max(kinetic_dose) + 1e-12) if kinetic_dose.size > 0 else kinetic_dose

        # 4) 综合处理充分度：以单调增长为主
        treatment_progress = (
            0.50 * temp_activation +
            0.30 * time_activation +
            0.20 * kinetic_norm
        )

        # 5) 高温长时局部风险：只做轻微修正，避免违背数据“总体升高”趋势
        high_temp_gate = self._sigmoid((temp - DOMAIN_PARAMS['high_temp_risk_boundary']) / 4.0)
        long_time_gate = self._sigmoid((time - DOMAIN_PARAMS['long_time_boundary']) / 2.5)
        overprocess_risk = high_temp_gate * long_time_gate

        # baseline tensile / yield:
        # 用“原始态 + 处理增益 - 轻微高温长时修正”的结构
        baseline_tensile = (
            self.base_tensile
            + 270.0 * treatment_progress
            - 25.0 * overprocess_risk
        )

        baseline_yield = (
            self.base_yield
            + 210.0 * treatment_progress
            - 18.0 * overprocess_risk
        )

        # 应变：数据中并非与强度严格负相关，部分高强条件下延性也提升
        baseline_strain = (
            self.base_strain
            - 2.2
            + 5.5 * treatment_progress
            + 0.8 * high_temp_gate
            - 0.6 * overprocess_risk
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        temp_k = temp + 273.15
        log_time = self._safe_log1p(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-6, None)

        # -----------------------------
        # 1. Raw features
        # -----------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # -----------------------------
        # 2. Physics baseline as features
        # -----------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        X['baseline_strength_sum'] = baseline_tensile + baseline_yield
        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_uniformity_index'] = baseline_tensile - baseline_yield

        # -----------------------------
        # 3. Basic nonlinear / interaction features
        # -----------------------------
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # -----------------------------
        # 4. Kinetics-equivalent features
        # -----------------------------
        arrhenius = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-6, None)))
        thermal_dose = time * arrhenius
        log_thermal_dose = self._safe_log1p(thermal_dose)

        X['arrhenius'] = arrhenius
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = log_thermal_dose

        # Larson-Miller Parameter
        # LMP = T(K) * (C + log10(t))
        log10_time = np.log10(np.clip(time, 1e-6, None))
        lmp = temp_k * (DOMAIN_PARAMS['lmp_C'] + log10_time)
        X['larson_miller'] = lmp

        # Hollomon/Jaffe style simplified temper parameter
        X['temper_parameter'] = temp_k * log_time

        # -----------------------------
        # 5. Regime boundary detection features (关键)
        # -----------------------------
        T1 = DOMAIN_PARAMS['critical_temp_regime_shift']
        T2 = DOMAIN_PARAMS['high_temp_risk_boundary']
        t1 = DOMAIN_PARAMS['critical_time_regime_shift']
        t2 = DOMAIN_PARAMS['long_time_boundary']

        is_low_temp = (temp < T1).astype(float)
        is_mid_temp = ((temp >= T1) & (temp < T2)).astype(float)
        is_high_temp = (temp >= T2).astype(float)

        is_short_time = (time < t1).astype(float)
        is_mid_time = ((time >= t1) & (time < t2)).astype(float)
        is_long_time = (time >= t2).astype(float)

        X['is_low_temp'] = is_low_temp
        X['is_mid_temp'] = is_mid_temp
        X['is_high_temp'] = is_high_temp

        X['is_short_time'] = is_short_time
        X['is_mid_time'] = is_mid_time
        X['is_long_time'] = is_long_time

        # 机制突变/工艺区布尔特征
        X['is_high_temp_long_time'] = ((temp >= T2) & (time >= t2)).astype(float)
        X['is_midhigh_temp_midlong_time'] = ((temp >= T1) & (time >= t1)).astype(float)
        X['is_low_temp_long_time'] = ((temp < T1) & (time >= t2)).astype(float)
        X['is_high_temp_short_time'] = ((temp >= T2) & (time < t1)).astype(float)

        # 相对边界距离
        X['temp_relative_to_T1'] = temp - T1
        X['temp_relative_to_T2'] = temp - T2
        X['time_relative_to_t1'] = time - t1
        X['time_relative_to_t2'] = time - t2

        X['relu_above_T1'] = np.maximum(temp - T1, 0)
        X['relu_above_T2'] = np.maximum(temp - T2, 0)
        X['relu_below_T1'] = np.maximum(T1 - temp, 0)

        X['relu_above_t1'] = np.maximum(time - t1, 0)
        X['relu_above_t2'] = np.maximum(time - t2, 0)
        X['relu_below_t1'] = np.maximum(t1 - time, 0)

        # 平滑门控版本
        X['temp_activation_T1'] = self._sigmoid((temp - T1) / 8.0)
        X['temp_activation_T2'] = self._sigmoid((temp - T2) / 5.0)
        X['time_activation_t1'] = self._sigmoid((time - t1) / 1.5)
        X['time_activation_t2'] = self._sigmoid((time - t2) / 2.0)

        # -----------------------------
        # 6. Mechanism-oriented features
        # -----------------------------
        # 低温区：时间是主要补偿变量
        X['low_temp_time_compensation'] = is_low_temp * log_time
        X['low_temp_long_hold'] = is_low_temp * np.maximum(time - t1, 0)

        # 中温区：更充分固溶/均匀化
        X['mid_temp_solution_progress'] = is_mid_temp * temp * log_time

        # 高温区：快速固溶 + 长时风险
        X['high_temp_fast_solution'] = is_high_temp * temp * np.minimum(log_time, np.log1p(t2))
        X['high_temp_long_time_risk'] = is_high_temp * np.maximum(time - t2, 0)
        X['high_temp_excess'] = is_high_temp * np.maximum(temp - T2, 0)

        # 温时耦合机制
        X['coupled_solution_index'] = X['temp_activation_T1'].values * (1.0 - np.exp(-time / 8.0))
        X['solution_overrisk_balance'] = (
            X['coupled_solution_index'].values
            - 0.03 * X['high_temp_excess'].values * X['relu_above_t2'].values
        )

        # -----------------------------
        # 7. Peak-window / near-optimal neighborhood
        # -----------------------------
        T_ref = DOMAIN_PARAMS['reference_temp_center']
        t_ref = DOMAIN_PARAMS['reference_time_center']
        log_t_ref = np.log1p(t_ref)

        temp_peak_prox = -((temp - T_ref) / 12.0) ** 2
        time_peak_prox = -((log_time - log_t_ref) / 0.7) ** 2
        joint_peak_prox = temp_peak_prox + time_peak_prox
        peak_window = np.exp(joint_peak_prox)

        X['temp_peak_proximity'] = temp_peak_prox
        X['time_peak_proximity'] = time_peak_prox
        X['joint_peak_proximity'] = joint_peak_prox
        X['peak_window'] = peak_window
        X['peak_window_x_dose'] = peak_window * log_thermal_dose

        # -----------------------------
        # 8. Relative-to-baseline residual style features
        # 让模型学“相对物理基线的修正”
        # -----------------------------
        X['temp_minus_baseline_tensile_scaled'] = temp - baseline_tensile / 1.0
        X['time_minus_baseline_yield_scaled'] = time - baseline_yield / 50.0

        X['baseline_tensile_per_temp'] = self._safe_divide(baseline_tensile, temp)
        X['baseline_yield_per_temp'] = self._safe_divide(baseline_yield, temp)
        X['baseline_strain_per_logtime'] = self._safe_divide(baseline_strain, log_time + 1e-6)

        # 相对边界和基线联合残差特征
        X['baseline_tensile_x_temp_rel_T1'] = baseline_tensile * (temp - T1) / 100.0
        X['baseline_yield_x_time_rel_t1'] = baseline_yield * (time - t1) / 10.0
        X['baseline_strain_x_highT_gate'] = baseline_strain * X['temp_activation_T2'].values

        # -----------------------------
        # 9. Strength / ductility competition features
        # -----------------------------
        # 用物理启发，不直接假设强塑完全对立
        strengthening_index = (
            0.45 * X['temp_activation_T1'].values
            + 0.35 * (1.0 - np.exp(-time / 6.0))
            + 0.20 * np.tanh(log_thermal_dose * 20.0)
        )
        softening_risk = (
            X['temp_activation_T2'].values * X['time_activation_t2'].values
            * (1.0 + np.maximum(temp - T2, 0) / 20.0)
        )

        X['strengthening_index'] = strengthening_index
        X['softening_risk'] = softening_risk
        X['net_strength_index'] = strengthening_index - 0.35 * softening_risk
        X['ductility_recovery_index'] = 0.4 * softening_risk + 0.2 * X['temp_activation_T2'].values
        X['strength_ductility_balance'] = X['net_strength_index'].values - X['ductility_recovery_index'].values

        # -----------------------------
        # 10. Piecewise slope features
        # -----------------------------
        X['temp_in_low_zone'] = temp * is_low_temp
        X['temp_in_mid_zone'] = temp * is_mid_temp
        X['temp_in_high_zone'] = temp * is_high_temp

        X['log_time_in_low_zone'] = log_time * is_low_temp
        X['log_time_in_mid_zone'] = log_time * is_mid_temp
        X['log_time_in_high_zone'] = log_time * is_high_temp

        X['temp_log_time_low_zone'] = temp * log_time * is_low_temp
        X['temp_log_time_mid_zone'] = temp * log_time * is_mid_temp
        X['temp_log_time_high_zone'] = temp * log_time * is_high_temp

        # -----------------------------
        # 11. Relative to original cast state
        # -----------------------------
        X['base_strain'] = self.base_strain
        X['base_tensile'] = self.base_tensile
        X['base_yield'] = self.base_yield

        X['baseline_tensile_gain_over_cast'] = baseline_tensile - self.base_tensile
        X['baseline_yield_gain_over_cast'] = baseline_yield - self.base_yield
        X['baseline_strain_gain_over_cast'] = baseline_strain - self.base_strain

        X['process_potential_index'] = (
            0.5 * X['baseline_tensile_gain_over_cast'].values / 300.0
            + 0.3 * X['baseline_yield_gain_over_cast'].values / 250.0
            + 0.2 * X['baseline_strain_gain_over_cast'].values / 5.0
        )

        # -----------------------------
        # 12. Small-sample robustness features
        # -----------------------------
        clipped_temp = np.clip(temp, 420.0, 480.0)
        clipped_time = np.clip(time, 1.0, 24.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['temp_saturation'] = self._sigmoid((clipped_temp - 450.0) / 8.0)
        X['combined_saturation'] = X['time_saturation'].values * X['temp_saturation'].values

        # -----------------------------
        # 13. Clean
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