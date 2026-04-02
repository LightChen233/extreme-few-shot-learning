import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # -----------------------------
    # 基础物理常数 / 动力学参数
    # -----------------------------
    'gas_constant_R': 8.314,                  # J/mol/K
    'activation_energy_Q': 72000.0,           # J/mol，7499铝合金时效/析出扩散经验量级

    # -----------------------------
    # 机制边界（Phase / Regime Boundaries）
    # 依据铝合金析出强化的一般规律做启发式设定
    # -----------------------------
    'critical_temp_regime_shift': 120.0,      # 从弱时效向明显析出强化过渡
    'critical_temp_peak_lower': 135.0,        # 峰值时效下边界
    'critical_temp_peak_upper': 155.0,        # 峰值时效上边界
    'critical_temp_overaging': 165.0,         # 过时效/粗化风险显著上升

    'critical_time_nucleation': 2.0,          # 初期析出/成核主导
    'critical_time_growth': 6.0,              # 强化快速建立
    'critical_time_peak': 10.0,               # 峰值时效附近
    'critical_time_overaging': 16.0,          # 长时过时效风险显著增加

    # -----------------------------
    # 等效时效参数
    # -----------------------------
    'lmp_constant_C': 20.0,                   # Larson-Miller风格常数
    'reference_temp_c': 140.0,                # 经验参考温度
    'reference_time_h': 8.0,                  # 经验参考时间

    # -----------------------------
    # 原始铸态样品基准性能
    # -----------------------------
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # -----------------------------
    # 物理基线中使用的经验峰值
    # 仅作世界知识先验，不依赖训练标签
    # -----------------------------
    'peak_temp_center': 145.0,
    'peak_log_time_center': np.log1p(8.0),
    'peak_temp_width': 22.0,
    'peak_log_time_width': 0.9,

    # -----------------------------
    # 经验峰值增益（相对铸态）
    # -----------------------------
    'max_tensile_gain': 260.0,
    'max_yield_gain': 210.0,
    'max_strain_drop': 2.6,
    'overaging_strength_loss': 90.0,
    'overaging_ductility_gain': 2.8,
}


class FeatureAgent:
    """基于材料热处理机理 + 物理基线残差学习的特征工程"""

    def __init__(self):
        self.feature_names = []
        self.params = DOMAIN_PARAMS

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def physics_baseline(self, temp, time):
        """
        基于世界知识的物理启发式基线：
        - 低温/短时：析出不足，强化弱
        - 中温/中时：接近峰值时效，强度提升明显
        - 高温/长时：过时效，强度回落，塑性恢复
        返回:
            baseline_strain, baseline_tensile, baseline_yield
        """
        p = self.params

        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # Arrhenius扩散强度：温度越高，扩散越快
        arr = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * np.clip(temp_k, 1e-6, None)))

        # 归一化动力学进程，相对参考状态
        ref_temp_k = p['reference_temp_c'] + 273.15
        ref_arr = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * ref_temp_k))
        kinetic_progress = self._safe_divide(time * arr, p['reference_time_h'] * ref_arr)
        kinetic_progress = np.clip(kinetic_progress, 0, 20)
        kinetic_saturation = 1.0 - np.exp(-kinetic_progress)

        # 峰值窗口：中温中时最优
        temp_peak = np.exp(-((temp - p['peak_temp_center']) / p['peak_temp_width']) ** 2)
        time_peak = np.exp(-((log_time - p['peak_log_time_center']) / p['peak_log_time_width']) ** 2)
        joint_peak = temp_peak * time_peak

        # 过时效门控：高温+长时
        over_temp_gate = self._sigmoid((temp - p['critical_temp_overaging']) / 8.0)
        over_time_gate = self._sigmoid((time - p['critical_time_overaging']) / 3.0)
        overaging = over_temp_gate * over_time_gate

        # 强化驱动：需要足够扩散 + 接近峰值窗口
        strengthening = 0.55 * kinetic_saturation + 0.95 * joint_peak
        strengthening = np.clip(strengthening, 0, 1.5)

        # 强度基线
        tensile = (
            p['base_tensile']
            + p['max_tensile_gain'] * strengthening
            - p['overaging_strength_loss'] * overaging
        )

        yield_strength = (
            p['base_yield']
            + p['max_yield_gain'] * strengthening
            - 0.85 * p['overaging_strength_loss'] * overaging
        )

        # 屈服不能超过抗拉，加入温和约束
        yield_strength = np.minimum(yield_strength, tensile - 8.0)

        # 塑性：强化增强时下降，过时效时部分恢复
        strain = (
            p['base_strain']
            - p['max_strain_drop'] * strengthening
            + p['overaging_ductility_gain'] * overaging
        )

        # 低温极短时，性能应接近原始态
        weak_treatment = self._sigmoid((p['critical_temp_regime_shift'] - temp) / 10.0) * self._sigmoid((p['critical_time_nucleation'] - time) / 1.0)
        tensile = weak_treatment * (0.85 * p['base_tensile'] + 0.15 * tensile) + (1 - weak_treatment) * tensile
        yield_strength = weak_treatment * (0.85 * p['base_yield'] + 0.15 * yield_strength) + (1 - weak_treatment) * yield_strength
        strain = weak_treatment * (0.85 * p['base_strain'] + 0.15 * strain) + (1 - weak_treatment) * strain

        # 数值裁剪，避免极端外推
        tensile = np.clip(tensile, 80, 650)
        yield_strength = np.clip(yield_strength, 50, 600)
        strain = np.clip(strain, 1.0, 20.0)

        return strain, tensile, yield_strength

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)
        p = self.params

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-6, None)

        # -----------------------------
        # 1. 原始特征
        # -----------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # -----------------------------
        # 2. 基础非线性
        # -----------------------------
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_cube'] = temp ** 3
        X['time_cube'] = time ** 3
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # -----------------------------
        # 3. 动力学等效特征
        # -----------------------------
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * np.clip(temp_k, 1e-6, None)))
        X['arrhenius_factor'] = arrhenius
        X['thermal_dose'] = time * arrhenius
        X['log_thermal_dose'] = self._safe_log(X['thermal_dose'].values)
        X['arrhenius_time_temp'] = arrhenius * time * temp

        # Larson-Miller风格参数
        X['larson_miller'] = temp_k * (p['lmp_constant_C'] + np.log10(np.clip(time, 1e-6, None)))
        X['lmp_log1p'] = self._safe_log(X['larson_miller'].values)

        # Hollomon/JMAK风格等效进程
        ref_temp_k = p['reference_temp_c'] + 273.15
        ref_arr = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * ref_temp_k))
        eq_progress = self._safe_divide(time * arrhenius, p['reference_time_h'] * ref_arr)
        X['equivalent_aging_progress'] = eq_progress
        X['aging_saturation'] = 1.0 - np.exp(-np.clip(eq_progress, 0, 20))
        X['aging_progress_sq'] = X['equivalent_aging_progress'].values ** 2

        # -----------------------------
        # 4. 物理机制区间检测特征（关键）
        # -----------------------------
        T1 = p['critical_temp_regime_shift']
        T2 = p['critical_temp_peak_lower']
        T3 = p['critical_temp_peak_upper']
        T4 = p['critical_temp_overaging']

        t1 = p['critical_time_nucleation']
        t2 = p['critical_time_growth']
        t3 = p['critical_time_peak']
        t4 = p['critical_time_overaging']

        is_low_temp = (temp < T1).astype(float)
        is_transition_temp = ((temp >= T1) & (temp < T2)).astype(float)
        is_peak_temp = ((temp >= T2) & (temp <= T3)).astype(float)
        is_high_temp = ((temp > T3) & (temp < T4)).astype(float)
        is_over_temp = (temp >= T4).astype(float)

        is_short_time = (time < t1).astype(float)
        is_growth_time = ((time >= t1) & (time < t2)).astype(float)
        is_peak_time = ((time >= t2) & (time <= t3)).astype(float)
        is_long_time = ((time > t3) & (time < t4)).astype(float)
        is_over_time = (time >= t4).astype(float)

        X['is_low_temp'] = is_low_temp
        X['is_transition_temp'] = is_transition_temp
        X['is_peak_temp'] = is_peak_temp
        X['is_high_temp'] = is_high_temp
        X['is_over_temp'] = is_over_temp

        X['is_short_time'] = is_short_time
        X['is_growth_time'] = is_growth_time
        X['is_peak_time'] = is_peak_time
        X['is_long_time'] = is_long_time
        X['is_over_time'] = is_over_time

        # 关键机制组合区
        X['is_underaged'] = ((temp < T2) & (time < t2)).astype(float)
        X['is_near_peak_aging'] = ((temp >= T2) & (temp <= T3) & (time >= t2) & (time <= t3)).astype(float)
        X['is_overaged'] = ((temp >= T4) & (time >= t4)).astype(float)
        X['is_highT_shortt'] = ((temp >= T4) & (time < t2)).astype(float)
        X['is_lowT_longt'] = ((temp < T2) & (time >= t4)).astype(float)

        # 距边界的连续特征
        X['temp_relative_to_regime_shift'] = temp - T1
        X['temp_relative_to_peak_lower'] = temp - T2
        X['temp_relative_to_peak_upper'] = temp - T3
        X['temp_relative_to_overaging'] = temp - T4

        X['time_relative_to_nucleation'] = time - t1
        X['time_relative_to_growth'] = time - t2
        X['time_relative_to_peak'] = time - t3
        X['time_relative_to_overaging'] = time - t4

        X['relu_above_regime_shift'] = np.maximum(temp - T1, 0)
        X['relu_above_peak_lower'] = np.maximum(temp - T2, 0)
        X['relu_above_peak_upper'] = np.maximum(temp - T3, 0)
        X['relu_above_overaging_temp'] = np.maximum(temp - T4, 0)

        X['relu_above_growth_time'] = np.maximum(time - t2, 0)
        X['relu_above_peak_time'] = np.maximum(time - t3, 0)
        X['relu_above_overaging_time'] = np.maximum(time - t4, 0)

        # 软门控，帮助模型捕捉临界点附近突变
        X['soft_temp_regime_gate'] = self._sigmoid((temp - T1) / 6.0)
        X['soft_peak_temp_gate'] = self._sigmoid((temp - T2) / 6.0) * self._sigmoid((T3 - temp) / 6.0)
        X['soft_overaging_temp_gate'] = self._sigmoid((temp - T4) / 6.0)

        X['soft_growth_time_gate'] = self._sigmoid((time - t2) / 1.5)
        X['soft_peak_time_gate'] = self._sigmoid((time - t2) / 1.5) * self._sigmoid((t3 - time) / 1.5)
        X['soft_overaging_time_gate'] = self._sigmoid((time - t4) / 2.0)

        # -----------------------------
        # 5. 物理机制驱动力
        # -----------------------------
        temp_peak_proximity = np.exp(-((temp - p['peak_temp_center']) / p['peak_temp_width']) ** 2)
        time_peak_proximity = np.exp(-((log_time - p['peak_log_time_center']) / p['peak_log_time_width']) ** 2)
        joint_peak_proximity = temp_peak_proximity * time_peak_proximity

        X['temp_peak_proximity'] = temp_peak_proximity
        X['time_peak_proximity'] = time_peak_proximity
        X['joint_peak_proximity'] = joint_peak_proximity

        precipitation_drive = (
            0.55 * X['aging_saturation'].values
            + 0.85 * joint_peak_proximity
            + 0.15 * is_transition_temp * log_time
            + 0.20 * is_peak_temp * log_time
        )
        X['precipitation_drive'] = precipitation_drive

        overaging_drive = (
            X['soft_overaging_temp_gate'].values * X['soft_overaging_time_gate'].values
            + 0.02 * np.maximum(temp - T4, 0) * np.maximum(time - t4, 0)
        )
        X['overaging_drive'] = overaging_drive

        recovery_drive = (
            0.6 * X['soft_overaging_temp_gate'].values
            + 0.4 * X['soft_overaging_time_gate'].values
            + 0.1 * is_long_time
        )
        X['recovery_drive'] = recovery_drive

        X['net_strengthening_index'] = X['precipitation_drive'].values - 0.9 * X['overaging_drive'].values
        X['strength_ductility_tradeoff_index'] = X['net_strengthening_index'].values - 0.5 * X['recovery_drive'].values

        # -----------------------------
        # 6. 高风险误差区特征
        # 聚焦最可能出现非单调和误差大的区域：
        # - 高温短时：快速强化但尚未稳定
        # - 中高温长时：过时效
        # - 低温长时：等效慢时效
        # -----------------------------
        X['highT_shortt_index'] = is_over_temp * np.exp(-time / max(t2, 1e-6))
        X['midhighT_longt_index'] = np.maximum(temp - T3, 0) * np.maximum(log_time - np.log1p(t3), 0)
        X['lowT_longt_index'] = np.maximum(T2 - temp, 0) * np.maximum(log_time - np.log1p(t4), 0)

        X['equivalent_lowT_longt_vs_highT_shortt'] = self._safe_divide(
            np.maximum(T2 - temp, 0) * np.maximum(time, 0),
            1.0 + np.maximum(temp - T4, 0)
        )

        # -----------------------------
        # 7. 分区交互项
        # -----------------------------
        X['temp_in_low_temp'] = temp * is_low_temp
        X['temp_in_peak_temp'] = temp * is_peak_temp
        X['temp_in_over_temp'] = temp * is_over_temp

        X['log_time_in_short_time'] = log_time * is_short_time
        X['log_time_in_peak_time'] = log_time * is_peak_time
        X['log_time_in_over_time'] = log_time * is_over_time

        X['temp_log_time_peak_zone'] = temp * log_time * X['is_near_peak_aging'].values
        X['temp_log_time_overaged_zone'] = temp * log_time * X['is_overaged'].values
        X['temp_log_time_underaged_zone'] = temp * log_time * X['is_underaged'].values

        X['peak_zone_arrhenius'] = arrhenius * X['is_near_peak_aging'].values
        X['overaged_zone_arrhenius'] = arrhenius * X['is_overaged'].values

        # -----------------------------
        # 8. 基线预测值作为特征（让模型学残差）
        # -----------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        # 基线派生关系
        X['baseline_yield_to_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield
        X['baseline_strength_sum'] = baseline_tensile + baseline_yield
        X['baseline_strength_ductility_balance'] = self._safe_divide(
            baseline_tensile + baseline_yield, baseline_strain
        )

        # 相对原始铸态的基线提升量
        X['baseline_delta_strain_vs_base'] = baseline_strain - p['base_strain']
        X['baseline_delta_tensile_vs_base'] = baseline_tensile - p['base_tensile']
        X['baseline_delta_yield_vs_base'] = baseline_yield - p['base_yield']

        # -----------------------------
        # 9. 相对基线/相对边界的残差式特征
        # 虽然没有真实标签残差，但可构造“偏离物理基线”的输入状态特征
        # -----------------------------
        X['temp_minus_peak_center'] = temp - p['peak_temp_center']
        X['log_time_minus_peak_center'] = log_time - p['peak_log_time_center']

        X['temp_to_peak_center_ratio'] = self._safe_divide(temp, p['peak_temp_center'])
        X['time_to_peak_time_ratio'] = self._safe_divide(time, p['reference_time_h'])

        X['distance_to_peak_window'] = (
            ((temp - p['peak_temp_center']) / p['peak_temp_width']) ** 2
            + ((log_time - p['peak_log_time_center']) / p['peak_log_time_width']) ** 2
        )

        X['baseline_weighted_peak_distance'] = X['distance_to_peak_window'].values * X['baseline_strength_sum'].values
        X['baseline_weighted_overaging_risk'] = X['baseline_strength_sum'].values * X['overaging_drive'].values
        X['baseline_weighted_precipitation'] = X['baseline_strength_sum'].values * X['precipitation_drive'].values

        # 工艺状态相对机制边界的综合描述
        X['process_relative_to_boundary_combo'] = (
            0.03 * X['temp_relative_to_peak_lower'].values
            + 0.08 * X['time_relative_to_growth'].values
        )
        X['process_relative_to_overaging_combo'] = (
            0.04 * X['temp_relative_to_overaging'].values
            + 0.10 * X['time_relative_to_overaging'].values
        )

        # -----------------------------
        # 10. 塑性与强度竞争特征
        # -----------------------------
        X['ductility_recovery_index'] = (
            0.5 * X['recovery_drive'].values
            + 0.6 * X['overaging_drive'].values
            - 0.35 * X['precipitation_drive'].values
        )

        X['expected_strength_bias'] = (
            X['baseline_tensile_minus_yield'].values
            * (1.0 + 0.2 * X['overaging_drive'].values)
        )

        # -----------------------------
        # 11. 截断/稳健特征
        # -----------------------------
        clipped_temp = np.clip(temp, 40, 250)
        clipped_time = np.clip(time, 0, 48)
        clipped_temp_k = clipped_temp + 273.15

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)
        X['clipped_arrhenius'] = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * clipped_temp_k))
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['temp_activation'] = self._sigmoid((clipped_temp - T1) / 10.0)
        X['combined_activation'] = X['time_saturation'].values * X['temp_activation'].values

        # -----------------------------
        # 12. 常数先验特征
        # -----------------------------
        X['base_strain_const'] = p['base_strain']
        X['base_tensile_const'] = p['base_tensile']
        X['base_yield_const'] = p['base_yield']
        X['base_yield_tensile_ratio_const'] = self._safe_divide(p['base_yield'], p['base_tensile'])

        # -----------------------------
        # 13. 清理数值
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