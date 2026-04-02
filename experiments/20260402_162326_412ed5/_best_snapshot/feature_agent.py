import pandas as pd
import numpy as np


# 基于7499铝合金时效/热处理的一般材料学知识构造的领域参数
# 注：这里不追求严格合金专属常数，而是提供对小样本建模有帮助的“机制边界 + 动力学尺度”
DOMAIN_PARAMS = {
    # ---------------------------
    # 基础物理常数
    # ---------------------------
    'gas_constant_R': 8.314,                 # J/(mol*K)
    'activation_energy_Q': 72000.0,          # J/mol，铝合金析出/扩散控制过程的经验量级

    # ---------------------------
    # 机制边界（Regime boundaries）
    # 这些阈值用于显式提示模型：温度/时间跨过某点后机制可能发生变化
    # ---------------------------
    'critical_temp_regime_shift': 120.0,     # 析出强化开始明显加速的温度点
    'critical_temp_peak_age': 145.0,         # 可能接近峰值时效/强化主导中心区
    'critical_temp_overage': 165.0,          # 过时效/粗化/软化风险显著上升温度
    'critical_temp_severe_overage': 185.0,   # 强过时效区

    'critical_time_nucleation': 1.5,         # 短时成核/初期响应边界
    'critical_time_growth': 6.0,             # 强化相快速发展区边界
    'critical_time_peak_window': 10.0,       # 峰值时效典型时间尺度
    'critical_time_overage': 16.0,           # 长时过时效边界
    'critical_time_severe_overage': 24.0,    # 强过时效时间尺度

    # ---------------------------
    # 峰值时效经验中心
    # ---------------------------
    'peak_temp_center': 140.0,
    'peak_time_center': 8.0,
    'peak_temp_width': 18.0,
    'peak_log_time_width': 0.75,

    # ---------------------------
    # 原始铸态基线
    # ---------------------------
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # ---------------------------
    # 基线物理模型的经验幅值
    # ---------------------------
    'max_tensile_increment': 240.0,          # 热处理相对铸态可能带来的强度提升尺度
    'max_yield_increment': 185.0,
    'max_strain_drop': 2.2,                  # 峰值强化时塑性可能下降的尺度
    'max_strain_recovery': 1.6,              # 过时效后塑性回升尺度

    # ---------------------------
    # Larson-Miller 风格参数
    # ---------------------------
    'lmp_constant_C': 20.0
}


class FeatureAgent:
    """基于热处理物理机制 + 动力学等效尺度的特征工程"""

    def __init__(self):
        self.feature_names = []
        self.params = DOMAIN_PARAMS
        self.R = self.params['gas_constant_R']
        self.Q = self.params['activation_energy_Q']

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
        基于领域知识构造不依赖标签拟合的物理基线预测
        返回:
            baseline_strain, baseline_tensile, baseline_yield
        逻辑：
        1) 低温短时：强化不足，接近铸态
        2) 中温中时：析出强化增强，强度上升
        3) 高温长时：过时效，强度回落、塑性恢复
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        p = self.params
        base_strain = p['base_strain']
        base_tensile = p['base_tensile']
        base_yield = p['base_yield']

        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # Arrhenius 等效热暴露
        arrhenius = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        thermal_dose = time * arrhenius
        log_thermal_dose = np.log1p(np.clip(thermal_dose * 1e10, 0, None))

        # 温度激活：表示析出动力学是否被明显激活
        temp_activation = self._sigmoid(
            (temp - p['critical_temp_regime_shift']) / 10.0
        )

        # 时间发展：表示析出随时间推进
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / p['critical_time_growth'])

        # 峰值邻近：中温中时强度最高
        peak_temp_term = np.exp(
            -((temp - p['peak_temp_center']) / p['peak_temp_width']) ** 2
        )
        peak_time_term = np.exp(
            -((log_time - np.log1p(p['peak_time_center'])) / p['peak_log_time_width']) ** 2
        )
        peak_window = peak_temp_term * peak_time_term

        # 强化驱动：温度激活 × 时间推进 × 峰值窗口 + 热暴露项
        strengthening = (
            0.55 * temp_activation * time_activation
            + 0.30 * peak_window
            + 0.15 * self._sigmoid(log_thermal_dose - 0.25)
        )
        strengthening = np.clip(strengthening, 0, 1.2)

        # 过时效驱动：高温 + 长时
        overage_temp = self._sigmoid((temp - p['critical_temp_overage']) / 8.0)
        overage_time = self._sigmoid((time - p['critical_time_overage']) / 3.0)
        severe_overage_temp = self._sigmoid((temp - p['critical_temp_severe_overage']) / 6.0)
        severe_overage_time = self._sigmoid((time - p['critical_time_severe_overage']) / 4.0)

        overaging = (
            0.75 * overage_temp * overage_time
            + 0.25 * severe_overage_temp * severe_overage_time
        )
        overaging = np.clip(overaging, 0, 1.5)

        # 基线强度：先升后降
        baseline_tensile = (
            base_tensile
            + p['max_tensile_increment'] * strengthening
            - 110.0 * overaging
        )
        baseline_yield = (
            base_yield
            + p['max_yield_increment'] * strengthening
            - 95.0 * overaging
        )

        # 塑性：峰值强化时下降，过时效后回升
        baseline_strain = (
            base_strain
            - p['max_strain_drop'] * strengthening
            + p['max_strain_recovery'] * overaging
            + 0.35 * self._sigmoid((temp - p['critical_temp_overage']) / 10.0)
        )

        # 合理截断，增强稳定性
        baseline_tensile = np.clip(baseline_tensile, 80, 500)
        baseline_yield = np.clip(baseline_yield, 50, 420)
        baseline_strain = np.clip(baseline_strain, 2.0, 15.0)

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)
        p = self.params

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-12, None)

        # ---------------------------
        # 1. 原始基础特征
        # ---------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # ---------------------------
        # 2. 常规非线性
        # ---------------------------
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['temp_cube_scaled'] = (temp ** 3) / 1e4
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # ---------------------------
        # 3. 动力学等效特征
        # ---------------------------
        arrhenius_factor = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        thermal_dose = time * arrhenius_factor
        scaled_thermal_dose = thermal_dose * 1e10

        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = thermal_dose
        X['scaled_thermal_dose'] = scaled_thermal_dose
        X['log_thermal_dose'] = np.log1p(np.clip(scaled_thermal_dose, 0, None))
        X['temp_log_thermal_dose'] = temp * X['log_thermal_dose'].values

        # Larson-Miller Parameter 风格
        # 常用于把温度-时间映射成一个等效标量
        lmp = temp_k * (p['lmp_constant_C'] + np.log10(np.clip(time, 1e-6, None)))
        X['larson_miller_param'] = lmp
        X['lmp_scaled'] = lmp / 1000.0

        # Hollomon/JMAK 风格启发
        X['jmak_like_time_power'] = np.power(np.clip(time, 0, None), 0.5)
        X['jmak_like_drive'] = arrhenius_factor * np.power(np.clip(time, 0, None) + 1e-6, 0.5)
        X['log_jmak_like_drive'] = np.log1p(X['jmak_like_drive'].values * 1e10)

        # ---------------------------
        # 4. 机制区间检测特征（关键）
        # ---------------------------
        T1 = p['critical_temp_regime_shift']
        T2 = p['critical_temp_peak_age']
        T3 = p['critical_temp_overage']
        T4 = p['critical_temp_severe_overage']

        t1 = p['critical_time_nucleation']
        t2 = p['critical_time_growth']
        t3 = p['critical_time_peak_window']
        t4 = p['critical_time_overage']
        t5 = p['critical_time_severe_overage']

        X['is_low_temp'] = (temp < T1).astype(float)
        X['is_precip_active_temp'] = ((temp >= T1) & (temp < T2)).astype(float)
        X['is_peak_age_temp'] = ((temp >= T2) & (temp < T3)).astype(float)
        X['is_overage_temp'] = ((temp >= T3) & (temp < T4)).astype(float)
        X['is_severe_overage_temp'] = (temp >= T4).astype(float)

        X['is_nucleation_time'] = (time < t1).astype(float)
        X['is_growth_time'] = ((time >= t1) & (time < t2)).astype(float)
        X['is_peak_window_time'] = ((time >= t2) & (time < t3)).astype(float)
        X['is_overage_time'] = ((time >= t4) & (time < t5)).astype(float)
        X['is_severe_overage_time'] = (time >= t5).astype(float)

        # 显式机制组合
        X['is_under_aged'] = ((temp < T2) & (time < t2)).astype(float)
        X['is_near_peak_aged'] = ((temp >= T1) & (temp < T3) & (time >= t2) & (time < t4)).astype(float)
        X['is_overaged'] = ((temp >= T3) & (time >= t4)).astype(float)
        X['is_severely_overaged'] = ((temp >= T4) & (time >= t5)).astype(float)

        # 高温短时：可能与低温长时在某些组织状态上接近
        X['is_high_temp_short_time'] = ((temp >= T3) & (time < t2)).astype(float)
        X['is_low_temp_long_time'] = ((temp < T1) & (time >= t4)).astype(float)

        # ---------------------------
        # 5. 相对机制边界距离特征
        # ---------------------------
        X['temp_relative_to_regime_shift'] = temp - T1
        X['temp_relative_to_peak_age'] = temp - T2
        X['temp_relative_to_overage'] = temp - T3
        X['temp_relative_to_severe_overage'] = temp - T4

        X['time_relative_to_nucleation'] = time - t1
        X['time_relative_to_growth'] = time - t2
        X['time_relative_to_peak_window'] = time - t3
        X['time_relative_to_overage'] = time - t4
        X['time_relative_to_severe_overage'] = time - t5

        X['relu_temp_above_regime_shift'] = np.maximum(temp - T1, 0)
        X['relu_temp_above_peak_age'] = np.maximum(temp - T2, 0)
        X['relu_temp_above_overage'] = np.maximum(temp - T3, 0)
        X['relu_temp_above_severe_overage'] = np.maximum(temp - T4, 0)

        X['relu_time_above_growth'] = np.maximum(time - t2, 0)
        X['relu_time_above_peak_window'] = np.maximum(time - t3, 0)
        X['relu_time_above_overage'] = np.maximum(time - t4, 0)
        X['relu_time_above_severe_overage'] = np.maximum(time - t5, 0)

        # ---------------------------
        # 6. 平滑门控特征
        # ---------------------------
        X['temp_activation_regime'] = self._sigmoid((temp - T1) / 10.0)
        X['temp_activation_peak'] = self._sigmoid((temp - T2) / 8.0)
        X['temp_activation_overage'] = self._sigmoid((temp - T3) / 8.0)
        X['temp_activation_severe_overage'] = self._sigmoid((temp - T4) / 6.0)

        X['time_activation_growth'] = self._sigmoid((time - t2) / 2.5)
        X['time_activation_peak_window'] = self._sigmoid((time - t3) / 2.5)
        X['time_activation_overage'] = self._sigmoid((time - t4) / 3.0)
        X['time_activation_severe_overage'] = self._sigmoid((time - t5) / 4.0)

        # ---------------------------
        # 7. 峰值时效邻近
        # ---------------------------
        peak_temp_proximity = -((temp - p['peak_temp_center']) / p['peak_temp_width']) ** 2
        peak_time_proximity = -((log_time - np.log1p(p['peak_time_center'])) / p['peak_log_time_width']) ** 2
        joint_peak_proximity = peak_temp_proximity + peak_time_proximity
        peak_window = np.exp(joint_peak_proximity)

        X['temp_peak_proximity'] = peak_temp_proximity
        X['time_peak_proximity'] = peak_time_proximity
        X['joint_peak_proximity'] = joint_peak_proximity
        X['peak_window'] = peak_window

        # ---------------------------
        # 8. 强化/软化竞争特征
        # ---------------------------
        precipitation_drive = (
            0.45 * X['temp_activation_regime'].values * (1.0 - X['temp_activation_overage'].values)
            + 0.30 * (1.0 - np.exp(-np.clip(time, 0, None) / t2))
            + 0.25 * peak_window
        )

        softening_drive = (
            0.55 * X['temp_activation_overage'].values * X['time_activation_overage'].values
            + 0.25 * X['temp_activation_severe_overage'].values * X['time_activation_severe_overage'].values
            + 0.20 * self._sigmoid(X['log_thermal_dose'].values - 1.0)
        )

        net_strengthening = precipitation_drive - softening_drive

        X['precipitation_drive'] = precipitation_drive
        X['softening_drive'] = softening_drive
        X['net_strengthening_index'] = net_strengthening

        # 过时效严重度
        X['overaging_severity'] = (
            X['temp_activation_overage'].values * X['time_activation_overage'].values
            + X['temp_activation_severe_overage'].values * X['time_activation_severe_overage'].values
        )

        # ---------------------------
        # 9. 分段交互
        # ---------------------------
        X['temp_in_low_temp_zone'] = temp * X['is_low_temp'].values
        X['temp_in_peak_zone'] = temp * X['is_peak_age_temp'].values
        X['temp_in_overage_zone'] = temp * X['is_overage_temp'].values
        X['temp_in_severe_overage_zone'] = temp * X['is_severe_overage_temp'].values

        X['log_time_in_growth_zone'] = log_time * X['is_growth_time'].values
        X['log_time_in_peak_window_zone'] = log_time * X['is_peak_window_time'].values
        X['log_time_in_overage_zone'] = log_time * X['is_overage_time'].values
        X['log_time_in_severe_overage_zone'] = log_time * X['is_severe_overage_time'].values

        X['temp_log_time_peak_zone'] = temp * log_time * X['is_near_peak_aged'].values
        X['temp_log_time_overaged_zone'] = temp * log_time * X['is_overaged'].values
        X['temp_log_time_severe_overaged_zone'] = temp * log_time * X['is_severely_overaged'].values

        # ---------------------------
        # 10. 强塑性权衡启发
        # ---------------------------
        X['ductility_recovery_index'] = (
            0.6 * softening_drive
            + 0.2 * X['is_overaged'].values
            + 0.2 * X['is_severely_overaged'].values
            - 0.35 * precipitation_drive
        )
        X['strength_ductility_tradeoff'] = net_strengthening - X['ductility_recovery_index'].values

        # ---------------------------
        # 11. 物理基线预测作为特征（让模型学残差）
        # ---------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        # 基线派生关系
        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_strength_sum'] = baseline_tensile + baseline_yield
        X['baseline_strength_diff'] = baseline_tensile - baseline_yield

        # 用工艺特征对基线进行“残差校正提示”
        X['baseline_tensile_x_peak_window'] = baseline_tensile * peak_window
        X['baseline_yield_x_peak_window'] = baseline_yield * peak_window
        X['baseline_strain_x_overaging'] = baseline_strain * X['overaging_severity'].values

        # ---------------------------
        # 12. 相对基线的残差型特征（无标签）
        # 不是“真实残差”，而是“工艺相对基线状态偏移”
        # ---------------------------
        X['temp_minus_peak_center'] = temp - p['peak_temp_center']
        X['log_time_minus_peak_center'] = log_time - np.log1p(p['peak_time_center'])

        X['baseline_relative_peak_temp_offset'] = self._safe_divide(
            temp - p['peak_temp_center'], p['peak_temp_width']
        )
        X['baseline_relative_peak_time_offset'] = self._safe_divide(
            log_time - np.log1p(p['peak_time_center']), p['peak_log_time_width']
        )

        X['baseline_relative_regime_temp'] = self._safe_divide(temp - T1, T1)
        X['baseline_relative_overage_temp'] = self._safe_divide(temp - T3, T3)
        X['baseline_relative_overage_time'] = self._safe_divide(time - t4, t4)

        X['process_minus_baseline_strength_scale'] = (
            net_strengthening * (baseline_tensile + baseline_yield) / 2.0
        )
        X['process_minus_baseline_ductility_scale'] = (
            X['ductility_recovery_index'].values * baseline_strain
        )

        # ---------------------------
        # 13. 基于铸态常数的相对潜力特征
        # ---------------------------
        base_tensile = p['base_tensile']
        base_yield = p['base_yield']
        base_strain = p['base_strain']

        X['base_tensile'] = base_tensile
        X['base_yield'] = base_yield
        X['base_strain'] = base_strain

        X['baseline_tensile_gain_over_cast'] = baseline_tensile - base_tensile
        X['baseline_yield_gain_over_cast'] = baseline_yield - base_yield
        X['baseline_strain_change_over_cast'] = baseline_strain - base_strain

        X['tensile_gain_ratio_over_cast'] = self._safe_divide(
            baseline_tensile - base_tensile, base_tensile
        )
        X['yield_gain_ratio_over_cast'] = self._safe_divide(
            baseline_yield - base_yield, base_yield
        )
        X['strain_change_ratio_over_cast'] = self._safe_divide(
            baseline_strain - base_strain, base_strain
        )

        # ---------------------------
        # 14. 额外稳健截断特征
        # ---------------------------
        clipped_temp = np.clip(temp, 60, 220)
        clipped_time = np.clip(time, 0, 48)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['temp_saturation'] = self._sigmoid((clipped_temp - T1) / 10.0)
        X['combined_saturation'] = X['time_saturation'].values * X['temp_saturation'].values

        X['high_temp_long_time_penalty'] = (
            np.maximum(clipped_temp - T3, 0) * np.maximum(clipped_time - t4, 0)
        )
        X['severe_overage_penalty'] = (
            np.maximum(clipped_temp - T4, 0) * np.maximum(clipped_time - t5, 0)
        )

        # ---------------------------
        # 15. 最终清理
        # ---------------------------
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