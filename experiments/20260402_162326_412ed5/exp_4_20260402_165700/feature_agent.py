import pandas as pd
import numpy as np


# 7499 铝合金高温热处理/均匀化场景下的经验领域参数
# 这里温度单位直接按数据中的摄氏度处理（样本在 440/460/470°C），
# 与常见低温时效不同，此处更接近“高温组织演化/溶质再分配/粗化敏感区”。
DOMAIN_PARAMS = {
    # ---------------------------
    # 基础物理常数
    # ---------------------------
    'gas_constant_R': 8.314,                  # J/(mol*K)

    # 高温扩散/粗化控制的经验激活能量级
    'activation_energy_Q': 135000.0,          # J/mol

    # ---------------------------
    # 温度机制边界（针对当前数据分布重点放在 440-470°C）
    # ---------------------------
    'critical_temp_regime_shift': 445.0,      # 从“较弱演化”进入“明显组织演化敏感区”
    'critical_temp_peak_age': 455.0,          # 强化/均匀化收益较高的中间区
    'critical_temp_overage': 465.0,           # 过时效/粗化/软化风险显著上升
    'critical_temp_severe_overage': 472.0,    # 强过处理/组织失稳更明显的区间

    # ---------------------------
    # 时间机制边界
    # ---------------------------
    'critical_time_nucleation': 1.5,          # 极短时间，初始快速响应区
    'critical_time_growth': 6.0,              # 中短时，强化/组织调整快速发展区
    'critical_time_peak_window': 12.0,        # 当前数据里 12h 是关键敏感点
    'critical_time_overage': 18.0,            # 长时开始明显进入软化/粗化
    'critical_time_severe_overage': 24.0,     # 强长时暴露

    # ---------------------------
    # 峰值/最优工艺经验中心（更贴近现有样本区间）
    # ---------------------------
    'peak_temp_center': 452.0,
    'peak_time_center': 10.0,
    'peak_temp_width': 10.0,
    'peak_log_time_width': 0.70,

    # ---------------------------
    # 误差热点专项边界
    # ---------------------------
    'hotspot_temp_440': 440.0,
    'hotspot_temp_460': 460.0,
    'hotspot_temp_470': 470.0,
    'hotspot_time_1h': 1.0,
    'hotspot_time_12h': 12.0,
    'hotspot_time_24h': 24.0,

    # ---------------------------
    # 原始铸态基线
    # ---------------------------
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # ---------------------------
    # 热处理相对铸态可能产生的性能变化尺度
    # ---------------------------
    'max_tensile_increment': 220.0,
    'max_yield_increment': 170.0,
    'max_strain_drop': 2.0,
    'max_strain_recovery': 2.4,

    # ---------------------------
    # LMP 风格参数
    # ---------------------------
    'lmp_constant_C': 20.0,

    # ---------------------------
    # 470°C@12h 等敏感区的局部宽度
    # ---------------------------
    'hotspot_temp_width': 4.0,
    'hotspot_log_time_width': 0.22,
}


class FeatureAgent:
    """基于高温热处理组织演化机制的特征工程"""

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

    def _gaussian(self, x, center, width):
        width = max(width, 1e-12)
        return np.exp(-((x - center) / width) ** 2)

    def physics_baseline(self, temp, time):
        """
        不依赖标签的物理启发基线：
        - 440°C / 1h：处理不足或组织尚未充分调整，强度提升有限
        - 450~460°C / 中等时间：性能提升较大
        - 470°C / 12h 或更高热暴露：可能进入敏感转折区，强度波动/回落、塑性回升
        - 440°C / 24h：较低温长时，可部分逼近中温中时的等效效果，但存在上限
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        p = self.params
        base_strain = p['base_strain']
        base_tensile = p['base_tensile']
        base_yield = p['base_yield']

        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # 动力学热暴露
        arrhenius = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        thermal_dose = time * arrhenius
        log_thermal_dose = np.log1p(np.clip(thermal_dose * 1e18, 0, None))

        # 温度、时间激活
        temp_activation = self._sigmoid((temp - p['critical_temp_regime_shift']) / 4.5)
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / p['critical_time_growth'])

        # 峰值强化窗口
        peak_temp_term = self._gaussian(temp, p['peak_temp_center'], p['peak_temp_width'])
        peak_time_term = self._gaussian(log_time, np.log1p(p['peak_time_center']), p['peak_log_time_width'])
        peak_window = peak_temp_term * peak_time_term

        # 低温长时等效
        low_temp_long_time = self._sigmoid((445.0 - temp) / 4.0) * self._sigmoid((time - 18.0) / 3.0)

        # 中温中时强化驱动
        strengthening = (
            0.38 * temp_activation * time_activation
            + 0.34 * peak_window
            + 0.16 * self._sigmoid(log_thermal_dose - 3.2)
            + 0.12 * low_temp_long_time
        )
        strengthening = np.clip(strengthening, 0.0, 1.2)

        # 过处理/粗化/软化
        overage_temp = self._sigmoid((temp - p['critical_temp_overage']) / 2.8)
        overage_time = self._sigmoid((time - p['critical_time_peak_window']) / 2.6)
        severe_overage_temp = self._sigmoid((temp - p['critical_temp_severe_overage']) / 2.0)
        severe_overage_time = self._sigmoid((time - p['critical_time_overage']) / 2.5)

        overaging = (
            0.65 * overage_temp * overage_time
            + 0.35 * severe_overage_temp * severe_overage_time
        )
        overaging = np.clip(overaging, 0.0, 1.5)

        # 470°C@12h 为已知高误差敏感区：加入转折/不稳定窗口
        hotspot_470_12 = self._gaussian(temp, 470.0, 4.0) * self._gaussian(log_time, np.log1p(12.0), 0.22)
        hotspot_440_1 = self._gaussian(temp, 440.0, 3.5) * self._gaussian(log_time, np.log1p(1.0), 0.18)
        hotspot_440_24 = self._gaussian(temp, 440.0, 3.5) * self._gaussian(log_time, np.log1p(24.0), 0.20)

        # 强度基线：先升后降，并在热点区加入非单调修正
        baseline_tensile = (
            base_tensile
            + p['max_tensile_increment'] * strengthening
            - 92.0 * overaging
            - 26.0 * hotspot_470_12
            - 10.0 * hotspot_440_1
            + 8.0 * hotspot_440_24
        )

        baseline_yield = (
            base_yield
            + p['max_yield_increment'] * strengthening
            - 82.0 * overaging
            - 18.0 * hotspot_470_12
            - 6.0 * hotspot_440_1
            + 10.0 * hotspot_440_24
        )

        # 塑性：强化时下降，过处理和局部热点下回升
        baseline_strain = (
            base_strain
            - p['max_strain_drop'] * strengthening
            + p['max_strain_recovery'] * overaging
            + 1.20 * hotspot_470_12
            - 0.55 * hotspot_440_1
            + 0.35 * hotspot_440_24
        )

        baseline_tensile = np.clip(baseline_tensile, 80, 420)
        baseline_yield = np.clip(baseline_yield, 50, 340)
        baseline_strain = np.clip(baseline_strain, 2.0, 16.0)

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
        X['temp_cube_scaled'] = (temp ** 3) / 1e5
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
        scaled_thermal_dose = thermal_dose * 1e18

        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = thermal_dose
        X['scaled_thermal_dose'] = scaled_thermal_dose
        X['log_thermal_dose'] = np.log1p(np.clip(scaled_thermal_dose, 0, None))
        X['temp_log_thermal_dose'] = temp * X['log_thermal_dose'].values

        lmp = temp_k * (p['lmp_constant_C'] + np.log10(np.clip(time, 1e-6, None)))
        X['larson_miller_param'] = lmp
        X['lmp_scaled'] = lmp / 1000.0

        X['jmak_like_time_power'] = np.power(np.clip(time, 0, None), 0.5)
        X['jmak_like_drive'] = arrhenius_factor * np.power(np.clip(time, 0, None) + 1e-6, 0.5)
        X['log_jmak_like_drive'] = np.log1p(X['jmak_like_drive'].values * 1e18)

        # Zener-Hollomon 风格（仅作等效尺度，不追求严格物理量纲）
        zh_like = np.log1p(np.exp(self.Q / np.clip(self.R * temp_k, 1e-12, None)) / np.clip(time + 1.0, 1e-12, None))
        X['zener_hollomon_like'] = zh_like

        # ---------------------------
        # 4. 机制区间检测特征
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
        X['is_peak_window_time'] = ((time >= t2) & (time < t4)).astype(float)
        X['is_overage_time'] = ((time >= t4) & (time < t5)).astype(float)
        X['is_severe_overage_time'] = (time >= t5).astype(float)

        X['is_under_processed'] = ((temp < T1) & (time < t2)).astype(float)
        X['is_near_peak_aged'] = ((temp >= T1) & (temp < T3) & (time >= t2) & (time <= t3)).astype(float)
        X['is_transition_sensitive'] = ((temp >= T3) & (time >= t2) & (time <= t4)).astype(float)
        X['is_overaged'] = ((temp >= T3) & (time >= t3)).astype(float)
        X['is_severely_overaged'] = ((temp >= T4) & (time >= t4)).astype(float)

        X['is_high_temp_short_time'] = ((temp >= T3) & (time < t2)).astype(float)
        X['is_low_temp_long_time'] = ((temp < T1) & (time >= t4)).astype(float)
        X['is_440_1_like'] = ((np.abs(temp - 440.0) <= 2.5) & (np.abs(time - 1.0) <= 0.5)).astype(float)
        X['is_460_12_like'] = ((np.abs(temp - 460.0) <= 2.5) & (np.abs(time - 12.0) <= 1.5)).astype(float)
        X['is_470_12_like'] = ((np.abs(temp - 470.0) <= 2.5) & (np.abs(time - 12.0) <= 1.5)).astype(float)
        X['is_440_24_like'] = ((np.abs(temp - 440.0) <= 2.5) & (np.abs(time - 24.0) <= 2.0)).astype(float)

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
        X['temp_activation_regime'] = self._sigmoid((temp - T1) / 4.0)
        X['temp_activation_peak'] = self._sigmoid((temp - T2) / 3.0)
        X['temp_activation_overage'] = self._sigmoid((temp - T3) / 2.8)
        X['temp_activation_severe_overage'] = self._sigmoid((temp - T4) / 2.0)

        X['time_activation_growth'] = self._sigmoid((time - t2) / 1.6)
        X['time_activation_peak_window'] = self._sigmoid((time - t3) / 1.8)
        X['time_activation_overage'] = self._sigmoid((time - t4) / 2.0)
        X['time_activation_severe_overage'] = self._sigmoid((time - t5) / 2.0)

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
        # 8. 误差热点工艺的局部特征（重点）
        # ---------------------------
        hotspot_temp_width = p['hotspot_temp_width']
        hotspot_log_time_width = p['hotspot_log_time_width']

        h470_12 = self._gaussian(temp, 470.0, hotspot_temp_width) * self._gaussian(log_time, np.log1p(12.0), hotspot_log_time_width)
        h460_12 = self._gaussian(temp, 460.0, hotspot_temp_width) * self._gaussian(log_time, np.log1p(12.0), hotspot_log_time_width)
        h440_24 = self._gaussian(temp, 440.0, hotspot_temp_width) * self._gaussian(log_time, np.log1p(24.0), hotspot_log_time_width)
        h440_1 = self._gaussian(temp, 440.0, hotspot_temp_width) * self._gaussian(log_time, np.log1p(1.0), 0.18)

        X['hotspot_470_12'] = h470_12
        X['hotspot_460_12'] = h460_12
        X['hotspot_440_24'] = h440_24
        X['hotspot_440_1'] = h440_1

        X['dist_temp_to_470'] = np.abs(temp - 470.0)
        X['dist_temp_to_460'] = np.abs(temp - 460.0)
        X['dist_temp_to_440'] = np.abs(temp - 440.0)
        X['dist_time_to_12'] = np.abs(time - 12.0)
        X['dist_time_to_24'] = np.abs(time - 24.0)
        X['dist_time_to_1'] = np.abs(time - 1.0)

        # 体现“470/12 是组织转折敏感区”
        X['transition_instability_index'] = (
            h470_12 * (
                0.45
                + 0.30 * X['temp_activation_overage'].values
                + 0.25 * X['time_activation_peak_window'].values
            )
        )

        # 440/1：早期不足处理
        X['underprocess_index'] = (
            h440_1 * (
                0.5
                + 0.3 * (1.0 - X['time_activation_growth'].values)
                + 0.2 * (1.0 - X['temp_activation_regime'].values)
            )
        )

        # 440/24：低温长时等效
        X['low_temp_long_time_equivalent_index'] = (
            h440_24 * (
                0.5
                + 0.25 * X['is_low_temp_long_time'].values
                + 0.25 * X['log_thermal_dose'].values / (1.0 + X['log_thermal_dose'].values)
            )
        )

        # ---------------------------
        # 9. 强化/软化竞争特征
        # ---------------------------
        precipitation_drive = (
            0.34 * X['temp_activation_regime'].values * (1.0 - X['temp_activation_overage'].values)
            + 0.22 * (1.0 - np.exp(-np.clip(time, 0, None) / t2))
            + 0.22 * peak_window
            + 0.12 * X['low_temp_long_time_equivalent_index'].values
            + 0.10 * h460_12
        )

        softening_drive = (
            0.42 * X['temp_activation_overage'].values * X['time_activation_peak_window'].values
            + 0.23 * X['temp_activation_severe_overage'].values * X['time_activation_overage'].values
            + 0.20 * self._sigmoid(X['log_thermal_dose'].values - 3.8)
            + 0.15 * h470_12
        )

        net_strengthening = precipitation_drive - softening_drive

        X['precipitation_drive'] = precipitation_drive
        X['softening_drive'] = softening_drive
        X['net_strengthening_index'] = net_strengthening

        X['overaging_severity'] = (
            X['temp_activation_overage'].values * X['time_activation_peak_window'].values
            + X['temp_activation_severe_overage'].values * X['time_activation_overage'].values
        )

        # ---------------------------
        # 10. 分段交互
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
        X['temp_log_time_transition_zone'] = temp * log_time * X['is_transition_sensitive'].values
        X['temp_log_time_overaged_zone'] = temp * log_time * X['is_overaged'].values
        X['temp_log_time_severe_overaged_zone'] = temp * log_time * X['is_severely_overaged'].values

        # 热点交互
        X['temp_log_time_x_470_12'] = temp * log_time * h470_12
        X['temp_log_time_x_440_1'] = temp * log_time * h440_1
        X['temp_log_time_x_440_24'] = temp * log_time * h440_24
        X['arrhenius_x_470_12'] = arrhenius_factor * h470_12
        X['arrhenius_x_440_24'] = arrhenius_factor * h440_24

        # ---------------------------
        # 11. 强塑性权衡启发
        # ---------------------------
        X['ductility_recovery_index'] = (
            0.62 * softening_drive
            + 0.18 * X['is_overaged'].values
            + 0.10 * X['is_severely_overaged'].values
            + 0.12 * h470_12
            - 0.30 * precipitation_drive
        )
        X['strength_ductility_tradeoff'] = net_strengthening - X['ductility_recovery_index'].values
        X['yield_sensitive_index'] = (
            0.55 * precipitation_drive
            - 0.35 * softening_drive
            - 0.15 * h470_12
            + 0.10 * h440_24
        )

        # ---------------------------
        # 12. 物理基线预测作为特征
        # ---------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_strength_sum'] = baseline_tensile + baseline_yield
        X['baseline_strength_diff'] = baseline_tensile - baseline_yield

        X['baseline_tensile_x_peak_window'] = baseline_tensile * peak_window
        X['baseline_yield_x_peak_window'] = baseline_yield * peak_window
        X['baseline_strain_x_overaging'] = baseline_strain * X['overaging_severity'].values

        # 基线与热点区交互
        X['baseline_tensile_x_470_12'] = baseline_tensile * h470_12
        X['baseline_yield_x_470_12'] = baseline_yield * h470_12
        X['baseline_strain_x_470_12'] = baseline_strain * h470_12

        X['baseline_tensile_x_440_1'] = baseline_tensile * h440_1
        X['baseline_yield_x_440_1'] = baseline_yield * h440_1

        X['baseline_tensile_x_440_24'] = baseline_tensile * h440_24
        X['baseline_yield_x_440_24'] = baseline_yield * h440_24

        # ---------------------------
        # 13. 相对基线/边界偏移特征
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
        # 14. 基于铸态常数的相对潜力特征
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
        # 15. 额外稳健截断特征
        # ---------------------------
        clipped_temp = np.clip(temp, 430, 480)
        clipped_time = np.clip(time, 0, 30)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['temp_saturation'] = self._sigmoid((clipped_temp - T1) / 4.0)
        X['combined_saturation'] = X['time_saturation'].values * X['temp_saturation'].values

        X['high_temp_long_time_penalty'] = (
            np.maximum(clipped_temp - T3, 0) * np.maximum(clipped_time - t3, 0)
        )
        X['severe_overage_penalty'] = (
            np.maximum(clipped_temp - T4, 0) * np.maximum(clipped_time - t4, 0)
        )

        # ---------------------------
        # 16. 重复工艺点/离散性提示特征
        # ---------------------------
        # 对 470/12、440/1、440/24 这些重复样本且误差波动大的组合，
        # 用局部“敏感性”特征帮助模型学习同一工艺条件下可能的高方差行为。
        X['local_instability_proxy'] = (
            0.45 * h470_12
            + 0.25 * h440_1
            + 0.20 * h440_24
            + 0.10 * h460_12
        )

        X['strength_instability_proxy'] = (
            X['local_instability_proxy'].values
            * (1.0 + X['overaging_severity'].values + np.maximum(net_strengthening, 0))
        )

        # ---------------------------
        # 17. 最终清理
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