import pandas as pd
import numpy as np


# 7499铝合金热处理/均匀化过程的领域启发参数
# 说明：
# 1) 当前数据温度范围是 440/460/470°C，明显不是传统低温人工时效区，而更像高温均匀化/固溶相关热暴露窗口
# 2) 误差最大的点集中在 460°C/12h、470°C/12h、440°C/24h、440°C/1h
#    说明模型尤其需要：
#    - 捕捉 460~470°C 附近的“高温敏感区”
#    - 捕捉 12h 左右的“中等保温转折区”
#    - 区分 440°C 下短时(1h)与长时(24h)的不同组织状态
# 3) 因此这里显式加入：
#    - 高温机制转折边界
#    - 时间机制边界
#    - 460/470°C 与 12h/24h 的局部敏感窗口
DOMAIN_PARAMS = {
    # 基础物理常数
    'gas_constant_R': 8.314,                    # J/(mol*K)

    # 高温扩散/均匀化/粗化相关过程的经验激活能量级
    'activation_energy_Q': 135000.0,           # J/mol

    # 原始铸态基线
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # ---------------------------
    # 机制边界：温度
    # ---------------------------
    # 440°C：相对较低的均匀化/热暴露水平
    'critical_temp_low': 445.0,

    # 460°C：误差最大之一，可能是析出相溶解/粗化/组织再分配的敏感区
    'critical_temp_regime_shift': 458.0,

    # 470°C：更高温端，可能接近另一类组织响应增强区
    'critical_temp_high_sensitive': 468.0,

    # 极高温风险边界（当前数据虽未必覆盖，但利于外推稳定）
    'critical_temp_severe': 478.0,

    # ---------------------------
    # 机制边界：时间
    # ---------------------------
    'critical_time_short': 1.5,                # 1h附近：短时、组织尚未充分演化
    'critical_time_transition': 8.0,           # 中间阶段
    'critical_time_sensitive': 12.0,           # 12h：误差最大敏感窗口
    'critical_time_long': 18.0,                # 长时开始
    'critical_time_very_long': 24.0,           # 24h：长时显著组织演化

    # ---------------------------
    # 局部热点中心（针对高误差工艺点）
    # ---------------------------
    'hotspot_temp_center_1': 460.0,
    'hotspot_temp_center_2': 470.0,
    'hotspot_time_center_1': 12.0,
    'hotspot_time_center_2': 24.0,
    'hotspot_temp_width': 6.0,
    'hotspot_time_width': 4.0,

    # ---------------------------
    # 热暴露/等效参数
    # ---------------------------
    'lmp_constant_C': 20.0,

    # ---------------------------
    # 基线预测幅值（经验启发）
    # ---------------------------
    'max_tensile_increment': 165.0,
    'max_yield_increment': 125.0,
    'max_strain_increment': 8.5,

    # 软化/脆化/组织粗化惩罚尺度
    'softening_tensile_penalty': 55.0,
    'softening_yield_penalty': 45.0,
    'strain_reduction_penalty': 3.0,
}


class FeatureAgent:
    """基于高温热暴露机制 + 区间边界 + 等效动力学参数的特征工程"""

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
        width = max(width, 1e-6)
        return np.exp(-((x - center) / width) ** 2)

    def physics_baseline(self, temp, time):
        """
        基于世界知识的高温热处理基线估计：
        - 高温+时间带来组织均匀化/缺陷缓解，强度和塑性均可能相对铸态提升
        - 但在 460~470°C、12h附近可能存在机制转折，表现出较强非线性
        - 长时间高温可能导致部分粗化/过度软化，尤其屈服和抗拉不再同步增长
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)
        p = self.params

        base_strain = p['base_strain']
        base_tensile = p['base_tensile']
        base_yield = p['base_yield']

        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # Arrhenius扩散尺度
        arrhenius = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        thermal_dose = time * arrhenius
        log_thermal_dose = np.log1p(thermal_dose * 1e18)

        # 总体激活：温度和时间共同促进组织演化
        temp_activation = self._sigmoid((temp - p['critical_temp_low']) / 6.0)
        high_temp_activation = self._sigmoid((temp - p['critical_temp_regime_shift']) / 4.0)
        extreme_temp_activation = self._sigmoid((temp - p['critical_temp_high_sensitive']) / 3.0)

        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / 6.0)
        mid_time_activation = self._sigmoid((time - p['critical_time_transition']) / 2.0)
        sensitive_time_activation = self._sigmoid((time - p['critical_time_sensitive']) / 1.8)
        long_time_activation = self._sigmoid((time - p['critical_time_long']) / 2.5)

        # 局部热点：专门针对大误差区
        hotspot_460_12 = self._gaussian(temp, p['hotspot_temp_center_1'], p['hotspot_temp_width']) * \
                         self._gaussian(time, p['hotspot_time_center_1'], p['hotspot_time_width'])

        hotspot_470_12 = self._gaussian(temp, p['hotspot_temp_center_2'], p['hotspot_temp_width']) * \
                         self._gaussian(time, p['hotspot_time_center_1'], p['hotspot_time_width'])

        hotspot_440_24 = self._gaussian(temp, 440.0, 5.0) * self._gaussian(time, 24.0, 4.0)
        hotspot_440_1 = self._gaussian(temp, 440.0, 5.0) * self._gaussian(time, 1.0, 1.2)

        # 强化/改善驱动：整体热暴露带来的性能改善
        strengthening = (
            0.30 * temp_activation +
            0.25 * time_activation +
            0.20 * mid_time_activation * high_temp_activation +
            0.15 * self._sigmoid(log_thermal_dose - 1.2) +
            0.10 * hotspot_440_24
        )
        strengthening = np.clip(strengthening, 0, 1.4)

        # 转折/软化驱动：高温中长时尤其影响屈服/抗拉
        softening = (
            0.35 * high_temp_activation * sensitive_time_activation +
            0.30 * extreme_temp_activation * mid_time_activation +
            0.20 * high_temp_activation * long_time_activation +
            0.15 * hotspot_460_12
        )
        softening = np.clip(softening, 0, 1.3)

        # 塑性恢复/脆性缓解：部分工艺下应变可明显提升
        ductility_gain = (
            0.35 * time_activation +
            0.25 * temp_activation +
            0.20 * hotspot_470_12 +
            0.20 * hotspot_440_24
        )
        ductility_gain = np.clip(ductility_gain, 0, 1.5)

        # 短时高温下塑性不足/不稳定
        early_instability = (
            self._gaussian(temp, 440.0, 6.0) * self._gaussian(time, 1.0, 1.0)
            + 0.6 * self._gaussian(temp, 460.0, 6.0) * self._gaussian(time, 1.0, 1.0)
        )
        early_instability = np.clip(early_instability, 0, 1.5)

        baseline_tensile = (
            base_tensile
            + p['max_tensile_increment'] * strengthening
            - p['softening_tensile_penalty'] * softening
            + 12.0 * hotspot_470_12
            - 10.0 * hotspot_440_1
        )

        baseline_yield = (
            base_yield
            + p['max_yield_increment'] * strengthening
            - p['softening_yield_penalty'] * (1.15 * softening)
            - 8.0 * hotspot_470_12
            - 6.0 * hotspot_460_12
        )

        baseline_strain = (
            base_strain
            + p['max_strain_increment'] * ductility_gain
            - p['strain_reduction_penalty'] * softening
            - 1.8 * early_instability
            + 1.0 * hotspot_470_12
        )

        baseline_tensile = np.clip(baseline_tensile, 100, 380)
        baseline_yield = np.clip(baseline_yield, 70, 280)
        baseline_strain = np.clip(baseline_strain, 2.0, 20.0)

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
        # 1. 原始与基础变换
        # ---------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['temp_cube_scaled'] = (temp ** 3) / 1e5
        X['time_cube_scaled'] = (time ** 3) / 1e2
        X['log_time_sq'] = log_time ** 2

        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp'] = self._safe_divide(time, temp)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # ---------------------------
        # 2. 动力学等效特征
        # ---------------------------
        arrhenius_factor = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        thermal_dose = time * arrhenius_factor
        thermal_dose_scaled = thermal_dose * 1e18

        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = thermal_dose
        X['thermal_dose_scaled'] = thermal_dose_scaled
        X['log_thermal_dose'] = np.log1p(np.clip(thermal_dose_scaled, 0, None))
        X['temp_log_thermal_dose'] = temp * X['log_thermal_dose'].values

        # Larson-Miller Parameter
        lmp = temp_k * (p['lmp_constant_C'] + np.log10(np.clip(time, 1e-6, None)))
        X['larson_miller_param'] = lmp
        X['lmp_scaled'] = lmp / 1000.0

        # Zener-Hollomon风格启发（简化）
        X['zener_like'] = np.log1p(np.clip(np.exp(self.Q / np.clip(self.R * temp_k, 1e-12, None)) / np.clip(time, 1e-6, None), 0, 1e300))
        X['jmak_like_drive'] = arrhenius_factor * np.power(np.clip(time, 0, None) + 1e-9, 0.5)
        X['log_jmak_like_drive'] = np.log1p(X['jmak_like_drive'].values * 1e18)

        # ---------------------------
        # 3. 机制边界布尔/分段特征（关键）
        # ---------------------------
        T_low = p['critical_temp_low']
        T_shift = p['critical_temp_regime_shift']
        T_high = p['critical_temp_high_sensitive']
        T_severe = p['critical_temp_severe']

        t_short = p['critical_time_short']
        t_trans = p['critical_time_transition']
        t_sensitive = p['critical_time_sensitive']
        t_long = p['critical_time_long']
        t_vlong = p['critical_time_very_long']

        X['is_temp_low_zone'] = (temp < T_low).astype(float)
        X['is_temp_mid_zone'] = ((temp >= T_low) & (temp < T_shift)).astype(float)
        X['is_temp_shift_zone'] = ((temp >= T_shift) & (temp < T_high)).astype(float)
        X['is_temp_high_zone'] = ((temp >= T_high) & (temp < T_severe)).astype(float)
        X['is_temp_severe_zone'] = (temp >= T_severe).astype(float)

        X['is_time_short_zone'] = (time < t_short).astype(float)
        X['is_time_transition_zone'] = ((time >= t_short) & (time < t_trans)).astype(float)
        X['is_time_sensitive_zone'] = ((time >= t_trans) & (time < t_long)).astype(float)
        X['is_time_long_zone'] = ((time >= t_long) & (time < t_vlong)).astype(float)
        X['is_time_very_long_zone'] = (time >= t_vlong).astype(float)

        # 显式机制组合
        X['is_lowT_shortt'] = ((temp < T_low) & (time < t_short)).astype(float)
        X['is_lowT_longt'] = ((temp < T_low) & (time >= t_long)).astype(float)
        X['is_shiftT_12h_window'] = ((temp >= T_shift) & (temp < T_high) & (time >= 8) & (time <= 16)).astype(float)
        X['is_highT_12h_window'] = ((temp >= T_high) & (time >= 8) & (time <= 16)).astype(float)
        X['is_highT_longt'] = ((temp >= T_high) & (time >= t_long)).astype(float)
        X['is_very_long_exposure'] = ((temp >= T_low) & (time >= t_vlong)).astype(float)

        # ---------------------------
        # 4. 相对边界距离
        # ---------------------------
        X['temp_relative_to_low'] = temp - T_low
        X['temp_relative_to_shift'] = temp - T_shift
        X['temp_relative_to_high'] = temp - T_high
        X['temp_relative_to_severe'] = temp - T_severe

        X['time_relative_to_short'] = time - t_short
        X['time_relative_to_transition'] = time - t_trans
        X['time_relative_to_sensitive'] = time - t_sensitive
        X['time_relative_to_long'] = time - t_long
        X['time_relative_to_very_long'] = time - t_vlong

        X['relu_temp_above_low'] = np.maximum(temp - T_low, 0)
        X['relu_temp_above_shift'] = np.maximum(temp - T_shift, 0)
        X['relu_temp_above_high'] = np.maximum(temp - T_high, 0)

        X['relu_time_above_transition'] = np.maximum(time - t_trans, 0)
        X['relu_time_above_sensitive'] = np.maximum(time - t_sensitive, 0)
        X['relu_time_above_long'] = np.maximum(time - t_long, 0)
        X['relu_time_above_very_long'] = np.maximum(time - t_vlong, 0)

        # ---------------------------
        # 5. 平滑门控
        # ---------------------------
        X['temp_activation_low'] = self._sigmoid((temp - T_low) / 5.0)
        X['temp_activation_shift'] = self._sigmoid((temp - T_shift) / 3.5)
        X['temp_activation_high'] = self._sigmoid((temp - T_high) / 3.0)
        X['temp_activation_severe'] = self._sigmoid((temp - T_severe) / 2.5)

        X['time_activation_short_end'] = self._sigmoid((time - t_short) / 0.8)
        X['time_activation_transition'] = self._sigmoid((time - t_trans) / 1.8)
        X['time_activation_sensitive'] = self._sigmoid((time - t_sensitive) / 1.5)
        X['time_activation_long'] = self._sigmoid((time - t_long) / 2.0)
        X['time_activation_very_long'] = self._sigmoid((time - t_vlong) / 2.0)

        # ---------------------------
        # 6. 高误差热点窗口特征（针对性设计）
        # ---------------------------
        X['gauss_temp_440'] = self._gaussian(temp, 440.0, 5.0)
        X['gauss_temp_460'] = self._gaussian(temp, 460.0, p['hotspot_temp_width'])
        X['gauss_temp_470'] = self._gaussian(temp, 470.0, p['hotspot_temp_width'])

        X['gauss_time_1'] = self._gaussian(time, 1.0, 1.0)
        X['gauss_time_12'] = self._gaussian(time, 12.0, p['hotspot_time_width'])
        X['gauss_time_24'] = self._gaussian(time, 24.0, 4.0)

        X['hotspot_440_1'] = X['gauss_temp_440'].values * X['gauss_time_1'].values
        X['hotspot_440_24'] = X['gauss_temp_440'].values * X['gauss_time_24'].values
        X['hotspot_460_12'] = X['gauss_temp_460'].values * X['gauss_time_12'].values
        X['hotspot_470_12'] = X['gauss_temp_470'].values * X['gauss_time_12'].values

        # ---------------------------
        # 7. 强化-软化竞争特征
        # ---------------------------
        precipitation_drive = (
            0.30 * X['temp_activation_low'].values +
            0.20 * (1.0 - np.exp(-np.clip(time, 0, None) / 5.0)) +
            0.20 * X['time_activation_transition'].values * (1.0 - X['temp_activation_high'].values) +
            0.15 * X['hotspot_440_24'].values +
            0.15 * self._sigmoid(X['log_thermal_dose'].values - 1.0)
        )

        softening_drive = (
            0.30 * X['temp_activation_shift'].values * X['time_activation_sensitive'].values +
            0.25 * X['temp_activation_high'].values * X['time_activation_transition'].values +
            0.20 * X['temp_activation_high'].values * X['time_activation_long'].values +
            0.15 * X['hotspot_460_12'].values +
            0.10 * X['hotspot_470_12'].values
        )

        ductility_drive = (
            0.25 * X['temp_activation_low'].values +
            0.25 * X['time_activation_transition'].values +
            0.20 * X['time_activation_long'].values +
            0.15 * X['hotspot_470_12'].values +
            0.15 * X['hotspot_440_24'].values
        )

        X['precipitation_drive'] = precipitation_drive
        X['softening_drive'] = softening_drive
        X['net_strengthening_index'] = precipitation_drive - softening_drive
        X['ductility_drive'] = ductility_drive
        X['strength_ductility_tradeoff'] = (precipitation_drive - softening_drive) - 0.6 * ductility_drive

        # ---------------------------
        # 8. 分段交互
        # ---------------------------
        X['temp_in_shift_zone'] = temp * X['is_temp_shift_zone'].values
        X['temp_in_high_zone'] = temp * X['is_temp_high_zone'].values
        X['time_in_sensitive_zone'] = time * X['is_time_sensitive_zone'].values
        X['time_in_very_long_zone'] = time * X['is_time_very_long_zone'].values

        X['temp_log_time_shift_zone'] = temp * log_time * X['is_temp_shift_zone'].values
        X['temp_log_time_high_zone'] = temp * log_time * X['is_temp_high_zone'].values
        X['temp_log_time_12h_window'] = temp * log_time * X['is_shiftT_12h_window'].values
        X['temp_log_time_24h_window'] = temp * log_time * X['is_very_long_exposure'].values

        X['temp_x_time_activation_sensitive'] = temp * X['time_activation_sensitive'].values
        X['time_x_temp_activation_high'] = time * X['temp_activation_high'].values

        # ---------------------------
        # 9. physics baseline 作为特征
        # ---------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_strength_sum'] = baseline_tensile + baseline_yield
        X['baseline_strength_diff'] = baseline_tensile - baseline_yield

        X['baseline_tensile_x_softening'] = baseline_tensile * softening_drive
        X['baseline_yield_x_softening'] = baseline_yield * softening_drive
        X['baseline_strain_x_ductility'] = baseline_strain * ductility_drive

        # ---------------------------
        # 10. 相对基线/相对边界的残差型提示特征
        # ---------------------------
        X['temp_minus_hotspot_460'] = temp - 460.0
        X['temp_minus_hotspot_470'] = temp - 470.0
        X['time_minus_hotspot_12'] = time - 12.0
        X['time_minus_hotspot_24'] = time - 24.0

        X['relative_temp_to_shift'] = self._safe_divide(temp - T_shift, T_shift)
        X['relative_temp_to_high'] = self._safe_divide(temp - T_high, T_high)
        X['relative_time_to_12h'] = self._safe_divide(time - t_sensitive, t_sensitive)
        X['relative_time_to_24h'] = self._safe_divide(time - t_vlong, t_vlong)

        X['process_minus_baseline_strength_scale'] = (
            X['net_strengthening_index'].values * X['baseline_strength_sum'].values
        )
        X['process_minus_baseline_ductility_scale'] = (
            X['ductility_drive'].values * X['baseline_strain'].values
        )

        # ---------------------------
        # 11. 铸态相对提升潜力
        # ---------------------------
        base_tensile = p['base_tensile']
        base_yield = p['base_yield']
        base_strain = p['base_strain']

        X['base_tensile'] = base_tensile
        X['base_yield'] = base_yield
        X['base_strain'] = base_strain

        X['baseline_tensile_gain_over_cast'] = baseline_tensile - base_tensile
        X['baseline_yield_gain_over_cast'] = baseline_yield - base_yield
        X['baseline_strain_gain_over_cast'] = baseline_strain - base_strain

        X['tensile_gain_ratio_over_cast'] = self._safe_divide(baseline_tensile - base_tensile, base_tensile)
        X['yield_gain_ratio_over_cast'] = self._safe_divide(baseline_yield - base_yield, base_yield)
        X['strain_gain_ratio_over_cast'] = self._safe_divide(baseline_strain - base_strain, base_strain)

        # ---------------------------
        # 12. 稳健裁剪特征
        # ---------------------------
        clipped_temp = np.clip(temp, 430, 480)
        clipped_time = np.clip(time, 0, 30)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['temp_saturation'] = self._sigmoid((clipped_temp - T_low) / 6.0)
        X['combined_saturation'] = X['time_saturation'].values * X['temp_saturation'].values

        X['high_temp_sensitive_penalty'] = np.maximum(clipped_temp - T_shift, 0) * np.maximum(clipped_time - t_sensitive, 0)
        X['very_high_temp_long_time_penalty'] = np.maximum(clipped_temp - T_high, 0) * np.maximum(clipped_time - t_long, 0)

        # ---------------------------
        # 13. 最终清理
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