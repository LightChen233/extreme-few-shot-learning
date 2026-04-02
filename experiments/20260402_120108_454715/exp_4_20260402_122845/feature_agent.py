import pandas as pd
import numpy as np

# === 领域知识参数（基于材料科学文献/工程经验的启发式常数） ===
DOMAIN_PARAMS = {
    # 基础物理常数
    'gas_constant_J_molK': 8.314,

    # 7499/7xxx铝合金热处理相关经验参数
    # 数值为工程启发范围，不代表严格单一材料常数
    'diffusion_activation_energy_J_mol': 65000.0,
    'vacancy_mobility_activation_energy_J_mol': 72000.0,
    'coarsening_activation_energy_J_mol': 85000.0,

    # 典型时效窗口
    'peak_aging_temp_C': 130.0,
    'peak_aging_time_h': 8.0,
    'under_aging_temp_C': 110.0,
    'over_aging_temp_C': 150.0,

    # 组织演化阈值
    'recovery_onset_temp_C': 140.0,
    'grain_coarsening_threshold_C': 150.0,
    'precipitate_dissolution_temp_C': 180.0,

    # 时间阈值
    'short_time_h': 4.0,
    'medium_time_h': 12.0,
    'long_time_h': 24.0,

    # 数据合理边界
    'min_valid_temp_C': 60.0,
    'max_valid_temp_C': 220.0,
    'min_valid_time_h': 0.0,
    'max_valid_time_h': 72.0,

    # 峰值窗口宽度
    'peak_temp_width_C': 18.0,
    'peak_log_time_width': 0.75,

    # 原始凝固态样品基准性能
    'base_strain_pct': 6.94,
    'base_tensile_MPa': 145.83,
    'base_yield_MPa': 96.60,

    # 经验缩放参数
    'softening_temp_scale_C': 25.0,
    'time_saturation_h': 6.0,
    'temp_activation_width_C': 10.0,
    'kinetics_center_temp_C': 130.0,
    'kinetics_temp_scale_C': 22.0,
    'arrhenius_reference_temp_C': 130.0,
    'arrhenius_sensitivity_scale': 1e7,
}


class FeatureAgent:
    """
    基于材料热处理机理的动态特征工程
    目标：根据 temp, time 预测 strain / tensile_strength / yield_strength
    """

    def __init__(self):
        self.feature_names = []
        self.params = DOMAIN_PARAMS

        # 原始凝固态基准
        self.base_strain = self.params['base_strain_pct']
        self.base_tensile = self.params['base_tensile_MPa']
        self.base_yield = self.params['base_yield_MPa']

        # 温度阈值
        self.low_temp_threshold = self.params['under_aging_temp_C']
        self.peak_temp = self.params['peak_aging_temp_C']
        self.high_temp_threshold = self.params['over_aging_temp_C']
        self.recovery_onset_temp = self.params['recovery_onset_temp_C']
        self.coarsening_temp = self.params['grain_coarsening_threshold_C']
        self.dissolution_temp = self.params['precipitate_dissolution_temp_C']

        # 时间阈值
        self.short_time_threshold = self.params['short_time_h']
        self.medium_time_threshold = self.params['medium_time_h']
        self.long_time_threshold = self.params['long_time_h']
        self.peak_time = self.params['peak_aging_time_h']

        # 热激活参数
        self.R = self.params['gas_constant_J_molK']
        self.Q_diff = self.params['diffusion_activation_energy_J_mol']
        self.Q_vac = self.params['vacancy_mobility_activation_energy_J_mol']
        self.Q_coarse = self.params['coarsening_activation_energy_J_mol']

        # 窗口参数
        self.peak_temp_width = self.params['peak_temp_width_C']
        self.peak_log_time_width = self.params['peak_log_time_width']

        # 数据边界
        self.min_temp = self.params['min_valid_temp_C']
        self.max_temp = self.params['max_valid_temp_C']
        self.min_time = self.params['min_valid_time_h']
        self.max_time = self.params['max_valid_time_h']

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _gaussian_window(self, x, center, width):
        width = max(width, 1e-6)
        return np.exp(-((x - center) / width) ** 2)

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        # 原始输入
        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        # 合理裁剪，防止小样本下极端值破坏
        clipped_temp = np.clip(temp, self.min_temp, self.max_temp)
        clipped_time = np.clip(time, self.min_time, self.max_time)

        # 基础变换
        temp_k = clipped_temp + 273.15
        log_time = self._safe_log(clipped_time)
        sqrt_time = np.sqrt(np.clip(clipped_time, 0, None))
        inv_temp_k = self._safe_divide(1.0, temp_k)
        peak_log_time = np.log1p(self.peak_time)

        # ==================================================
        # 1. 基础特征
        # ==================================================
        X['temp'] = clipped_temp
        X['time'] = clipped_time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # ==================================================
        # 2. 基础非线性与交互
        # ==================================================
        X['temp_sq'] = clipped_temp ** 2
        X['time_sq'] = clipped_time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_cube_scaled'] = (clipped_temp ** 3) / 1e4
        X['time_cube_scaled'] = (clipped_time ** 3) / 1e3

        X['temp_time'] = clipped_temp * clipped_time
        X['temp_log_time'] = clipped_temp * log_time
        X['temp_sqrt_time'] = clipped_temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(clipped_time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)
        X['temp_over_log_time1'] = self._safe_divide(clipped_temp, log_time + 1.0)
        X['temp_over_time1'] = self._safe_divide(clipped_temp, clipped_time + 1.0)

        # ==================================================
        # 3. Arrhenius / 热激活动力学特征
        # ==================================================
        arr_diff = np.exp(-self.Q_diff / np.clip(self.R * temp_k, 1e-12, None))
        arr_vac = np.exp(-self.Q_vac / np.clip(self.R * temp_k, 1e-12, None))
        arr_coarse = np.exp(-self.Q_coarse / np.clip(self.R * temp_k, 1e-12, None))

        X['arrhenius_diffusion'] = arr_diff
        X['arrhenius_vacancy'] = arr_vac
        X['arrhenius_coarsening'] = arr_coarse

        thermal_dose = clipped_time * arr_diff
        vacancy_dose = clipped_time * arr_vac
        coarsening_dose = clipped_time * arr_coarse

        X['thermal_dose'] = thermal_dose
        X['vacancy_dose'] = vacancy_dose
        X['coarsening_dose'] = coarsening_dose
        X['log_thermal_dose'] = self._safe_log(thermal_dose)
        X['log_vacancy_dose'] = self._safe_log(vacancy_dose)
        X['log_coarsening_dose'] = self._safe_log(coarsening_dose)

        # 参考温度归一化的热激活比值，帮助小样本学习相对加速效应
        ref_temp_k = self.params['arrhenius_reference_temp_C'] + 273.15
        ref_arr = np.exp(-self.Q_diff / (self.R * ref_temp_k))
        X['relative_diffusion_rate'] = self._safe_divide(arr_diff, ref_arr)

        # ==================================================
        # 4. 时间-温度等效老化特征
        # ==================================================
        # 类似 Larson-Miller / Hollomon-Jaffe 风格经验特征
        X['temp_log_time_kelvin'] = temp_k * (20.0 + log_time)
        X['equivalent_aging_index'] = clipped_temp * log_time
        X['equivalent_aging_index_k'] = temp_k * log_time
        X['kinetics_compensation'] = self._safe_divide(log_time, inv_temp_k * 1000.0 + 1e-6)
        X['diffusion_weighted_time'] = clipped_time * np.exp((clipped_temp - self.peak_temp) / 30.0)

        # ==================================================
        # 5. 温度分区逻辑
        # ==================================================
        low_temp_mask = (clipped_temp < self.low_temp_threshold).astype(float)
        peak_temp_mask = ((clipped_temp >= self.low_temp_threshold) & (clipped_temp < self.high_temp_threshold)).astype(float)
        high_temp_mask = (clipped_temp >= self.high_temp_threshold).astype(float)
        dissolution_mask = (clipped_temp >= self.dissolution_temp).astype(float)

        X['is_low_temp'] = low_temp_mask
        X['is_peak_temp_zone'] = peak_temp_mask
        X['is_high_temp'] = high_temp_mask
        X['is_dissolution_risk_temp'] = dissolution_mask

        X['dist_to_underaging_temp'] = clipped_temp - self.low_temp_threshold
        X['dist_to_peak_temp'] = clipped_temp - self.peak_temp
        X['dist_to_overaging_temp'] = clipped_temp - self.high_temp_threshold
        X['relu_above_underaging_temp'] = np.maximum(clipped_temp - self.low_temp_threshold, 0.0)
        X['relu_above_peak_temp'] = np.maximum(clipped_temp - self.peak_temp, 0.0)
        X['relu_above_overaging_temp'] = np.maximum(clipped_temp - self.high_temp_threshold, 0.0)
        X['relu_above_coarsening_temp'] = np.maximum(clipped_temp - self.coarsening_temp, 0.0)
        X['relu_above_dissolution_temp'] = np.maximum(clipped_temp - self.dissolution_temp, 0.0)
        X['relu_below_peak_temp'] = np.maximum(self.peak_temp - clipped_temp, 0.0)

        # ==================================================
        # 6. 时间分区逻辑
        # ==================================================
        short_time_mask = (clipped_time < self.short_time_threshold).astype(float)
        medium_time_mask = ((clipped_time >= self.short_time_threshold) & (clipped_time < self.medium_time_threshold)).astype(float)
        long_time_mask = (clipped_time >= self.medium_time_threshold).astype(float)
        extra_long_time_mask = (clipped_time >= self.long_time_threshold).astype(float)

        X['is_short_time'] = short_time_mask
        X['is_medium_time'] = medium_time_mask
        X['is_long_time'] = long_time_mask
        X['is_extra_long_time'] = extra_long_time_mask

        X['dist_to_peak_time_log'] = log_time - peak_log_time
        X['relu_above_short_time'] = np.maximum(clipped_time - self.short_time_threshold, 0.0)
        X['relu_above_medium_time'] = np.maximum(clipped_time - self.medium_time_threshold, 0.0)
        X['relu_above_long_time'] = np.maximum(clipped_time - self.long_time_threshold, 0.0)
        X['relu_below_short_time'] = np.maximum(self.short_time_threshold - clipped_time, 0.0)

        # ==================================================
        # 7. 峰值时效邻近度
        # ==================================================
        temp_peak_proximity = -((clipped_temp - self.peak_temp) / self.peak_temp_width) ** 2
        time_peak_proximity = -((log_time - peak_log_time) / self.peak_log_time_width) ** 2
        joint_peak_proximity = temp_peak_proximity + time_peak_proximity
        peak_window = np.exp(joint_peak_proximity)

        X['temp_peak_proximity'] = temp_peak_proximity
        X['time_peak_proximity'] = time_peak_proximity
        X['joint_peak_proximity'] = joint_peak_proximity
        X['peak_window'] = peak_window

        # ==================================================
        # 8. 动态组织演化推理特征
        # ==================================================
        # 8.1 析出强化驱动力：低温慢、中温强、高温前强后衰
        precipitation_drive = np.zeros_like(clipped_temp, dtype=float)

        low_idx = clipped_temp < self.low_temp_threshold
        precipitation_drive[low_idx] = (
            0.45 * log_time[low_idx]
            + 0.004 * (clipped_temp[low_idx] - self.min_temp) * log_time[low_idx]
            + 0.6 * thermal_dose[low_idx] * self.params['arrhenius_sensitivity_scale']
        )

        peak_idx = (clipped_temp >= self.low_temp_threshold) & (clipped_temp < self.high_temp_threshold)
        precipitation_drive[peak_idx] = (
            0.95 * log_time[peak_idx]
            + 0.010 * (clipped_temp[peak_idx] - self.low_temp_threshold) * log_time[peak_idx]
            + 0.9 * peak_window[peak_idx]
            - 0.035 * np.maximum(clipped_time[peak_idx] - self.medium_time_threshold, 0.0)
        )

        high_idx = clipped_temp >= self.high_temp_threshold
        precipitation_drive[high_idx] = (
            0.80 * np.minimum(log_time[high_idx], np.log1p(self.medium_time_threshold))
            + 0.20
            - 0.18 * np.maximum(log_time[high_idx] - np.log1p(self.short_time_threshold), 0.0)
        )

        X['precipitation_drive'] = precipitation_drive

        # 8.2 恢复、粗化、溶解驱动力
        recovery_drive = (
            np.maximum(clipped_temp - self.recovery_onset_temp, 0.0) / 20.0
        ) * log_time

        coarsening_drive = (
            np.maximum(clipped_temp - self.coarsening_temp, 0.0) / self.params['softening_temp_scale_C']
        ) * np.maximum(log_time - np.log1p(self.short_time_threshold), 0.0)

        dissolution_drive = (
            np.maximum(clipped_temp - self.dissolution_temp, 0.0) / 15.0
        ) * self._sigmoid(clipped_time - self.short_time_threshold)

        softening_drive = (
            0.7 * recovery_drive
            + 1.0 * coarsening_drive
            + 1.2 * dissolution_drive
            + 0.03 * np.maximum(clipped_time - self.medium_time_threshold, 0.0) * high_temp_mask
        )

        X['recovery_drive'] = recovery_drive
        X['coarsening_drive'] = coarsening_drive
        X['dissolution_drive'] = dissolution_drive
        X['softening_drive'] = softening_drive

        net_strengthening = precipitation_drive - softening_drive
        X['net_strengthening_index'] = net_strengthening

        # ==================================================
        # 9. 工艺状态推理：欠时效 / 峰值时效 / 过时效
        # ==================================================
        underaging_tendency = (
            low_temp_mask * (1.0 + 0.5 * short_time_mask)
            + self._sigmoid((self.peak_temp - clipped_temp) / 10.0)
            * self._sigmoid((peak_log_time - log_time) / 0.5)
        )

        peak_strength_tendency = peak_window * self._sigmoid(net_strengthening)

        overaging_tendency = self._sigmoid(
            0.8 * (clipped_temp - self.high_temp_threshold) / 10.0
            + 1.2 * (log_time - np.log1p(self.medium_time_threshold))
            + coarsening_drive
        )

        X['underaging_tendency'] = underaging_tendency
        X['peak_strength_tendency'] = peak_strength_tendency
        X['overaging_tendency'] = overaging_tendency

        # ==================================================
        # 10. 机制分区交互特征
        # ==================================================
        X['temp_in_low_zone'] = clipped_temp * low_temp_mask
        X['temp_in_peak_zone'] = clipped_temp * peak_temp_mask
        X['temp_in_high_zone'] = clipped_temp * high_temp_mask

        X['log_time_in_low_zone'] = log_time * low_temp_mask
        X['log_time_in_peak_zone'] = log_time * peak_temp_mask
        X['log_time_in_high_zone'] = log_time * high_temp_mask

        X['temp_log_time_low_zone'] = clipped_temp * log_time * low_temp_mask
        X['temp_log_time_peak_zone'] = clipped_temp * log_time * peak_temp_mask
        X['temp_log_time_high_zone'] = clipped_temp * log_time * high_temp_mask

        X['high_temp_short_time_equiv'] = high_temp_mask * self._safe_divide(clipped_temp, log_time + 1.0)
        X['low_temp_long_time_equiv'] = low_temp_mask * clipped_temp * log_time

        # ==================================================
        # 11. 饱和与激活特征
        # ==================================================
        time_saturation = 1.0 - np.exp(-clipped_time / self.params['time_saturation_h'])
        temp_activation = self._sigmoid((clipped_temp - self.low_temp_threshold) / self.params['temp_activation_width_C'])
        peak_temp_activation = self._gaussian_window(clipped_temp, self.peak_temp, 18.0)

        X['time_saturation'] = time_saturation
        X['temp_activation'] = temp_activation
        X['peak_temp_activation'] = peak_temp_activation
        X['combined_activation'] = time_saturation * temp_activation
        X['peak_zone_activation'] = time_saturation * peak_temp_activation

        # ==================================================
        # 12. 强塑竞争特征
        # ==================================================
        ductility_recovery = (
            0.55 * softening_drive
            + 0.20 * high_temp_mask
            + 0.20 * long_time_mask
            - 0.25 * precipitation_drive
        )

        strength_ductility_tradeoff = net_strengthening - ductility_recovery

        X['ductility_recovery_index'] = ductility_recovery
        X['strength_ductility_tradeoff'] = strength_ductility_tradeoff

        # ==================================================
        # 13. 风险与边界约束特征
        # ==================================================
        high_temp_long_time_penalty = (
            np.maximum(clipped_temp - self.high_temp_threshold, 0.0)
            * np.maximum(clipped_time - self.short_time_threshold, 0.0)
        )

        dissolution_penalty = (
            np.maximum(clipped_temp - self.dissolution_temp, 0.0)
            * np.maximum(clipped_time - self.short_time_threshold, 0.0)
        )

        extrapolation_temp_risk = (
            np.maximum(temp - self.max_temp, 0.0) + np.maximum(self.min_temp - temp, 0.0)
        )
        extrapolation_time_risk = (
            np.maximum(time - self.max_time, 0.0) + np.maximum(self.min_time - time, 0.0)
        )

        X['high_temp_long_time_penalty'] = high_temp_long_time_penalty
        X['dissolution_penalty'] = dissolution_penalty
        X['extrapolation_temp_risk'] = extrapolation_temp_risk
        X['extrapolation_time_risk'] = extrapolation_time_risk
        X['instability_index'] = (
            0.6 * high_temp_long_time_penalty / 100.0
            + 0.8 * dissolution_penalty / 100.0
            + 0.2 * extrapolation_temp_risk
            + 0.2 * extrapolation_time_risk
        )

        # ==================================================
        # 14. 基于基准样品的相对潜力特征
        # ==================================================
        base_strength_sum = self.base_tensile + self.base_yield
        base_strength_ratio = self._safe_divide(self.base_yield, self.base_tensile)

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['base_strain'] = self.base_strain
        X['base_tensile'] = self.base_tensile
        X['base_yield'] = self.base_yield

        process_to_base_potential = (
            0.8 * peak_strength_tendency
            + 0.5 * net_strengthening
            + 0.3 * X['combined_activation'].values
            - 0.3 * X['instability_index'].values
        )

        X['process_to_base_potential'] = process_to_base_potential
        X['base_relative_strengthening'] = self._safe_divide(process_to_base_potential, base_strength_sum)
        X['base_relative_ductility_shift'] = self._safe_divide(ductility_recovery, self.base_strain + 1e-6)

        # ==================================================
        # 15. 多输出共享状态特征
        # ==================================================
        strength_state_index = (
            0.9 * net_strengthening
            + 0.6 * peak_window
            - 0.5 * overaging_tendency
        )

        ductility_state_index = (
            0.8 * ductility_recovery
            + 0.4 * overaging_tendency
            - 0.3 * peak_strength_tendency
        )

        yield_tensile_coupling_index = (
            strength_state_index * (1.0 + 0.2 * peak_temp_mask - 0.1 * dissolution_mask)
        )

        X['strength_state_index'] = strength_state_index
        X['ductility_state_index'] = ductility_state_index
        X['yield_tensile_coupling_index'] = yield_tensile_coupling_index

        # ==================================================
        # 16. 更稳健的小样本压缩特征
        # ==================================================
        # 避免单纯高次项主导，增加压缩型特征帮助拟合
        X['compressed_temp'] = np.tanh((clipped_temp - self.peak_temp) / 30.0)
        X['compressed_log_time'] = np.tanh((log_time - peak_log_time) / 1.0)
        X['compressed_joint_state'] = X['compressed_temp'].values * X['compressed_log_time'].values

        # ==================================================
        # 17. 最终稳健处理
        # ==================================================
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