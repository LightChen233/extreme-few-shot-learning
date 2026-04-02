import pandas as pd
import numpy as np

# === 领域知识参数（基于材料科学文献/世界知识的启发式设定） ===
# 说明：
# 1) 这些参数用于构造“机理感知特征”，不是严格热力学反演；
# 2) 数值取典型 Al-Zn-Mg-Cu 系高强铝合金时效过程的经验量级，适合小样本稳健建模；
# 3) 7499 铝合金的性能演化通常受析出强化、扩散控制、粗化/过时效与局部溶解共同影响。
DOMAIN_PARAMS = {
    # 基本物理常数
    'gas_constant_J_molK': 8.314,

    # 扩散/析出相关经验激活能（铝合金中常见量级）
    'diffusion_activation_energy_J_mol': 65000.0,

    # 温度特征点（°C）
    'min_valid_temp_C': 60.0,
    'low_temp_threshold_C': 110.0,            # 扩散不足，欠时效更常见
    'peak_aging_temp_C': 130.0,               # 典型峰值时效附近
    'mid_temp_threshold_C': 150.0,            # 强化-过时效转折敏感区
    'grain_coarsening_threshold_C': 155.0,    # 粗化/回复风险增强
    'precipitate_dissolution_temp_C': 180.0,  # 析出相稳定性下降
    'max_valid_temp_C': 200.0,

    # 时间特征点（h）
    'min_valid_time_h': 0.0,
    'short_time_threshold_h': 4.0,            # 时效初期
    'peak_aging_time_h': 8.0,                 # 典型峰值时效时间
    'long_time_threshold_h': 12.0,            # 过时效风险上升
    'saturation_time_h': 6.0,                 # 饱和时间尺度
    'max_valid_time_h': 24.0,

    # 窗口宽度
    'peak_temp_window_C': 18.0,
    'peak_log_time_window': 0.75,

    # 门控尺度
    'activation_temp_scale_C': 10.0,
    'overaging_temp_scale_C': 22.0,

    # 原始凝固态样品性能基准
    'base_strain_pct': 6.94,
    'base_tensile_MPa': 145.83,
    'base_yield_MPa': 96.60,

    # 分区强化系数
    'low_temp_precip_coeff': 0.55,
    'mid_temp_precip_coeff': 1.20,
    'high_temp_precip_coeff': 0.95,

    # 软化/过时效系数
    'softening_temp_coeff': 0.030,
    'softening_time_coeff': 0.075,
    'softening_long_time_coeff': 0.015,
    'dissolution_penalty_coeff': 0.070,

    # 机理组合权重
    'activation_weight': 0.45,
    'strengthening_weight': 0.75,
    'peak_window_weight': 0.35,
    'overaging_penalty_weight': 0.25,

    'ductility_softening_weight': 0.55,
    'ductility_long_time_weight': 0.20,
    'ductility_high_temp_weight': 0.15,
    'ductility_very_high_penalty_weight': 0.20,
}


class FeatureAgent:
    """基于材料热处理机理的动态特征工程"""

    def __init__(self):
        self.feature_names = []

        # 从 DOMAIN_PARAMS 中读取，避免硬编码
        self.R = DOMAIN_PARAMS['gas_constant_J_molK']
        self.Q = DOMAIN_PARAMS['diffusion_activation_energy_J_mol']

        self.min_valid_temp = DOMAIN_PARAMS['min_valid_temp_C']
        self.low_temp_threshold = DOMAIN_PARAMS['low_temp_threshold_C']
        self.peak_temp = DOMAIN_PARAMS['peak_aging_temp_C']
        self.mid_temp_threshold = DOMAIN_PARAMS['mid_temp_threshold_C']
        self.coarsening_temp = DOMAIN_PARAMS['grain_coarsening_threshold_C']
        self.dissolution_temp = DOMAIN_PARAMS['precipitate_dissolution_temp_C']
        self.max_valid_temp = DOMAIN_PARAMS['max_valid_temp_C']

        self.min_valid_time = DOMAIN_PARAMS['min_valid_time_h']
        self.short_time_threshold = DOMAIN_PARAMS['short_time_threshold_h']
        self.peak_time = DOMAIN_PARAMS['peak_aging_time_h']
        self.long_time_threshold = DOMAIN_PARAMS['long_time_threshold_h']
        self.saturation_time = DOMAIN_PARAMS['saturation_time_h']
        self.max_valid_time = DOMAIN_PARAMS['max_valid_time_h']

        self.peak_temp_window = DOMAIN_PARAMS['peak_temp_window_C']
        self.peak_log_time_window = DOMAIN_PARAMS['peak_log_time_window']
        self.activation_temp_scale = DOMAIN_PARAMS['activation_temp_scale_C']
        self.overaging_temp_scale = DOMAIN_PARAMS['overaging_temp_scale_C']

        self.base_strain = DOMAIN_PARAMS['base_strain_pct']
        self.base_tensile = DOMAIN_PARAMS['base_tensile_MPa']
        self.base_yield = DOMAIN_PARAMS['base_yield_MPa']

        self.low_temp_precip_coeff = DOMAIN_PARAMS['low_temp_precip_coeff']
        self.mid_temp_precip_coeff = DOMAIN_PARAMS['mid_temp_precip_coeff']
        self.high_temp_precip_coeff = DOMAIN_PARAMS['high_temp_precip_coeff']

        self.softening_temp_coeff = DOMAIN_PARAMS['softening_temp_coeff']
        self.softening_time_coeff = DOMAIN_PARAMS['softening_time_coeff']
        self.softening_long_time_coeff = DOMAIN_PARAMS['softening_long_time_coeff']
        self.dissolution_penalty_coeff = DOMAIN_PARAMS['dissolution_penalty_coeff']

        self.activation_weight = DOMAIN_PARAMS['activation_weight']
        self.strengthening_weight = DOMAIN_PARAMS['strengthening_weight']
        self.peak_window_weight = DOMAIN_PARAMS['peak_window_weight']
        self.overaging_penalty_weight = DOMAIN_PARAMS['overaging_penalty_weight']

        self.ductility_softening_weight = DOMAIN_PARAMS['ductility_softening_weight']
        self.ductility_long_time_weight = DOMAIN_PARAMS['ductility_long_time_weight']
        self.ductility_high_temp_weight = DOMAIN_PARAMS['ductility_high_temp_weight']
        self.ductility_very_high_penalty_weight = DOMAIN_PARAMS['ductility_very_high_penalty_weight']

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        # 物理边界裁剪：避免极端输入导致的指数失稳，同时保留原始输入特征
        clipped_temp = np.clip(temp, self.min_valid_temp, self.max_valid_temp)
        clipped_time = np.clip(time, self.min_valid_time, self.max_valid_time)

        temp_k = temp + 273.15
        clipped_temp_k = clipped_temp + 273.15

        log_time = self._safe_log(time)
        clipped_log_time = self._safe_log(clipped_time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # =========================================================
        # 1. 基础工艺特征
        # =========================================================
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = self._safe_divide(1.0, temp_k)

        # =========================================================
        # 2. 基础非线性与交互项
        # 小样本下保留少量高价值项，避免无效冗余
        # =========================================================
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)
        X['temp_over_time1p'] = self._safe_divide(temp, 1.0 + time)

        # =========================================================
        # 3. Arrhenius / 扩散动力学特征
        # 温度升高会加快扩散，时间和温度存在等效补偿关系
        # =========================================================
        arrhenius_factor = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        clipped_arrhenius_factor = np.exp(-self.Q / np.clip(self.R * clipped_temp_k, 1e-12, None))

        thermal_dose = time * arrhenius_factor
        thermal_dose_clipped = clipped_time * clipped_arrhenius_factor

        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = self._safe_log(thermal_dose)
        X['equivalent_aging_index'] = log_time - self.Q / np.clip(self.R * temp_k, 1e-12, None)
        X['thermal_dose_clipped'] = thermal_dose_clipped
        X['log_thermal_dose_clipped'] = self._safe_log(thermal_dose_clipped)
        X['temp_log_thermal_dose'] = temp * X['log_thermal_dose'].values

        # =========================================================
        # 4. 温度区间状态特征
        # 根据不同温区采用不同机理解释
        # =========================================================
        low_temp_mask = (temp < self.low_temp_threshold).astype(float)
        mid_temp_mask = ((temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)).astype(float)
        high_temp_mask = (temp >= self.mid_temp_threshold).astype(float)
        very_high_temp_mask = (temp >= self.dissolution_temp).astype(float)

        X['is_low_temp'] = low_temp_mask
        X['is_mid_temp'] = mid_temp_mask
        X['is_high_temp'] = high_temp_mask
        X['is_very_high_temp'] = very_high_temp_mask

        X['dist_to_peak_temp'] = temp - self.peak_temp
        X['dist_to_mid_temp'] = temp - self.mid_temp_threshold
        X['dist_to_dissolution_temp'] = temp - self.dissolution_temp
        X['relu_above_low_temp'] = np.maximum(temp - self.low_temp_threshold, 0.0)
        X['relu_above_peak_temp'] = np.maximum(temp - self.peak_temp, 0.0)
        X['relu_above_mid_temp'] = np.maximum(temp - self.mid_temp_threshold, 0.0)
        X['relu_above_dissolution_temp'] = np.maximum(temp - self.dissolution_temp, 0.0)
        X['relu_below_low_temp'] = np.maximum(self.low_temp_threshold - temp, 0.0)

        # =========================================================
        # 5. 时间区间状态特征
        # 初期/峰值附近/长时效阶段的响应斜率不同
        # =========================================================
        short_time_mask = (time < self.short_time_threshold).astype(float)
        mid_time_mask = ((time >= self.short_time_threshold) & (time < self.long_time_threshold)).astype(float)
        long_time_mask = (time >= self.long_time_threshold).astype(float)

        X['is_short_time'] = short_time_mask
        X['is_mid_time'] = mid_time_mask
        X['is_long_time'] = long_time_mask

        X['dist_to_peak_time'] = time - self.peak_time
        X['dist_to_long_time'] = time - self.long_time_threshold
        X['relu_above_short_time'] = np.maximum(time - self.short_time_threshold, 0.0)
        X['relu_above_peak_time'] = np.maximum(time - self.peak_time, 0.0)
        X['relu_above_long_time'] = np.maximum(time - self.long_time_threshold, 0.0)
        X['relu_below_short_time'] = np.maximum(self.short_time_threshold - time, 0.0)

        # =========================================================
        # 6. 工艺激活门控
        # 温度不足则扩散慢；时间过短则析出尚未完成
        # =========================================================
        temp_activation = self._sigmoid((clipped_temp - self.low_temp_threshold) / self.activation_temp_scale)
        time_saturation = 1.0 - np.exp(-clipped_time / self.saturation_time)
        peak_temp_gate = np.exp(-((temp - self.peak_temp) / self.peak_temp_window) ** 2)
        peak_time_gate = np.exp(-((clipped_log_time - np.log1p(self.peak_time)) / self.peak_log_time_window) ** 2)

        X['temp_activation'] = temp_activation
        X['time_saturation'] = time_saturation
        X['peak_temp_gate'] = peak_temp_gate
        X['peak_time_gate'] = peak_time_gate
        X['combined_activation'] = temp_activation * time_saturation
        X['peak_window'] = peak_temp_gate * peak_time_gate

        # =========================================================
        # 7. 动态析出强化驱动力
        # 按温区切换逻辑，而不是简单全局公式
        # =========================================================
        precipitation_drive = np.zeros_like(temp, dtype=float)

        low_idx = temp < self.low_temp_threshold
        precipitation_drive[low_idx] = (
            self.low_temp_precip_coeff * log_time[low_idx]
            + 0.004 * np.maximum(temp[low_idx] - self.min_valid_temp, 0.0) * log_time[low_idx]
        )

        mid_idx = (temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)
        precipitation_drive[mid_idx] = (
            self.mid_temp_precip_coeff * log_time[mid_idx]
            + 0.012 * (temp[mid_idx] - self.low_temp_threshold) * log_time[mid_idx]
            - 0.05 * np.maximum(time[mid_idx] - self.long_time_threshold, 0.0)
        )

        high_idx = temp >= self.mid_temp_threshold
        precipitation_drive[high_idx] = (
            self.high_temp_precip_coeff * np.minimum(log_time[high_idx], np.log1p(self.long_time_threshold))
            + 0.01 * (self.mid_temp_threshold - self.low_temp_threshold)
            - 0.11 * np.maximum(log_time[high_idx] - np.log1p(self.short_time_threshold), 0.0)
        )

        # 极高温下析出相稳定性下降，增加惩罚
        dissolution_penalty = (
            self.dissolution_penalty_coeff
            * np.maximum(temp - self.dissolution_temp, 0.0)
            * (0.5 + log_time)
        )
        precipitation_drive = precipitation_drive - dissolution_penalty

        X['precipitation_drive'] = precipitation_drive

        # =========================================================
        # 8. 软化/过时效驱动力
        # 高温 + 长时间 -> 粗化、回复、溶解，强度下降
        # =========================================================
        overaging_index = log_time * np.exp((temp - self.mid_temp_threshold) / self.overaging_temp_scale)

        softening_drive = (
            np.maximum(temp - self.coarsening_temp, 0.0) * self.softening_temp_coeff * log_time
            + high_temp_mask * np.maximum(time - self.short_time_threshold, 0.0) * self.softening_time_coeff
            + long_time_mask * np.maximum(temp - self.low_temp_threshold, 0.0) * self.softening_long_time_coeff
            + very_high_temp_mask * 0.05 * np.maximum(temp - self.dissolution_temp, 0.0) * (1.0 + log_time)
        )

        net_strengthening_index = precipitation_drive - softening_drive

        X['overaging_index'] = overaging_index
        X['softening_drive'] = softening_drive
        X['net_strengthening_index'] = net_strengthening_index

        # =========================================================
        # 9. 峰值时效邻近度
        # 峰值窗口附近通常获得较优强度
        # =========================================================
        temp_peak_proximity = -((temp - self.peak_temp) / self.peak_temp_window) ** 2
        time_peak_proximity = -((log_time - np.log1p(self.peak_time)) / self.peak_log_time_window) ** 2
        joint_peak_proximity = temp_peak_proximity + time_peak_proximity

        X['temp_peak_proximity'] = temp_peak_proximity
        X['time_peak_proximity'] = time_peak_proximity
        X['joint_peak_proximity'] = joint_peak_proximity
        X['peak_window_strength_drive'] = X['peak_window'].values * precipitation_drive
        X['peak_window_softening_balance'] = X['peak_window'].values * net_strengthening_index

        # =========================================================
        # 10. 强塑性竞争关系
        # 强化通常提高强度但损害塑性；过时效可能恢复部分塑性
        # =========================================================
        ductility_recovery_index = (
            0.50 * softening_drive
            + 0.30 * high_temp_mask
            + 0.20 * long_time_mask
            - 0.30 * precipitation_drive
        )

        X['ductility_recovery_index'] = ductility_recovery_index
        X['strength_ductility_tradeoff'] = net_strengthening_index - ductility_recovery_index

        # =========================================================
        # 11. 分区交互项
        # 显式表达不同工艺区间中的不同斜率
        # =========================================================
        X['temp_in_low_zone'] = temp * low_temp_mask
        X['temp_in_mid_zone'] = temp * mid_temp_mask
        X['temp_in_high_zone'] = temp * high_temp_mask

        X['log_time_in_low_zone'] = log_time * low_temp_mask
        X['log_time_in_mid_zone'] = log_time * mid_temp_mask
        X['log_time_in_high_zone'] = log_time * high_temp_mask

        X['temp_log_time_low_zone'] = temp * log_time * low_temp_mask
        X['temp_log_time_mid_zone'] = temp * log_time * mid_temp_mask
        X['temp_log_time_high_zone'] = temp * log_time * high_temp_mask

        X['high_temp_long_time_coupling'] = (
            np.maximum(temp - self.mid_temp_threshold, 0.0) *
            np.maximum(time - self.short_time_threshold, 0.0)
        )

        X['dissolution_coupling'] = (
            np.maximum(temp - self.dissolution_temp, 0.0) *
            np.maximum(log_time, 0.0)
        )

        # =========================================================
        # 12. 动态状态推理特征
        # 通过“欠时效/峰值/过时效”三个态进行机理编码
        # =========================================================
        under_aging_indicator = (
            (1.0 - temp_activation) * 0.6
            + (1.0 - time_saturation) * 0.4
        )

        peak_aging_indicator = X['peak_window'].values * np.maximum(net_strengthening_index, 0.0)

        over_aging_indicator = (
            self._sigmoid((temp - self.mid_temp_threshold) / 10.0)
            * self._sigmoid((time - self.long_time_threshold) / 2.0)
            * (1.0 + softening_drive)
        )

        X['under_aging_indicator'] = under_aging_indicator
        X['peak_aging_indicator'] = peak_aging_indicator
        X['over_aging_indicator'] = over_aging_indicator

        # 面向输出的潜力指标
        strength_potential_index = (
            self.activation_weight * X['combined_activation'].values
            + self.strengthening_weight * np.maximum(net_strengthening_index, 0.0)
            + self.peak_window_weight * X['peak_window'].values
            - self.overaging_penalty_weight * over_aging_indicator
        )

        ductility_potential_index = (
            self.ductility_softening_weight * ductility_recovery_index
            + self.ductility_long_time_weight * long_time_mask
            + self.ductility_high_temp_weight * high_temp_mask
            - self.ductility_very_high_penalty_weight
            * very_high_temp_mask
            * np.maximum(temp - self.dissolution_temp, 0.0) / 20.0
        )

        yield_tensile_synergy_index = (
            0.80 * np.maximum(net_strengthening_index, 0.0)
            + 0.20 * X['combined_activation'].values
            - 0.15 * softening_drive
        )

        X['strength_potential_index'] = strength_potential_index
        X['ductility_potential_index'] = ductility_potential_index
        X['yield_tensile_synergy_index'] = yield_tensile_synergy_index

        # =========================================================
        # 13. 裁剪边界下的稳健惩罚特征
        # 用于抑制极端工艺条件的过拟合外推
        # =========================================================
        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * clipped_log_time

        X['high_temp_long_time_penalty'] = (
            np.maximum(clipped_temp - self.mid_temp_threshold, 0.0) *
            np.maximum(clipped_time - self.short_time_threshold, 0.0)
        )

        X['extreme_overprocess_penalty'] = (
            np.maximum(clipped_temp - self.dissolution_temp, 0.0) *
            np.maximum(clipped_time - self.long_time_threshold, 0.0)
        )

        # =========================================================
        # 14. 基于原始态基准的相对潜力特征
        # 使用已知材料基准常数，不引入标签泄漏
        # =========================================================
        base_strength_sum = self.base_tensile + self.base_yield
        base_strength_ratio = self._safe_divide(self.base_yield, self.base_tensile)

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio

        X['process_to_base_potential'] = (
            X['combined_activation'].values
            + 0.50 * net_strengthening_index
            - 0.003 * X['high_temp_long_time_penalty'].values
            - 0.004 * X['extreme_overprocess_penalty'].values
        )

        X['relative_strength_gain_potential'] = self._safe_divide(
            strength_potential_index,
            1.0 + base_strength_sum / 100.0
        )

        X['relative_ductility_shift_potential'] = self._safe_divide(
            ductility_potential_index,
            1.0 + self.base_strain
        )

        # =========================================================
        # 15. 多输出相关性启发特征
        # UTS 和 YS 一般同向变化；应变与强度通常存在竞争
        # =========================================================
        X['strength_vs_ductility_balance'] = strength_potential_index - ductility_potential_index

        X['yield_preference_index'] = (
            yield_tensile_synergy_index
            + 0.15 * mid_temp_mask
            - 0.10 * very_high_temp_mask
        )

        X['tensile_preference_index'] = (
            strength_potential_index
            + 0.10 * X['peak_window'].values
            - 0.10 * over_aging_indicator
        )

        # =========================================================
        # 16. 清理数值
        # =========================================================
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