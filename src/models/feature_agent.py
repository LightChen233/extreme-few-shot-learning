import pandas as pd
import numpy as np

# === 领域知识参数（基于材料科学文献/材料世界知识的启发式设定） ===
# 说明：
# 1. 这些参数用于构造“机制感知特征”，不是严格热力学反演；
# 2. 数值取典型铝合金时效/析出/过时效过程的经验量级，目标是在小样本下提高泛化稳定性；
# 3. 7499 属于高强 Al-Zn-Mg-Cu 系铝合金，性能演化通常受析出强化、扩散动力学、过时效软化共同控制。
DOMAIN_PARAMS = {
    # 基本物理常数
    'gas_constant_J_molK': 8.314,

    # 析出/扩散相关经验常数（量级参考铝合金析出扩散控制过程）
    'diffusion_activation_energy_J_mol': 65000.0,

    # 热处理关键温度区间（启发式）
    'low_temp_threshold_C': 110.0,             # 低温：扩散较慢，组织演化弱
    'peak_aging_temp_C': 130.0,                # 峰值时效附近典型温度
    'mid_temp_threshold_C': 150.0,             # 强化到过时效机制切换敏感区
    'grain_coarsening_threshold_C': 150.0,     # 粗化/过时效风险开始明显
    'precipitate_dissolution_temp_C': 180.0,   # 高温下析出相溶解/失稳风险增加

    # 时间阈值（启发式）
    'short_time_threshold_h': 4.0,             # 初期时效阶段
    'peak_aging_time_h': 8.0,                  # 峰值时效附近典型时间
    'long_time_threshold_h': 12.0,             # 长时效/过时效风险增强
    'saturation_time_h': 6.0,                  # 饱和型时间响应尺度

    # 经验窗口宽度
    'peak_temp_window_C': 20.0,
    'peak_log_time_window': 0.8,

    # 参考工艺边界（用于裁剪，降低极端外推噪声）
    'min_valid_temp_C': 60.0,
    'max_valid_temp_C': 200.0,
    'min_valid_time_h': 0.0,
    'max_valid_time_h': 24.0,

    # 原始凝固态样品性能基准
    'base_strain_pct': 6.94,
    'base_tensile_MPa': 145.83,
    'base_yield_MPa': 96.60,

    # 机制特征权重（启发式，小样本下强调稳健性）
    'low_temp_precip_coeff': 0.6,
    'mid_temp_precip_coeff': 1.2,
    'high_temp_precip_coeff': 1.0,

    'softening_temp_coeff': 0.03,
    'softening_time_coeff': 0.08,
    'softening_long_time_coeff': 0.015,

    'overaging_temp_scale_C': 25.0,
    'activation_temp_scale_C': 10.0,
}


class FeatureAgent:
    """基于材料热处理机理的动态特征工程"""

    def __init__(self):
        self.feature_names = []

        # 从 DOMAIN_PARAMS 中读取，避免硬编码
        self.R = DOMAIN_PARAMS['gas_constant_J_molK']
        self.Q = DOMAIN_PARAMS['diffusion_activation_energy_J_mol']

        self.low_temp_threshold = DOMAIN_PARAMS['low_temp_threshold_C']
        self.peak_temp = DOMAIN_PARAMS['peak_aging_temp_C']
        self.mid_temp_threshold = DOMAIN_PARAMS['mid_temp_threshold_C']
        self.coarsening_temp = DOMAIN_PARAMS['grain_coarsening_threshold_C']
        self.dissolution_temp = DOMAIN_PARAMS['precipitate_dissolution_temp_C']

        self.short_time_threshold = DOMAIN_PARAMS['short_time_threshold_h']
        self.peak_time = DOMAIN_PARAMS['peak_aging_time_h']
        self.long_time_threshold = DOMAIN_PARAMS['long_time_threshold_h']
        self.saturation_time = DOMAIN_PARAMS['saturation_time_h']

        self.peak_temp_window = DOMAIN_PARAMS['peak_temp_window_C']
        self.peak_log_time_window = DOMAIN_PARAMS['peak_log_time_window']

        self.min_valid_temp = DOMAIN_PARAMS['min_valid_temp_C']
        self.max_valid_temp = DOMAIN_PARAMS['max_valid_temp_C']
        self.min_valid_time = DOMAIN_PARAMS['min_valid_time_h']
        self.max_valid_time = DOMAIN_PARAMS['max_valid_time_h']

        self.base_strain = DOMAIN_PARAMS['base_strain_pct']
        self.base_tensile = DOMAIN_PARAMS['base_tensile_MPa']
        self.base_yield = DOMAIN_PARAMS['base_yield_MPa']

        self.low_temp_precip_coeff = DOMAIN_PARAMS['low_temp_precip_coeff']
        self.mid_temp_precip_coeff = DOMAIN_PARAMS['mid_temp_precip_coeff']
        self.high_temp_precip_coeff = DOMAIN_PARAMS['high_temp_precip_coeff']

        self.softening_temp_coeff = DOMAIN_PARAMS['softening_temp_coeff']
        self.softening_time_coeff = DOMAIN_PARAMS['softening_time_coeff']
        self.softening_long_time_coeff = DOMAIN_PARAMS['softening_long_time_coeff']

        self.overaging_temp_scale = DOMAIN_PARAMS['overaging_temp_scale_C']
        self.activation_temp_scale = DOMAIN_PARAMS['activation_temp_scale_C']

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def engineer_features(self, df):
        """构造具有动态机制推理的材料热处理特征"""
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = self._safe_divide(1.0, temp_k)

        clipped_temp = np.clip(temp, self.min_valid_temp, self.max_valid_temp)
        clipped_time = np.clip(time, self.min_valid_time, self.max_valid_time)
        clipped_temp_k = clipped_temp + 273.15
        clipped_log_time = self._safe_log(clipped_time)

        # =========================================================
        # 1. 基础工艺主特征
        # =========================================================
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # =========================================================
        # 2. 基础非线性与交互项
        # =========================================================
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp'] = self._safe_divide(log_time, temp_k)
        X['temp_over_time1p'] = self._safe_divide(temp, 1.0 + time)

        # =========================================================
        # 3. Arrhenius / 扩散动力学特征
        # 物理意义：析出与组织演化速率受热激活控制
        # =========================================================
        arrhenius_factor = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        clipped_arrhenius_factor = np.exp(-self.Q / np.clip(self.R * clipped_temp_k, 1e-12, None))

        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = time * arrhenius_factor
        X['log_thermal_dose'] = self._safe_log(X['thermal_dose'].values)
        X['temp_log_thermal_dose'] = temp * X['log_thermal_dose'].values

        # 等效热处理强度：高温短时 ≈ 低温长时的一种粗略状态表示
        X['equivalent_aging_index'] = log_time - self.Q / np.clip(self.R * temp_k, 1e-12, None)
        X['thermal_dose_clipped'] = clipped_time * clipped_arrhenius_factor
        X['log_thermal_dose_clipped'] = self._safe_log(X['thermal_dose_clipped'].values)

        # =========================================================
        # 4. 温度区间机制划分
        # =========================================================
        low_temp_mask = (temp < self.low_temp_threshold).astype(float)
        mid_temp_mask = ((temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)).astype(float)
        high_temp_mask = (temp >= self.mid_temp_threshold).astype(float)
        very_high_temp_mask = (temp >= self.dissolution_temp).astype(float)

        X['is_low_temp'] = low_temp_mask
        X['is_mid_temp'] = mid_temp_mask
        X['is_high_temp'] = high_temp_mask
        X['is_very_high_temp'] = very_high_temp_mask

        X['dist_to_low_temp_th'] = temp - self.low_temp_threshold
        X['dist_to_peak_temp'] = temp - self.peak_temp
        X['dist_to_mid_temp_th'] = temp - self.mid_temp_threshold
        X['dist_to_dissolution_temp'] = temp - self.dissolution_temp

        X['relu_above_low_temp'] = np.maximum(temp - self.low_temp_threshold, 0)
        X['relu_above_peak_temp'] = np.maximum(temp - self.peak_temp, 0)
        X['relu_above_mid_temp'] = np.maximum(temp - self.mid_temp_threshold, 0)
        X['relu_above_dissolution_temp'] = np.maximum(temp - self.dissolution_temp, 0)
        X['relu_below_low_temp'] = np.maximum(self.low_temp_threshold - temp, 0)

        # =========================================================
        # 5. 时间区间机制划分
        # =========================================================
        short_time_mask = (time < self.short_time_threshold).astype(float)
        mid_time_mask = ((time >= self.short_time_threshold) & (time < self.long_time_threshold)).astype(float)
        long_time_mask = (time >= self.long_time_threshold).astype(float)

        X['is_short_time'] = short_time_mask
        X['is_mid_time'] = mid_time_mask
        X['is_long_time'] = long_time_mask

        X['dist_to_short_time'] = time - self.short_time_threshold
        X['dist_to_peak_time'] = time - self.peak_time
        X['dist_to_long_time'] = time - self.long_time_threshold

        X['relu_above_short_time'] = np.maximum(time - self.short_time_threshold, 0)
        X['relu_above_peak_time'] = np.maximum(time - self.peak_time, 0)
        X['relu_above_long_time'] = np.maximum(time - self.long_time_threshold, 0)
        X['relu_below_short_time'] = np.maximum(self.short_time_threshold - time, 0)

        # =========================================================
        # 6. 热处理状态门控特征
        # 物理意义：
        # - 温度门控：温度越接近有效时效区，析出强化越容易发生
        # - 时间门控：时效有“快速增长→趋于饱和”特性
        # =========================================================
        temp_activation = self._sigmoid((clipped_temp - self.low_temp_threshold) / self.activation_temp_scale)
        peak_temp_gate = np.exp(-((temp - self.peak_temp) / self.peak_temp_window) ** 2)
        time_saturation = 1.0 - np.exp(-clipped_time / self.saturation_time)
        peak_time_gate = np.exp(-((clipped_log_time - np.log1p(self.peak_time)) / self.peak_log_time_window) ** 2)

        X['temp_activation'] = temp_activation
        X['time_saturation'] = time_saturation
        X['peak_temp_gate'] = peak_temp_gate
        X['peak_time_gate'] = peak_time_gate
        X['combined_activation'] = temp_activation * time_saturation
        X['peak_window'] = peak_temp_gate * peak_time_gate

        # =========================================================
        # 7. 强化驱动力 precipitation_drive
        # 动态逻辑：
        # - 低温区：主要受扩散不足限制，时间累积起作用
        # - 中温区：最有利于形成强化析出相，强化最敏感
        # - 高温区：初期强化快，但随后更易进入过时效/粗化
        # =========================================================
        precipitation_drive = np.zeros_like(temp, dtype=float)

        low_idx = temp < self.low_temp_threshold
        precipitation_drive[low_idx] = (
            self.low_temp_precip_coeff * log_time[low_idx]
            + 0.004 * (temp[low_idx] - self.min_valid_temp) * log_time[low_idx]
        )

        mid_idx = (temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)
        precipitation_drive[mid_idx] = (
            self.mid_temp_precip_coeff * log_time[mid_idx]
            + 0.012 * (temp[mid_idx] - self.low_temp_threshold) * log_time[mid_idx]
            - 0.06 * np.maximum(time[mid_idx] - self.long_time_threshold, 0)
        )

        high_idx = temp >= self.mid_temp_threshold
        precipitation_drive[high_idx] = (
            self.high_temp_precip_coeff * np.minimum(log_time[high_idx], np.log1p(self.long_time_threshold))
            + 0.01 * (self.mid_temp_threshold - self.low_temp_threshold)
            - 0.10 * np.maximum(log_time[high_idx] - np.log1p(self.short_time_threshold), 0)
        )

        # 高温过高时析出稳定性下降，给一个溶解/失稳惩罚
        dissolution_penalty = 0.08 * np.maximum(temp - self.dissolution_temp, 0) * (0.5 + log_time)
        precipitation_drive = precipitation_drive - dissolution_penalty

        X['precipitation_drive'] = precipitation_drive

        # =========================================================
        # 8. 软化/过时效驱动力 softening_drive
        # 物理意义：
        # 高温 + 长时间 → 析出相粗化、回复、局部溶解，强度回落
        # =========================================================
        overaging_index = log_time * np.exp((temp - self.mid_temp_threshold) / self.overaging_temp_scale)

        softening_drive = (
            np.maximum(temp - self.coarsening_temp, 0) * self.softening_temp_coeff * log_time
            + high_temp_mask * np.maximum(time - self.short_time_threshold, 0) * self.softening_time_coeff
            + long_time_mask * np.maximum(temp - self.low_temp_threshold, 0) * self.softening_long_time_coeff
            + very_high_temp_mask * 0.06 * np.maximum(temp - self.dissolution_temp, 0) * (1.0 + log_time)
        )

        X['overaging_index'] = overaging_index
        X['softening_drive'] = softening_drive
        X['net_strengthening_index'] = precipitation_drive - softening_drive

        # =========================================================
        # 9. 峰值时效邻近度特征
        # 物理意义：性能往往在某个温-时窗口附近达到最优
        # =========================================================
        temp_peak_proximity = -((temp - self.peak_temp) / self.peak_temp_window) ** 2
        time_peak_proximity = -((log_time - np.log1p(self.peak_time)) / self.peak_log_time_window) ** 2
        joint_peak_proximity = temp_peak_proximity + time_peak_proximity

        X['temp_peak_proximity'] = temp_peak_proximity
        X['time_peak_proximity'] = time_peak_proximity
        X['joint_peak_proximity'] = joint_peak_proximity
        X['peak_window_strength_drive'] = X['peak_window'].values * precipitation_drive
        X['peak_window_softening_balance'] = X['peak_window'].values * X['net_strengthening_index'].values

        # =========================================================
        # 10. 强塑性竞争特征
        # 物理意义：
        # - 强化析出相增多通常提升强度但降低塑性
        # - 过时效/软化有时会带来一定塑性恢复
        # =========================================================
        ductility_recovery_index = (
            0.5 * softening_drive
            + 0.3 * high_temp_mask
            + 0.2 * long_time_mask
            - 0.3 * precipitation_drive
        )

        X['ductility_recovery_index'] = ductility_recovery_index
        X['strength_ductility_tradeoff'] = X['net_strengthening_index'].values - ductility_recovery_index

        # =========================================================
        # 11. 分区交互项：显式表示不同机制区间内的响应斜率
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
            np.maximum(temp - self.mid_temp_threshold, 0) *
            np.maximum(time - self.short_time_threshold, 0)
        )

        X['dissolution_coupling'] = (
            np.maximum(temp - self.dissolution_temp, 0) *
            np.maximum(log_time, 0)
        )

        # =========================================================
        # 12. 机制态势特征：用“强化/软化/峰值窗口”进行动态推理组合
        # =========================================================
        # 欠时效：温度或时间不足，强化尚未充分展开
        under_aging_indicator = (
            (1.0 - temp_activation) * 0.6
            + (1.0 - time_saturation) * 0.4
        )

        # 峰值时效态：处于较优温时窗口，同时净强化较高
        peak_aging_indicator = (
            X['peak_window'].values * np.maximum(X['net_strengthening_index'].values, 0)
        )

        # 过时效态：高温/长时/软化共同增强
        over_aging_indicator = (
            self._sigmoid((temp - self.mid_temp_threshold) / 10.0)
            * self._sigmoid((time - self.long_time_threshold) / 2.0)
            * (1.0 + softening_drive)
        )

        X['under_aging_indicator'] = under_aging_indicator
        X['peak_aging_indicator'] = peak_aging_indicator
        X['over_aging_indicator'] = over_aging_indicator

        # 强度潜力：强调强化主导、峰值窗口和工艺激活
        X['strength_potential_index'] = (
            0.45 * X['combined_activation'].values
            + 0.75 * np.maximum(X['net_strengthening_index'].values, 0)
            + 0.35 * X['peak_window'].values
            - 0.25 * over_aging_indicator
        )

        # 塑性潜力：强调软化恢复，但避免极端高温长时误判为优塑性
        X['ductility_potential_index'] = (
            0.55 * ductility_recovery_index
            + 0.20 * long_time_mask
            + 0.15 * high_temp_mask
            - 0.20 * very_high_temp_mask * np.maximum(temp - self.dissolution_temp, 0) / 20.0
        )

        # 屈强协同：UTS 与 YS 通常同向变化，此特征强调稳定强化状态
        X['yield_tensile_synergy_index'] = (
            0.8 * np.maximum(X['net_strengthening_index'].values, 0)
            + 0.2 * X['combined_activation'].values
            - 0.15 * softening_drive
        )

        # =========================================================
        # 13. 裁剪特征与物理约束特征
        # =========================================================
        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * clipped_log_time

        X['high_temp_long_time_penalty'] = (
            np.maximum(clipped_temp - self.mid_temp_threshold, 0) *
            np.maximum(clipped_time - self.short_time_threshold, 0)
        )

        X['extreme_overprocess_penalty'] = (
            np.maximum(clipped_temp - self.dissolution_temp, 0) *
            np.maximum(clipped_time - self.long_time_threshold, 0)
        )

        # =========================================================
        # 14. 基于原始态基准的工艺潜力指标
        # 不使用标签泄漏，只引入已知凝固态材料基准常数
        # =========================================================
        base_strength_sum = self.base_tensile + self.base_yield
        base_strength_ratio = self._safe_divide(self.base_yield, self.base_tensile)

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio

        X['process_to_base_potential'] = (
            X['combined_activation'].values
            + 0.5 * X['net_strengthening_index'].values
            - 0.3 * X['high_temp_long_time_penalty'].values / 100.0
            - 0.2 * X['extreme_overprocess_penalty'].values / 100.0
        )

        X['relative_strength_gain_potential'] = (
            self._safe_divide(X['strength_potential_index'].values, 1.0 + base_strength_sum / 100.0)
        )

        X['relative_ductility_shift_potential'] = (
            self._safe_divide(X['ductility_potential_index'].values, 1.0 + self.base_strain)
        )

        # =========================================================
        # 15. 多输出相关性启发特征
        # 物理意义：
        # - UTS 与 YS 通常高度正相关
        # - 应变与强度常存在竞争关系
        # =========================================================
        X['strength_vs_ductility_balance'] = (
            X['strength_potential_index'].values - X['ductility_potential_index'].values
        )

        X['yield_preference_index'] = (
            X['yield_tensile_synergy_index'].values
            + 0.15 * mid_temp_mask
            - 0.10 * very_high_temp_mask
        )

        X['tensile_preference_index'] = (
            X['strength_potential_index'].values
            + 0.10 * X['peak_window'].values
            - 0.10 * X['over_aging_indicator'].values
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