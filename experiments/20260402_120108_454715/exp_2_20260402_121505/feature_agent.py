import pandas as pd
import numpy as np

# === 领域知识参数（基于材料科学文献/世界知识的启发式设定） ===
# 说明：
# 1) 这些参数用于构造“机理感知特征”，不是严格热力学反演；
# 2) 7499 属于高强 Al-Zn-Mg-Cu 系铝合金，典型规律为：
#    欠时效 -> 峰值时效 -> 过时效；
# 3) 小样本（29 条）下应避免过多自由度，因此采用“少而稳”的动态特征，
#    保留必要的非线性、温时耦合和分段机制，不再堆叠大量高度相关的人造指数。
DOMAIN_PARAMS = {
    # 基本物理常数
    'gas_constant_J_molK': 8.314,

    # 铝合金时效/扩散控制过程的典型激活能量级（启发式）
    'diffusion_activation_energy_J_mol': 65000.0,

    # 典型时效窗口（启发式，适用于高强 Al-Zn-Mg-Cu 合金的经验量级）
    'low_temp_threshold_C': 110.0,
    'peak_aging_temp_C': 130.0,
    'overaging_start_temp_C': 150.0,
    'precipitate_dissolution_temp_C': 180.0,

    'short_time_threshold_h': 4.0,
    'peak_aging_time_h': 8.0,
    'long_time_threshold_h': 12.0,

    # 峰值窗口宽度
    'peak_temp_window_C': 18.0,
    'peak_log_time_window': 0.70,

    # 动力学平滑尺度
    'temp_activation_scale_C': 10.0,
    'time_saturation_h': 6.0,
    'overaging_temp_scale_C': 22.0,

    # 数据有效工艺边界（用于裁剪，降低极端值影响）
    'min_valid_temp_C': 60.0,
    'max_valid_temp_C': 200.0,
    'min_valid_time_h': 0.0,
    'max_valid_time_h': 24.0,

    # 原始凝固态样品性能基准
    'base_strain_pct': 6.94,
    'base_tensile_MPa': 145.83,
    'base_yield_MPa': 96.60,
}


class FeatureAgent:
    """基于材料热处理机理的稳健动态特征工程"""

    def __init__(self):
        self.feature_names = []

        # 读取领域参数，避免硬编码
        self.R = DOMAIN_PARAMS['gas_constant_J_molK']
        self.Q = DOMAIN_PARAMS['diffusion_activation_energy_J_mol']

        self.low_temp_threshold = DOMAIN_PARAMS['low_temp_threshold_C']
        self.peak_temp = DOMAIN_PARAMS['peak_aging_temp_C']
        self.overaging_start_temp = DOMAIN_PARAMS['overaging_start_temp_C']
        self.dissolution_temp = DOMAIN_PARAMS['precipitate_dissolution_temp_C']

        self.short_time_threshold = DOMAIN_PARAMS['short_time_threshold_h']
        self.peak_time = DOMAIN_PARAMS['peak_aging_time_h']
        self.long_time_threshold = DOMAIN_PARAMS['long_time_threshold_h']

        self.peak_temp_window = DOMAIN_PARAMS['peak_temp_window_C']
        self.peak_log_time_window = DOMAIN_PARAMS['peak_log_time_window']

        self.temp_activation_scale = DOMAIN_PARAMS['temp_activation_scale_C']
        self.time_saturation_h = DOMAIN_PARAMS['time_saturation_h']
        self.overaging_temp_scale = DOMAIN_PARAMS['overaging_temp_scale_C']

        self.min_valid_temp = DOMAIN_PARAMS['min_valid_temp_C']
        self.max_valid_temp = DOMAIN_PARAMS['max_valid_temp_C']
        self.min_valid_time = DOMAIN_PARAMS['min_valid_time_h']
        self.max_valid_time = DOMAIN_PARAMS['max_valid_time_h']

        self.base_strain = DOMAIN_PARAMS['base_strain_pct']
        self.base_tensile = DOMAIN_PARAMS['base_tensile_MPa']
        self.base_yield = DOMAIN_PARAMS['base_yield_MPa']

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def engineer_features(self, df):
        """
        设计原则：
        1) 仅保留对小样本更友好的少量高价值特征；
        2) 显式表达：
           - 温度与时间非线性
           - 热激活/扩散动力学
           - 峰值时效邻近
           - 过时效/高温长时惩罚
           - 强度-塑性竞争
        3) 避免构造过多“二次组合的组合”，降低多重共线性和过拟合风险。
        """
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

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
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['temp_sq'] = temp ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)

        # =========================================================
        # 2. Arrhenius / 等效时效特征
        # 物理意义：组织演化受热激活控制，温度升高显著加快等效时效进程
        # =========================================================
        arrhenius_factor = np.exp(-self.Q / np.clip(self.R * clipped_temp_k, 1e-12, None))
        thermal_dose = clipped_time * arrhenius_factor
        equivalent_aging_index = clipped_log_time - self.Q / np.clip(self.R * clipped_temp_k, 1e-12, None)

        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = self._safe_log(thermal_dose)
        X['equivalent_aging_index'] = equivalent_aging_index

        # =========================================================
        # 3. 分段区间特征
        # 物理意义：
        # - 低温：扩散不足，强化受限
        # - 中温：更接近有效析出强化窗口
        # - 高温：容易快速达到峰值并进入过时效
        # =========================================================
        low_temp_mask = (temp < self.low_temp_threshold).astype(float)
        mid_temp_mask = ((temp >= self.low_temp_threshold) & (temp < self.overaging_start_temp)).astype(float)
        high_temp_mask = (temp >= self.overaging_start_temp).astype(float)
        very_high_temp_mask = (temp >= self.dissolution_temp).astype(float)

        short_time_mask = (time < self.short_time_threshold).astype(float)
        mid_time_mask = ((time >= self.short_time_threshold) & (time < self.long_time_threshold)).astype(float)
        long_time_mask = (time >= self.long_time_threshold).astype(float)

        X['is_low_temp'] = low_temp_mask
        X['is_mid_temp'] = mid_temp_mask
        X['is_high_temp'] = high_temp_mask
        X['is_very_high_temp'] = very_high_temp_mask

        X['is_short_time'] = short_time_mask
        X['is_mid_time'] = mid_time_mask
        X['is_long_time'] = long_time_mask

        X['relu_above_low_temp'] = np.maximum(temp - self.low_temp_threshold, 0.0)
        X['relu_above_peak_temp'] = np.maximum(temp - self.peak_temp, 0.0)
        X['relu_above_overaging_temp'] = np.maximum(temp - self.overaging_start_temp, 0.0)
        X['relu_above_dissolution_temp'] = np.maximum(temp - self.dissolution_temp, 0.0)

        X['relu_above_short_time'] = np.maximum(time - self.short_time_threshold, 0.0)
        X['relu_above_peak_time'] = np.maximum(time - self.peak_time, 0.0)
        X['relu_above_long_time'] = np.maximum(time - self.long_time_threshold, 0.0)

        # =========================================================
        # 4. 动态门控：工艺激活、时间饱和、峰值窗口
        # =========================================================
        temp_activation = self._sigmoid((clipped_temp - self.low_temp_threshold) / self.temp_activation_scale)
        time_saturation = 1.0 - np.exp(-clipped_time / self.time_saturation_h)

        peak_temp_gate = np.exp(-((temp - self.peak_temp) / self.peak_temp_window) ** 2)
        peak_time_gate = np.exp(-((log_time - np.log1p(self.peak_time)) / self.peak_log_time_window) ** 2)
        peak_window = peak_temp_gate * peak_time_gate

        X['temp_activation'] = temp_activation
        X['time_saturation'] = time_saturation
        X['combined_activation'] = temp_activation * time_saturation
        X['peak_temp_gate'] = peak_temp_gate
        X['peak_time_gate'] = peak_time_gate
        X['peak_window'] = peak_window

        # =========================================================
        # 5. 强化驱动力：随温时增加先增强，接近峰值区最有效
        # 采用“温度激活 × 时间饱和 × 峰值邻近修正”
        # =========================================================
        precipitation_drive = (
            0.55 * temp_activation +
            0.85 * time_saturation +
            0.75 * peak_window +
            0.35 * equivalent_aging_index
        )

        # 低温扩散不足，强化效率降低
        precipitation_drive = precipitation_drive - 0.25 * low_temp_mask * (1.0 - time_saturation)

        X['precipitation_drive'] = precipitation_drive

        # =========================================================
        # 6. 过时效/软化驱动力
        # 物理意义：高温 + 长时间 → 析出相粗化、回复、局部失稳
        # =========================================================
        overaging_temp_factor = np.exp(
            np.clip((temp - self.overaging_start_temp) / self.overaging_temp_scale, -8, 8)
        )
        overaging_index = log_time * overaging_temp_factor

        softening_drive = (
            0.035 * np.maximum(temp - self.overaging_start_temp, 0.0) * log_time +
            0.060 * high_temp_mask * np.maximum(time - self.short_time_threshold, 0.0) +
            0.020 * long_time_mask * np.maximum(temp - self.low_temp_threshold, 0.0) +
            0.055 * very_high_temp_mask * np.maximum(temp - self.dissolution_temp, 0.0) * (1.0 + log_time)
        )

        X['overaging_index'] = overaging_index
        X['softening_drive'] = softening_drive

        # =========================================================
        # 7. 净强化指数
        # 物理意义：真实强度水平通常由“强化 - 软化”共同决定
        # =========================================================
        net_strengthening_index = precipitation_drive - softening_drive
        X['net_strengthening_index'] = net_strengthening_index

        # =========================================================
        # 8. 显式温时耦合特征
        # 小样本下只保留最关键的几项
        # =========================================================
        X['mid_temp_log_time'] = mid_temp_mask * temp * log_time
        X['high_temp_log_time'] = high_temp_mask * temp * log_time
        X['high_temp_long_time_coupling'] = (
            np.maximum(temp - self.overaging_start_temp, 0.0) *
            np.maximum(time - self.short_time_threshold, 0.0)
        )
        X['extreme_overprocess_penalty'] = (
            np.maximum(temp - self.dissolution_temp, 0.0) *
            np.maximum(time - self.long_time_threshold, 0.0)
        )

        # =========================================================
        # 9. 时效状态指示：欠时效 / 峰值 / 过时效
        # =========================================================
        under_aging_indicator = (
            0.6 * (1.0 - temp_activation) +
            0.4 * (1.0 - time_saturation)
        )

        peak_aging_indicator = peak_window * np.maximum(net_strengthening_index, 0.0)

        over_aging_indicator = (
            self._sigmoid((temp - self.overaging_start_temp) / 10.0) *
            self._sigmoid((time - self.long_time_threshold) / 2.0) *
            (1.0 + np.maximum(softening_drive, 0.0))
        )

        X['under_aging_indicator'] = under_aging_indicator
        X['peak_aging_indicator'] = peak_aging_indicator
        X['over_aging_indicator'] = over_aging_indicator

        # =========================================================
        # 10. 面向多输出的稳健物理索引
        # 注意：
        # - tensile / yield 通常同向变化，因此构造共享强度潜力特征
        # - strain 通常与强化存在竞争，因此构造塑性恢复特征
        # =========================================================
        strength_potential_index = (
            0.50 * np.maximum(net_strengthening_index, 0.0) +
            0.25 * combined_activation +
            0.25 * peak_window -
            0.20 * over_aging_indicator
        )

        ductility_recovery_index = (
            0.55 * softening_drive +
            0.20 * long_time_mask +
            0.20 * high_temp_mask -
            0.30 * np.maximum(precipitation_drive, 0.0)
        )

        yield_tensile_synergy_index = (
            0.75 * np.maximum(net_strengthening_index, 0.0) +
            0.15 * combined_activation -
            0.10 * softening_drive
        )

        X['strength_potential_index'] = strength_potential_index
        X['ductility_recovery_index'] = ductility_recovery_index
        X['yield_tensile_synergy_index'] = yield_tensile_synergy_index
        X['strength_vs_ductility_balance'] = strength_potential_index - ductility_recovery_index

        # =========================================================
        # 11. 基于原始凝固态基准的相对工艺潜力
        # 不引入标签泄漏，只引入已知常数
        # =========================================================
        base_strength_sum = self.base_tensile + self.base_yield
        base_strength_ratio = self._safe_divide(self.base_yield, self.base_tensile)

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio

        X['relative_strength_gain_potential'] = self._safe_divide(
            strength_potential_index,
            1.0 + base_strength_sum / 100.0
        )

        X['relative_ductility_shift_potential'] = self._safe_divide(
            ductility_recovery_index,
            1.0 + self.base_strain
        )

        # =========================================================
        # 12. 清理数值
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