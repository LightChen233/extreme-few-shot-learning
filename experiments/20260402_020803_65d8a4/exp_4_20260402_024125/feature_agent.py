import pandas as pd
import numpy as np

# === 领域知识参数（基于材料科学/铝合金热处理常识的启发式常数） ===
DOMAIN_PARAMS = {
    # 基础物理常数
    'gas_constant_J_mol_K': 8.314,

    # 铝合金析出/扩散相关经验量级
    'diffusion_activation_energy_J_mol': 65000.0,   # 扩散/析出控制过程经验激活能
    'coarsening_activation_energy_J_mol': 80000.0,  # 粗化/过时效更高能垒的经验量级

    # 热处理关键窗口（启发式，适合小样本稳健建模）
    'peak_aging_temp_C': 130.0,             # 典型峰值时效温度中心
    'peak_aging_time_h': 8.0,               # 典型峰值时效时间中心
    'low_temp_threshold_C': 110.0,          # 低温扩散不足阈值
    'mid_temp_threshold_C': 150.0,          # 过时效/粗化风险开始显著上升阈值
    'grain_coarsening_threshold_C': 150.0,  # 晶粒/析出相粗化风险阈值
    'precipitate_dissolution_temp_C': 180.0,# 析出相溶解/回复更强阈值

    # 时间窗口
    'short_time_threshold_h': 4.0,          # 初期时效阈值
    'long_time_threshold_h': 12.0,          # 长时时效阈值
    'saturation_time_h': 6.0,               # 饱和增长时间尺度

    # 机制平滑尺度
    'temp_window_sigma_C': 20.0,            # 温度峰值窗口宽度
    'log_time_window_sigma': 0.8,           # 对数时间窗口宽度
    'activation_slope_temp_C': 10.0,        # 温度激活sigmoid斜率
    'softening_temp_scale_C': 25.0,         # 软化指数温度尺度

    # 参考工艺范围（用于约束外推）
    'min_reasonable_temp_C': 60.0,
    'max_reasonable_temp_C': 200.0,
    'max_reasonable_time_h': 24.0,

    # 原始凝固态样品基准性能
    'base_strain_pct': 6.94,
    'base_tensile_MPa': 145.83,
    'base_yield_MPa': 96.60,
}


class FeatureAgent:
    """基于材料热处理机理的动态特征工程"""

    def __init__(self):
        self.feature_names = []

        # 从 DOMAIN_PARAMS 读取，不使用硬编码
        self.R = DOMAIN_PARAMS['gas_constant_J_mol_K']
        self.Q_diff = DOMAIN_PARAMS['diffusion_activation_energy_J_mol']
        self.Q_coarsen = DOMAIN_PARAMS['coarsening_activation_energy_J_mol']

        self.peak_temp = DOMAIN_PARAMS['peak_aging_temp_C']
        self.peak_time = DOMAIN_PARAMS['peak_aging_time_h']

        self.low_temp_threshold = DOMAIN_PARAMS['low_temp_threshold_C']
        self.mid_temp_threshold = DOMAIN_PARAMS['mid_temp_threshold_C']
        self.coarsening_threshold = DOMAIN_PARAMS['grain_coarsening_threshold_C']
        self.dissolution_threshold = DOMAIN_PARAMS['precipitate_dissolution_temp_C']

        self.short_time_threshold = DOMAIN_PARAMS['short_time_threshold_h']
        self.long_time_threshold = DOMAIN_PARAMS['long_time_threshold_h']
        self.saturation_time = DOMAIN_PARAMS['saturation_time_h']

        self.temp_sigma = DOMAIN_PARAMS['temp_window_sigma_C']
        self.log_time_sigma = DOMAIN_PARAMS['log_time_window_sigma']
        self.activation_slope_temp = DOMAIN_PARAMS['activation_slope_temp_C']
        self.softening_temp_scale = DOMAIN_PARAMS['softening_temp_scale_C']

        self.min_temp = DOMAIN_PARAMS['min_reasonable_temp_C']
        self.max_temp = DOMAIN_PARAMS['max_reasonable_temp_C']
        self.max_time = DOMAIN_PARAMS['max_reasonable_time_h']

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
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-6, None)

        clipped_temp = np.clip(temp, self.min_temp, self.max_temp)
        clipped_time = np.clip(time, 0, self.max_time)
        clipped_temp_k = clipped_temp + 273.15
        clipped_log_time = self._safe_log(clipped_time)

        # -----------------------------
        # 1. 基础工艺特征
        # -----------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # -----------------------------
        # 2. 基础非线性与交互
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
        # 3. 基于扩散/热激活的动力学特征
        # 反映温度升高会加速析出、长大、粗化过程
        # -----------------------------
        arrhenius_diff = np.exp(-self.Q_diff / (self.R * np.clip(temp_k, 1e-6, None)))
        arrhenius_coarsen = np.exp(-self.Q_coarsen / (self.R * np.clip(temp_k, 1e-6, None)))

        X['arrhenius_diff'] = arrhenius_diff
        X['arrhenius_coarsen'] = arrhenius_coarsen

        X['thermal_dose_diff'] = time * arrhenius_diff
        X['thermal_dose_coarsen'] = time * arrhenius_coarsen
        X['log_thermal_dose_diff'] = self._safe_log(X['thermal_dose_diff'].values)
        X['log_thermal_dose_coarsen'] = self._safe_log(X['thermal_dose_coarsen'].values)

        # 类 Larson-Miller / tempering parameter 启发式温时综合参数
        X['temp_log_time_kelvin'] = temp_k * log_time
        X['temp_log10_time_kelvin'] = temp_k * np.log10(np.clip(time + 1.0, 1.0, None))

        # -----------------------------
        # 4. 温度区间动态分段
        # -----------------------------
        low_temp_mask = (temp < self.low_temp_threshold).astype(float)
        mid_temp_mask = ((temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)).astype(float)
        high_temp_mask = (temp >= self.mid_temp_threshold).astype(float)
        very_high_temp_mask = (temp >= self.dissolution_threshold).astype(float)

        X['is_low_temp'] = low_temp_mask
        X['is_mid_temp'] = mid_temp_mask
        X['is_high_temp'] = high_temp_mask
        X['is_very_high_temp'] = very_high_temp_mask

        X['dist_to_low_temp_th'] = temp - self.low_temp_threshold
        X['dist_to_mid_temp_th'] = temp - self.mid_temp_threshold
        X['dist_to_dissolution_th'] = temp - self.dissolution_threshold

        X['relu_above_low_temp'] = np.maximum(temp - self.low_temp_threshold, 0)
        X['relu_above_mid_temp'] = np.maximum(temp - self.mid_temp_threshold, 0)
        X['relu_above_dissolution_temp'] = np.maximum(temp - self.dissolution_threshold, 0)
        X['relu_below_low_temp'] = np.maximum(self.low_temp_threshold - temp, 0)

        # -----------------------------
        # 5. 时间区间动态分段
        # -----------------------------
        short_time_mask = (time < self.short_time_threshold).astype(float)
        mid_time_mask = ((time >= self.short_time_threshold) & (time < self.long_time_threshold)).astype(float)
        long_time_mask = (time >= self.long_time_threshold).astype(float)

        X['is_short_time'] = short_time_mask
        X['is_mid_time'] = mid_time_mask
        X['is_long_time'] = long_time_mask

        X['relu_above_short_time'] = np.maximum(time - self.short_time_threshold, 0)
        X['relu_above_long_time'] = np.maximum(time - self.long_time_threshold, 0)
        X['relu_below_short_time'] = np.maximum(self.short_time_threshold - time, 0)

        # -----------------------------
        # 6. 工艺激活与饱和
        # 物理意义：
        # - 温度达到一定阈值后析出/扩散激活增强
        # - 时间存在前快后慢的饱和效应
        # -----------------------------
        temp_activation = self._sigmoid((clipped_temp - self.low_temp_threshold) / self.activation_slope_temp)
        high_temp_activation = self._sigmoid((clipped_temp - self.mid_temp_threshold) / self.activation_slope_temp)
        very_high_temp_activation = self._sigmoid((clipped_temp - self.dissolution_threshold) / self.activation_slope_temp)

        time_saturation = 1.0 - np.exp(-clipped_time / self.saturation_time)
        long_time_saturation = 1.0 - np.exp(-np.maximum(clipped_time - self.short_time_threshold, 0) / self.saturation_time)

        X['temp_activation'] = temp_activation
        X['high_temp_activation'] = high_temp_activation
        X['very_high_temp_activation'] = very_high_temp_activation
        X['time_saturation'] = time_saturation
        X['long_time_saturation'] = long_time_saturation
        X['combined_activation'] = temp_activation * time_saturation

        # -----------------------------
        # 7. 峰值时效邻近度
        # 强度常在特定温-时窗口附近达到峰值
        # -----------------------------
        opt_log_time_center = np.log1p(self.peak_time)

        temp_peak_proximity = -((temp - self.peak_temp) / self.temp_sigma) ** 2
        time_peak_proximity = -((log_time - opt_log_time_center) / self.log_time_sigma) ** 2
        joint_peak_proximity = temp_peak_proximity + time_peak_proximity
        peak_window = np.exp(joint_peak_proximity)

        X['temp_peak_proximity'] = temp_peak_proximity
        X['time_peak_proximity'] = time_peak_proximity
        X['joint_peak_proximity'] = joint_peak_proximity
        X['peak_window'] = peak_window

        # -----------------------------
        # 8. 机制驱动：强化驱动力
        # 分区动态推理，而非简单堆叠
        # -----------------------------
        precipitation_drive = np.zeros_like(temp, dtype=float)

        # 低温：扩散不足，时间累积主导，强化增长较慢
        low_idx = temp < self.low_temp_threshold
        precipitation_drive[low_idx] = (
            0.45 * log_time[low_idx]
            + 0.0035 * (temp[low_idx] - self.min_temp) * log_time[low_idx]
        )

        # 中温：析出强化最显著，温时协同最强
        mid_idx = (temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)
        precipitation_drive[mid_idx] = (
            1.10 * log_time[mid_idx]
            + 0.011 * (temp[mid_idx] - self.low_temp_threshold) * log_time[mid_idx]
            - 0.045 * np.maximum(time[mid_idx] - self.long_time_threshold, 0)
        )

        # 高温：短时可快速强化，但长时强化趋于衰减并转向过时效
        high_idx = temp >= self.mid_temp_threshold
        precipitation_drive[high_idx] = (
            0.95 * np.minimum(log_time[high_idx], np.log1p(self.long_time_threshold))
            + 0.006 * (self.mid_temp_threshold - self.low_temp_threshold)
            - 0.12 * np.maximum(log_time[high_idx] - np.log1p(self.short_time_threshold), 0)
        )

        X['precipitation_drive'] = precipitation_drive

        # -----------------------------
        # 9. 机制驱动：软化/过时效/溶解驱动力
        # 高温+长时会推动粗化、回复、软化
        # 极高温下析出相可能部分溶解，强度进一步受损
        # -----------------------------
        softening_drive = (
            np.maximum(temp - self.coarsening_threshold, 0) * 0.028 * log_time
            + high_temp_mask * np.maximum(time - self.short_time_threshold, 0) * 0.075
            + long_time_mask * np.maximum(temp - self.low_temp_threshold, 0) * 0.013
        )

        dissolution_drive = (
            very_high_temp_mask
            * np.maximum(temp - self.dissolution_threshold, 0)
            * (0.020 + 0.015 * log_time)
        )

        overaging_index = log_time * np.exp((temp - self.mid_temp_threshold) / self.softening_temp_scale)

        X['softening_drive'] = softening_drive
        X['dissolution_drive'] = dissolution_drive
        X['overaging_index'] = overaging_index

        # 净强化指数
        net_strengthening_index = precipitation_drive - softening_drive - 0.6 * dissolution_drive
        X['net_strengthening_index'] = net_strengthening_index

        # -----------------------------
        # 10. 温时等效与机制竞争
        # 高温短时 vs 低温长时近似等效
        # -----------------------------
        X['temp_time_equiv'] = self._safe_divide(temp * log_time, inv_temp_k * 1000.0 + 1e-6)
        X['kinetics_competition'] = X['thermal_dose_diff'].values - X['thermal_dose_coarsen'].values
        X['activation_overaging_balance'] = X['combined_activation'].values - 0.5 * high_temp_activation * long_time_saturation

        # -----------------------------
        # 11. 分区交互：不同温区下响应斜率不同
        # -----------------------------
        X['temp_in_low_zone'] = temp * low_temp_mask
        X['temp_in_mid_zone'] = temp * mid_temp_mask
        X['temp_in_high_zone'] = temp * high_temp_mask

        X['log_time_in_low_zone'] = log_time * low_temp_mask
        X['log_time_in_mid_zone'] = log_time * mid_temp_mask
        X['log_time_in_high_zone'] = log_time * high_temp_mask

        X['temp_log_time_low_zone'] = temp * log_time * low_temp_mask
        X['temp_log_time_mid_zone'] = temp * log_time * mid_temp_mask
        X['temp_log_time_high_zone'] = temp * log_time * high_temp_mask

        # -----------------------------
        # 12. 峰值窗口调制后的强化/软化平衡
        # 在接近峰值窗口时，强化信息更可信
        # -----------------------------
        X['peak_window_strength_drive'] = peak_window * precipitation_drive
        X['peak_window_net_strength'] = peak_window * net_strengthening_index
        X['peak_window_softening_balance'] = peak_window * (precipitation_drive - softening_drive)

        # -----------------------------
        # 13. 塑性相关启发特征
        # 强度与塑性常存在竞争，但过时效可能使塑性回升
        # -----------------------------
        ductility_recovery_index = (
            0.55 * softening_drive
            + 0.35 * high_temp_mask
            + 0.22 * long_time_mask
            + 0.25 * dissolution_drive
            - 0.30 * precipitation_drive
        )

        strength_ductility_tradeoff = net_strengthening_index - ductility_recovery_index

        X['ductility_recovery_index'] = ductility_recovery_index
        X['strength_ductility_tradeoff'] = strength_ductility_tradeoff

        # -----------------------------
        # 14. 物理约束型截断特征
        # 防止极端输入导致小样本外推失稳
        # -----------------------------
        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * clipped_log_time
        X['clipped_temp_time'] = clipped_temp * clipped_time

        X['high_temp_long_time_penalty'] = (
            np.maximum(clipped_temp - self.mid_temp_threshold, 0)
            * np.maximum(clipped_time - self.short_time_threshold, 0)
        )

        X['very_high_temp_penalty'] = (
            np.maximum(clipped_temp - self.dissolution_threshold, 0)
            * (1.0 + clipped_log_time)
        )

        # -----------------------------
        # 15. 相对原始铸态的工艺潜力指标
        # 不使用标签，仅使用已知常数构造无泄漏特征
        # -----------------------------
        base_strength_sum = self.base_tensile + self.base_yield
        base_strength_ratio = self._safe_divide(self.base_yield, self.base_tensile)

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['base_strain'] = self.base_strain

        X['process_to_base_potential'] = (
            X['combined_activation'].values
            + 0.50 * net_strengthening_index
            - 0.003 * X['high_temp_long_time_penalty'].values
            - 0.004 * X['very_high_temp_penalty'].values
        )

        X['relative_strengthening_potential'] = (
            1.0
            + 0.40 * precipitation_drive
            - 0.25 * softening_drive
            - 0.20 * dissolution_drive
        )

        X['relative_ductility_potential'] = (
            1.0
            + 0.30 * ductility_recovery_index
            - 0.15 * precipitation_drive
        )

        # -----------------------------
        # 16. 针对三目标关系的结构性特征
        # 屈服/抗拉正相关，应变与强度常竞争
        # -----------------------------
        X['strength_path_index'] = (
            0.60 * net_strengthening_index
            + 0.25 * peak_window
            + 0.15 * X['combined_activation'].values
        )

        X['yield_bias_index'] = (
            0.70 * net_strengthening_index
            + 0.20 * arrhenius_diff
            - 0.15 * dissolution_drive
        )

        X['tensile_bias_index'] = (
            0.55 * net_strengthening_index
            + 0.20 * precipitation_drive
            - 0.10 * softening_drive
        )

        X['strain_bias_index'] = (
            0.60 * ductility_recovery_index
            - 0.35 * net_strengthening_index
            + 0.15 * long_time_saturation
        )

        # -----------------------------
        # 17. 数值清理
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