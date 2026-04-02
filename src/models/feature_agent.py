import pandas as pd
import numpy as np


class FeatureAgent:
    """基于材料热处理机理的动态特征工程"""

    def __init__(self):
        self.feature_names = []

        # 原始凝固态样品基准性能
        self.base_strain = 6.94
        self.base_tensile = 145.83
        self.base_yield = 96.60

        # 7499铝合金热处理经验阈值（小样本下采用稳健的启发式分段）
        # 物理含义：
        # - 低温区：扩散慢，组织演化弱
        # - 中温区：析出强化主导，可能接近峰值时效
        # - 高温区：粗化/过时效/软化风险上升
        self.low_temp_threshold = 110.0
        self.mid_temp_threshold = 150.0

        # 时间阈值（前期快、后期慢，长时间易过时效）
        self.short_time_threshold = 4.0
        self.long_time_threshold = 12.0

        # 热激活近似常数（仅用于构造相对特征，不追求严格物理定量）
        self.R = 8.314  # J/mol/K
        self.Q = 65000.0  # J/mol，铝合金析出扩散的经验量级

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def engineer_features(self, df):
        """构造带物理分段逻辑的动态特征"""
        X = pd.DataFrame(index=df.index)

        # 原始输入
        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        # 基础量
        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-6, None)

        # -----------------------------
        # 1. 基础主特征：保留直接工艺信息
        # -----------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # -----------------------------
        # 2. 常规非线性特征：捕捉峰值时效/过时效的弯曲关系
        # -----------------------------
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp'] = self._safe_divide(log_time, temp_k)

        # -----------------------------
        # 3. Arrhenius / 扩散动力学启发特征
        # 物理意义：组织演化受扩散控制，温度升高时有效处理强度显著提升
        # -----------------------------
        arrhenius_factor = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-6, None)))
        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = time * arrhenius_factor
        X['log_thermal_dose'] = self._safe_log(X['thermal_dose'].values)
        X['temp_log_thermal_dose'] = temp * X['log_thermal_dose'].values

        # Zener-Hollomon风格的近似反向热处理指标（仅作状态表征）
        # 温度越高、时间越长，材料更可能向过时效/软化推进
        X['overaging_index'] = log_time * np.exp((temp - self.mid_temp_threshold) / 25.0)

        # -----------------------------
        # 4. 动态分区：按温度划分机制区间
        # -----------------------------
        low_temp_mask = (temp < self.low_temp_threshold).astype(float)
        mid_temp_mask = ((temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)).astype(float)
        high_temp_mask = (temp >= self.mid_temp_threshold).astype(float)

        X['is_low_temp'] = low_temp_mask
        X['is_mid_temp'] = mid_temp_mask
        X['is_high_temp'] = high_temp_mask

        # 温度区间内的“距离阈值”特征，体现接近机制转变边界时的敏感性
        X['dist_to_low_temp_th'] = temp - self.low_temp_threshold
        X['dist_to_mid_temp_th'] = temp - self.mid_temp_threshold
        X['relu_above_low_temp'] = np.maximum(temp - self.low_temp_threshold, 0)
        X['relu_above_mid_temp'] = np.maximum(temp - self.mid_temp_threshold, 0)
        X['relu_below_low_temp'] = np.maximum(self.low_temp_threshold - temp, 0)

        # -----------------------------
        # 5. 动态分区：按时间划分早期/峰值附近/长时阶段
        # 物理意义：
        # - 短时：析出成核/初期强化
        # - 中时：可能接近峰值强化
        # - 长时：粗化、过时效风险上升
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
        # 6. 温-时耦合的机制状态特征
        # 动态逻辑：不同温区下时间作用强度不同
        # -----------------------------
        # 低温：扩散慢，时间作用弱但持续累积
        X['low_temp_time_accum'] = low_temp_mask * log_time

        # 中温：最可能出现析出强化主导，温度和时间耦合最敏感
        X['mid_temp_peak_drive'] = mid_temp_mask * temp * log_time

        # 高温：长时间更易粗化/软化，过时效敏感性更强
        X['high_temp_overaging_drive'] = high_temp_mask * temp * np.maximum(log_time - np.log1p(self.short_time_threshold), 0)

        # 高温短时 vs 低温长时的等效处理近似
        X['temp_time_equiv'] = self._safe_divide(temp * log_time, inv_temp_k * 1000.0 + 1e-6)

        # -----------------------------
        # 7. 条件逻辑：分段构造“强化驱动力”和“软化驱动力”
        # 这是核心动态推理，不是简单堆叠
        # -----------------------------
        # 强化驱动力：
        # - 中温、中等时间最有利
        # - 温度过低扩散不足，过高则强化可能很快达到并转入衰减
        precipitation_drive = np.zeros_like(temp, dtype=float)

        # 低温区：强化随时间缓慢增长
        low_idx = temp < self.low_temp_threshold
        precipitation_drive[low_idx] = (
            0.6 * log_time[low_idx]
            + 0.004 * (temp[low_idx] - 60.0) * log_time[low_idx]
        )

        # 中温区：强化最强，近似峰值区
        mid_idx = (temp >= self.low_temp_threshold) & (temp < self.mid_temp_threshold)
        precipitation_drive[mid_idx] = (
            1.2 * log_time[mid_idx]
            + 0.012 * (temp[mid_idx] - self.low_temp_threshold) * log_time[mid_idx]
            - 0.06 * np.maximum(time[mid_idx] - self.long_time_threshold, 0)
        )

        # 高温区：短时可快速强化，但长时会迅速转入过时效
        high_idx = temp >= self.mid_temp_threshold
        precipitation_drive[high_idx] = (
            1.0 * np.minimum(log_time[high_idx], np.log1p(self.long_time_threshold))
            + 0.01 * (self.mid_temp_threshold - self.low_temp_threshold)
            - 0.10 * np.maximum(log_time[high_idx] - np.log1p(self.short_time_threshold), 0)
        )

        X['precipitation_drive'] = precipitation_drive

        # 软化/过时效驱动力：
        # 高温和长时间共同推动粗化、回复和强度回落
        softening_drive = (
            np.maximum(temp - self.mid_temp_threshold, 0) * 0.03 * log_time
            + high_temp_mask * np.maximum(time - self.short_time_threshold, 0) * 0.08
            + long_time_mask * np.maximum(temp - self.low_temp_threshold, 0) * 0.015
        )
        X['softening_drive'] = softening_drive

        # 净强化指数：反映强化与软化竞争
        X['net_strengthening_index'] = X['precipitation_drive'].values - X['softening_drive'].values

        # -----------------------------
        # 8. 峰值时效邻近度特征
        # 物理意义：强度常在某一“温-时组合”附近达到峰值
        # 这里只做启发式近似，不硬编码唯一峰值点
        # -----------------------------
        # 定义一个经验“最佳窗口”中心
        opt_temp_center = 130.0
        opt_log_time_center = np.log1p(8.0)

        X['temp_peak_proximity'] = -((temp - opt_temp_center) / 20.0) ** 2
        X['time_peak_proximity'] = -((log_time - opt_log_time_center) / 0.8) ** 2
        X['joint_peak_proximity'] = X['temp_peak_proximity'].values + X['time_peak_proximity'].values

        # 接近最佳区间时给出软门控
        peak_window = np.exp(X['joint_peak_proximity'].values)
        X['peak_window'] = peak_window
        X['peak_window_strength_drive'] = peak_window * X['precipitation_drive'].values
        X['peak_window_softening_balance'] = peak_window * X['net_strengthening_index'].values

        # -----------------------------
        # 9. 塑性相关启发特征
        # 物理意义：强度和塑性存在权衡，过时效可能使塑性回升
        # -----------------------------
        X['ductility_recovery_index'] = (
            0.5 * X['softening_drive'].values
            + 0.3 * high_temp_mask
            + 0.2 * long_time_mask
            - 0.3 * X['precipitation_drive'].values
        )

        X['strength_ductility_tradeoff'] = (
            X['net_strengthening_index'].values - X['ductility_recovery_index'].values
        )

        # -----------------------------
        # 10. 分段交互项：明确不同机制区间内的响应斜率
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
        # 11. 物理约束型截断特征
        # 避免极端外推时特征失真，增强小样本稳定性
        # -----------------------------
        clipped_time = np.clip(time, 0, 24)
        clipped_temp = np.clip(temp, 60, 200)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        # 饱和型特征：前期增长快、后期趋缓
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['temp_activation'] = 1.0 / (1.0 + np.exp(-(clipped_temp - self.low_temp_threshold) / 10.0))
        X['combined_activation'] = X['time_saturation'].values * X['temp_activation'].values

        # 高温长时惩罚：体现物理约束
        X['high_temp_long_time_penalty'] = (
            np.maximum(clipped_temp - self.mid_temp_threshold, 0)
            * np.maximum(clipped_time - self.short_time_threshold, 0)
        )

        # -----------------------------
        # 12. 相对原始态的工艺潜力指标（不使用标签，仅利用已知基准常数）
        # 物理意义：衡量热处理相对凝固态可能带来的组织改变量
        # -----------------------------
        base_strength_sum = self.base_tensile + self.base_yield
        base_strength_ratio = self._safe_divide(self.base_yield, self.base_tensile)

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['process_to_base_potential'] = (
            X['combined_activation'].values
            + 0.5 * X['net_strengthening_index'].values
            - 0.3 * X['high_temp_long_time_penalty'].values / 100.0
        )

        # -----------------------------
        # 13. 最终整理：清理数值
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