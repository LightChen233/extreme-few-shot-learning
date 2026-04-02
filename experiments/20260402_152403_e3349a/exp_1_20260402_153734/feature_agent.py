import pandas as pd
import numpy as np


# 基于7499高强铝合金热处理/时效的一般材料规律定义
DOMAIN_PARAMS = {
    # 温度机制边界（℃）
    # 低于该范围，扩散与析出动力学较慢，强化不足
    'low_temp_threshold': 110.0,

    # 中温区通常更接近有效时效强化区间
    'mid_temp_threshold': 150.0,

    # 较高温区，组织粗化/过时效风险明显增强
    'high_temp_threshold': 170.0,

    # 时间机制边界（h）
    # 短时：析出初期/强化尚未充分
    'short_time_threshold': 4.0,

    # 中等时间：更可能接近峰值时效
    'mid_time_threshold': 8.0,

    # 长时：过时效风险上升
    'long_time_threshold': 12.0,

    # 经验峰值窗口中心（启发式，不依赖标签）
    'peak_temp_center': 130.0,
    'peak_time_center': 8.0,

    # 峰值窗口尺度
    'peak_temp_scale': 18.0,
    'peak_log_time_scale': 0.75,

    # 热激活参数
    # 铝合金析出/扩散控制过程常用Arrhenius型描述，Q取经验量级
    'R': 8.314,          # J/(mol·K)
    'Q': 65000.0,        # J/mol

    # 基准性能：原始凝固态样品
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # 热处理后性能增益的启发式幅值
    # 这里只给物理可解释的近似上限，不依赖训练标签拟合
    'max_tensile_gain': 260.0,
    'max_yield_gain': 230.0,
    'max_strain_drop': 2.2,
    'max_strain_recovery': 2.8,
}


class FeatureAgent:
    """基于材料热处理机理的动态特征工程"""

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
        基于材料世界知识的粗粒度物理基线：
        - 中温+中等时间：析出强化最显著
        - 高温+长时：过时效/粗化导致强度回落、塑性恢复
        - 低温/短时：强化不足，接近原始态
        返回:
            baseline_strain, baseline_tensile, baseline_yield
        """
        p = self.params

        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # Arrhenius扩散激活：温度越高，组织演化速度越快
        arrhenius = np.exp(-p['Q'] / (p['R'] * np.clip(temp_k, 1e-9, None)))

        # 归一化的热剂量（对数压缩后增强小样本稳定性）
        thermal_dose = time * arrhenius
        log_thermal_dose = np.log1p(np.clip(thermal_dose * 1e10, 0, None))

        # 温度激活：低温不足，中温进入强化窗口
        temp_activation = self._sigmoid((temp - p['low_temp_threshold']) / 10.0)

        # 时间激活：随时间先增强
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / 4.5)

        # 峰值时效窗口：中温+中等时间最接近峰值
        temp_peak = np.exp(-((temp - p['peak_temp_center']) / p['peak_temp_scale']) ** 2)
        time_peak = np.exp(-((log_time - np.log1p(p['peak_time_center'])) / p['peak_log_time_scale']) ** 2)
        peak_window = temp_peak * time_peak

        # 强化驱动：低温短时小，中温中时大
        strengthening = (
            0.35 * temp_activation
            + 0.25 * time_activation
            + 0.40 * peak_window
        )

        # 过时效驱动：高温+长时显著
        high_temp_act = self._sigmoid((temp - p['mid_temp_threshold']) / 8.0)
        long_time_act = self._sigmoid((time - p['long_time_threshold']) / 2.5)
        overaging = high_temp_act * (0.55 * long_time_act + 0.45 * np.clip(log_time - np.log1p(p['short_time_threshold']), 0, None))

        # 净强化指数
        net_strength = np.clip(strengthening - 0.55 * overaging, 0, None)

        # 抗拉、屈服基线
        baseline_tensile = p['base_tensile'] + p['max_tensile_gain'] * net_strength
        baseline_yield = p['base_yield'] + p['max_yield_gain'] * (0.92 * strengthening - 0.60 * overaging)

        # 物理约束：屈服强度不应显著超过抗拉强度
        baseline_yield = np.minimum(baseline_yield, baseline_tensile - 8.0)

        # 应变：强化增强时下降，过时效/回复时有所恢复
        strain_drop = p['max_strain_drop'] * strengthening
        strain_recovery = p['max_strain_recovery'] * overaging
        baseline_strain = p['base_strain'] - strain_drop + strain_recovery

        # 合理截断
        baseline_tensile = np.clip(baseline_tensile, 80.0, 650.0)
        baseline_yield = np.clip(baseline_yield, 50.0, 600.0)
        baseline_strain = np.clip(baseline_strain, 1.0, 20.0)

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)
        p = self.params

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-9, None)

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
        # 2. 物理基线预测
        # -----------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        baseline_gap_ty = baseline_tensile - baseline_yield
        baseline_yield_ratio = self._safe_divide(baseline_yield, baseline_tensile)
        baseline_strength_sum = baseline_tensile + baseline_yield

        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield
        X['baseline_gap_ty'] = baseline_gap_ty
        X['baseline_yield_ratio'] = baseline_yield_ratio
        X['baseline_strength_sum'] = baseline_strength_sum

        # 相对原始凝固态的基线增益
        X['baseline_tensile_gain_from_base'] = baseline_tensile - p['base_tensile']
        X['baseline_yield_gain_from_base'] = baseline_yield - p['base_yield']
        X['baseline_strain_change_from_base'] = baseline_strain - p['base_strain']

        # -----------------------------
        # 3. 基础非线性与交互
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
        # 4. Arrhenius / 热剂量特征
        # -----------------------------
        arrhenius_factor = np.exp(-p['Q'] / (p['R'] * np.clip(temp_k, 1e-9, None)))
        thermal_dose = time * arrhenius_factor
        log_thermal_dose = np.log1p(np.clip(thermal_dose * 1e10, 0, None))

        X['arrhenius_factor'] = arrhenius_factor
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = log_thermal_dose
        X['temp_log_thermal_dose'] = temp * log_thermal_dose
        X['time_arrhenius_scaled'] = time * arrhenius_factor * 1e10

        # -----------------------------
        # 5. 机制区间检测特征
        # -----------------------------
        low_temp_mask = (temp < p['low_temp_threshold']).astype(float)
        mid_temp_mask = ((temp >= p['low_temp_threshold']) & (temp < p['mid_temp_threshold'])).astype(float)
        high_temp_mask = ((temp >= p['mid_temp_threshold']) & (temp < p['high_temp_threshold'])).astype(float)
        very_high_temp_mask = (temp >= p['high_temp_threshold']).astype(float)

        short_time_mask = (time < p['short_time_threshold']).astype(float)
        mid_time_mask = ((time >= p['short_time_threshold']) & (time < p['long_time_threshold'])).astype(float)
        long_time_mask = (time >= p['long_time_threshold']).astype(float)

        X['is_low_temp'] = low_temp_mask
        X['is_mid_temp'] = mid_temp_mask
        X['is_high_temp'] = high_temp_mask
        X['is_very_high_temp'] = very_high_temp_mask

        X['is_short_time'] = short_time_mask
        X['is_mid_time'] = mid_time_mask
        X['is_long_time'] = long_time_mask

        # -----------------------------
        # 6. 相对机制边界的残差/距离特征
        # -----------------------------
        X['temp_relative_to_low_boundary'] = temp - p['low_temp_threshold']
        X['temp_relative_to_mid_boundary'] = temp - p['mid_temp_threshold']
        X['temp_relative_to_high_boundary'] = temp - p['high_temp_threshold']

        X['time_relative_to_short_boundary'] = time - p['short_time_threshold']
        X['time_relative_to_mid_boundary'] = time - p['mid_time_threshold']
        X['time_relative_to_long_boundary'] = time - p['long_time_threshold']

        X['relu_above_low_temp'] = np.maximum(temp - p['low_temp_threshold'], 0)
        X['relu_above_mid_temp'] = np.maximum(temp - p['mid_temp_threshold'], 0)
        X['relu_above_high_temp'] = np.maximum(temp - p['high_temp_threshold'], 0)
        X['relu_below_low_temp'] = np.maximum(p['low_temp_threshold'] - temp, 0)

        X['relu_above_short_time'] = np.maximum(time - p['short_time_threshold'], 0)
        X['relu_above_mid_time'] = np.maximum(time - p['mid_time_threshold'], 0)
        X['relu_above_long_time'] = np.maximum(time - p['long_time_threshold'], 0)
        X['relu_below_short_time'] = np.maximum(p['short_time_threshold'] - time, 0)

        # 与峰值窗口中心的相对偏差
        X['temp_residual_to_peak_center'] = temp - p['peak_temp_center']
        X['log_time_residual_to_peak_center'] = log_time - np.log1p(p['peak_time_center'])
        X['abs_temp_residual_to_peak_center'] = np.abs(X['temp_residual_to_peak_center'].values)
        X['abs_log_time_residual_to_peak_center'] = np.abs(X['log_time_residual_to_peak_center'].values)

        # -----------------------------
        # 7. 峰值窗口与机制竞争特征
        # -----------------------------
        temp_peak_proximity = np.exp(-((temp - p['peak_temp_center']) / p['peak_temp_scale']) ** 2)
        time_peak_proximity = np.exp(-((log_time - np.log1p(p['peak_time_center'])) / p['peak_log_time_scale']) ** 2)
        joint_peak_window = temp_peak_proximity * time_peak_proximity

        X['temp_peak_proximity'] = temp_peak_proximity
        X['time_peak_proximity'] = time_peak_proximity
        X['joint_peak_window'] = joint_peak_window

        temp_activation = self._sigmoid((temp - p['low_temp_threshold']) / 10.0)
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / 4.5)
        high_temp_activation = self._sigmoid((temp - p['mid_temp_threshold']) / 8.0)
        very_high_temp_activation = self._sigmoid((temp - p['high_temp_threshold']) / 6.0)
        long_time_activation = self._sigmoid((time - p['long_time_threshold']) / 2.5)

        strengthening_drive = (
            0.35 * temp_activation
            + 0.25 * time_activation
            + 0.40 * joint_peak_window
        )
        softening_drive = (
            high_temp_activation * (0.50 * long_time_activation + 0.25 * np.maximum(log_time - np.log1p(p['short_time_threshold']), 0))
            + 0.35 * very_high_temp_activation * time_activation
        )
        net_strengthening = strengthening_drive - softening_drive

        X['temp_activation'] = temp_activation
        X['time_activation'] = time_activation
        X['high_temp_activation'] = high_temp_activation
        X['very_high_temp_activation'] = very_high_temp_activation
        X['long_time_activation'] = long_time_activation

        X['strengthening_drive'] = strengthening_drive
        X['softening_drive'] = softening_drive
        X['net_strengthening_index'] = net_strengthening

        # -----------------------------
        # 8. 分区耦合特征
        # -----------------------------
        X['low_temp_time_accum'] = low_temp_mask * log_time
        X['mid_temp_peak_drive'] = mid_temp_mask * temp * log_time
        X['high_temp_overaging_drive'] = (high_temp_mask + very_high_temp_mask) * temp * np.maximum(log_time - np.log1p(p['short_time_threshold']), 0)

        X['temp_in_low_zone'] = temp * low_temp_mask
        X['temp_in_mid_zone'] = temp * mid_temp_mask
        X['temp_in_high_zone'] = temp * high_temp_mask
        X['temp_in_very_high_zone'] = temp * very_high_temp_mask

        X['log_time_in_short_zone'] = log_time * short_time_mask
        X['log_time_in_mid_zone'] = log_time * mid_time_mask
        X['log_time_in_long_zone'] = log_time * long_time_mask

        X['temp_log_time_low_zone'] = temp * log_time * low_temp_mask
        X['temp_log_time_mid_zone'] = temp * log_time * mid_temp_mask
        X['temp_log_time_high_zone'] = temp * log_time * high_temp_mask
        X['temp_log_time_very_high_zone'] = temp * log_time * very_high_temp_mask

        # -----------------------------
        # 9. 基于基线的残差型工艺表征
        # 这里“残差”指输入条件相对物理基线状态的偏离，而非标签残差
        # -----------------------------
        X['tensile_gain_per_hour_baseline'] = self._safe_divide(
            baseline_tensile - p['base_tensile'], time + 1.0
        )
        X['yield_gain_per_hour_baseline'] = self._safe_divide(
            baseline_yield - p['base_yield'], time + 1.0
        )
        X['strain_change_per_hour_baseline'] = self._safe_divide(
            baseline_strain - p['base_strain'], time + 1.0
        )

        X['baseline_strength_per_temp'] = self._safe_divide(baseline_strength_sum, temp_k)
        X['baseline_gap_over_strength'] = self._safe_divide(baseline_gap_ty, baseline_strength_sum)

        # 基线与机制边界共同编码
        X['baseline_tensile_x_temp_mid_residual'] = baseline_tensile * (temp - p['mid_temp_threshold'])
        X['baseline_yield_x_time_long_residual'] = baseline_yield * (time - p['long_time_threshold'])
        X['baseline_strain_x_overaging'] = baseline_strain * softening_drive
        X['baseline_tensile_x_peak_window'] = baseline_tensile * joint_peak_window
        X['baseline_yield_x_peak_window'] = baseline_yield * joint_peak_window
        X['baseline_strength_x_net_strengthening'] = baseline_strength_sum * net_strengthening

        # -----------------------------
        # 10. 强塑权衡启发特征
        # -----------------------------
        ductility_recovery_index = (
            0.60 * softening_drive
            + 0.20 * long_time_mask
            + 0.20 * very_high_temp_mask
            - 0.35 * strengthening_drive
        )
        strength_ductility_tradeoff = net_strengthening - ductility_recovery_index

        X['ductility_recovery_index'] = ductility_recovery_index
        X['strength_ductility_tradeoff'] = strength_ductility_tradeoff

        # -----------------------------
        # 11. 稳健裁剪与饱和型特征
        # -----------------------------
        clipped_temp = np.clip(temp, 60, 220)
        clipped_time = np.clip(time, 0, 24)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['combined_activation'] = X['time_saturation'].values * temp_activation
        X['high_temp_long_time_penalty'] = np.maximum(clipped_temp - p['mid_temp_threshold'], 0) * np.maximum(clipped_time - p['short_time_threshold'], 0)
        X['very_high_temp_long_time_penalty'] = np.maximum(clipped_temp - p['high_temp_threshold'], 0) * np.maximum(clipped_time - p['mid_time_threshold'], 0)

        # -----------------------------
        # 12. 原始凝固态常数相关特征
        # -----------------------------
        base_strength_sum = p['base_tensile'] + p['base_yield']
        base_strength_ratio = p['base_yield'] / p['base_tensile']

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['process_to_base_potential'] = (
            X['combined_activation'].values
            + 0.50 * net_strengthening
            - 0.003 * X['high_temp_long_time_penalty'].values
        )

        # -----------------------------
        # 13. 多目标关联先验特征
        # -----------------------------
        X['predicted_yield_tensile_ratio_baseline'] = baseline_yield_ratio
        X['predicted_tensile_minus_yield_baseline'] = baseline_gap_ty
        X['predicted_strength_ductility_coupling'] = self._safe_divide(
            baseline_strength_sum, baseline_strain + 1e-6
        )

        # -----------------------------
        # 14. 数值清理
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