import pandas as pd
import numpy as np


# =========================================================
# Domain knowledge for 7499 Al alloy solution-treatment-like regime
# =========================================================
# 说明：
# 1) 当前数据温度范围是 420–480°C，而不是低温时效区；
# 2) 根据题目给出的统计趋势，在该范围内：
#    - 更高温度通常对应更高强度
#    - 更长时间通常也有利于强度提升，但存在温-时交互和局部非单调
# 3) 因此这里不能把“高温长时=一定软化”当作主先验，
#    只能把它作为“局部偏离/转折风险”特征。
DOMAIN_PARAMS = {
    # 机制区间边界：依据题目给定数据范围 420–480°C 构造
    # 420-440：较低固溶充分度区
    # 450-465：中间过渡区
    # 470+：高固溶充分度区
    'critical_temp_regime_shift': 450.0,
    'critical_temp_high_strength': 470.0,
    'critical_temp_upper': 480.0,

    # 时间机制边界
    # 1h: 短时，12h: 中长时拐点，24h: 长时上限
    'critical_time_short': 1.0,
    'critical_time_mid': 12.0,
    'critical_time_long': 24.0,

    # 动力学参数：铝合金中溶质扩散/组织演化的经验量级
    'activation_energy_Q': 135000.0,   # J/mol, 经验级别，用于相对刻画
    'gas_constant_R': 8.314,

    # LMP 常数
    'larson_miller_C': 20.0,

    # 基准原始样品性能
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # 启发式“最优响应窗口”中心
    # 来自题目领域知识：460-480°C、12-24h 常出现较高强度，且部分点塑性也高
    'opt_temp_center': 470.0,
    'opt_time_center': 12.0,

    # 机制突变相关的温-时组合阈值
    'high_temp_long_time_temp': 470.0,
    'high_temp_long_time_time': 12.0,
}


class FeatureAgent:
    """基于材料热处理机理的特征工程"""

    def __init__(self):
        self.feature_names = []
        self.params = DOMAIN_PARAMS

        self.base_strain = self.params['base_strain']
        self.base_tensile = self.params['base_tensile']
        self.base_yield = self.params['base_yield']

    def _safe_log(self, x):
        return np.log(np.clip(x, 1e-12, None))

    def _safe_log1p(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def physics_baseline(self, temp, time):
        """
        基于领域知识的基线预测：
        - 与题目统计趋势保持一致：
          更高温度通常带来更高强度；
          更长时间通常也提升强度，但后期趋缓并伴随局部交互。
        - 不依赖训练标签逐点拟合，只提供物理先验近似。
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = self._safe_log1p(time)

        # 温度驱动：420→480°C内总体增强
        temp_norm = (temp - 420.0) / 60.0
        temp_norm = np.clip(temp_norm, 0.0, 1.2)

        # 时间驱动：1h到12h变化较明显，之后趋缓
        time_norm = log_time / np.log1p(24.0)
        time_norm = np.clip(time_norm, 0.0, 1.2)

        # 温-时交互：高温会放大时间作用
        interaction = temp_norm * time_norm

        # 中高温窗口提升：反映 460-480°C 整体更优
        high_temp_activation = self._sigmoid((temp - 455.0) / 8.0)

        # 高温长时局部转折风险，不作为主趋势，只作为微调
        # 避免与题目统计方向冲突，因此仅给很弱权重
        local_turn_risk = self._sigmoid((temp - 470.0) / 4.0) * self._sigmoid((time - 18.0) / 3.0)

        # tensile baseline
        baseline_tensile = (
            self.base_tensile
            + 55.0 * temp_norm
            + 28.0 * time_norm
            + 35.0 * interaction
            + 10.0 * high_temp_activation
            - 6.0 * local_turn_risk
        )

        # yield baseline
        baseline_yield = (
            self.base_yield
            + 42.0 * temp_norm
            + 22.0 * time_norm
            + 28.0 * interaction
            + 8.0 * high_temp_activation
            - 5.0 * local_turn_risk
        )

        # strain baseline
        # 题目提示当前数据中应变并非与强度简单负相关，
        # 中高温/适中到较长时间下可同步提升。
        baseline_strain = (
            self.base_strain
            + 0.9 * temp_norm
            + 0.8 * time_norm
            + 0.9 * interaction
            + 0.5 * high_temp_activation
            - 0.35 * local_turn_risk
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        p = self.params
        R = p['gas_constant_R']
        Q = p['activation_energy_Q']

        temp_k = temp + 273.15
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-12, None)
        log_time = self._safe_log1p(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # -------------------------------------------------
        # 1. 原始基础特征
        # -------------------------------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['inv_temp_k'] = inv_temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time

        # -------------------------------------------------
        # 2. 基础非线性与交互
        # -------------------------------------------------
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # -------------------------------------------------
        # 3. 物理机制区间检测特征（关键）
        # -------------------------------------------------
        T_shift = p['critical_temp_regime_shift']
        T_high = p['critical_temp_high_strength']
        T_upper = p['critical_temp_upper']
        t_short = p['critical_time_short']
        t_mid = p['critical_time_mid']
        t_long = p['critical_time_long']

        X['is_low_temp_regime'] = (temp < T_shift).astype(float)
        X['is_transition_temp_regime'] = ((temp >= T_shift) & (temp < T_high)).astype(float)
        X['is_high_temp_regime'] = (temp >= T_high).astype(float)
        X['is_upper_temp_edge'] = (temp >= T_upper).astype(float)

        X['is_short_time_regime'] = (time <= t_short).astype(float)
        X['is_mid_time_regime'] = ((time > t_short) & (time < t_mid)).astype(float)
        X['is_long_time_regime'] = (time >= t_mid).astype(float)
        X['is_very_long_time_regime'] = (time >= t_long).astype(float)

        # 显式机制突变布尔特征
        X['is_high_temp_long_time'] = ((temp >= p['high_temp_long_time_temp']) & (time >= p['high_temp_long_time_time'])).astype(float)
        X['is_low_temp_long_time'] = ((temp < T_shift) & (time >= t_mid)).astype(float)
        X['is_high_temp_short_time'] = ((temp >= T_high) & (time <= t_short)).astype(float)
        X['is_transition_window'] = ((temp >= 460.0) & (temp <= 480.0) & (time >= 8.0) & (time <= 24.0)).astype(float)

        # 相对边界距离
        X['temp_relative_to_shift'] = temp - T_shift
        X['temp_relative_to_high'] = temp - T_high
        X['temp_relative_to_upper'] = temp - T_upper
        X['time_relative_to_short'] = time - t_short
        X['time_relative_to_mid'] = time - t_mid
        X['time_relative_to_long'] = time - t_long

        # ReLU 型分段激活
        X['relu_above_temp_shift'] = np.maximum(temp - T_shift, 0.0)
        X['relu_above_temp_high'] = np.maximum(temp - T_high, 0.0)
        X['relu_below_temp_shift'] = np.maximum(T_shift - temp, 0.0)
        X['relu_above_time_mid'] = np.maximum(time - t_mid, 0.0)
        X['relu_above_time_long'] = np.maximum(time - t_long, 0.0)
        X['relu_below_time_mid'] = np.maximum(t_mid - time, 0.0)

        # -------------------------------------------------
        # 4. 动力学等效特征
        # -------------------------------------------------
        # Arrhenius项：反映热激活过程
        arrhenius = np.exp(-Q / (R * np.clip(temp_k, 1e-12, None)))
        X['arrhenius_factor'] = arrhenius
        X['thermal_dose_arrhenius'] = time * arrhenius
        X['log_thermal_dose_arrhenius'] = self._safe_log1p(X['thermal_dose_arrhenius'].values)

        # Larson-Miller parameter
        # 常见形式 LMP = T(K) * (C + log10(t))
        log10_time = np.log10(np.clip(time, 1e-6, None))
        lmp = temp_k * (p['larson_miller_C'] + log10_time)
        X['larson_miller_parameter'] = lmp

        # Hollomon/JMAK风格简化特征
        X['temp_log_time_kinetic'] = temp_k * log_time
        X['log_time_times_inv_temp'] = log_time * inv_temp_k
        X['time_times_inv_temp'] = time * inv_temp_k

        # 等效处理程度：强调“高温会放大时间作用”
        X['equivalent_process_severity'] = (temp - 400.0) * log_time
        X['equivalent_process_severity_k'] = temp_k * log_time

        # -------------------------------------------------
        # 5. 物理基线预测值作为特征（让模型学残差）
        # -------------------------------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        # 基线派生
        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield

        # 相对原始态增益基线
        X['baseline_delta_strain_from_cast'] = baseline_strain - self.base_strain
        X['baseline_delta_tensile_from_cast'] = baseline_tensile - self.base_tensile
        X['baseline_delta_yield_from_cast'] = baseline_yield - self.base_yield

        # -------------------------------------------------
        # 6. 相对基线/相对边界的残差型输入特征
        # 注意：这里的“残差型”是相对物理基线的输入映射，不涉及真实标签
        # -------------------------------------------------
        X['temp_minus_opt_temp'] = temp - p['opt_temp_center']
        X['time_minus_opt_time'] = time - p['opt_time_center']
        X['log_time_minus_opt_log_time'] = log_time - np.log1p(p['opt_time_center'])

        # 距离最优窗口的邻近度
        X['opt_temp_proximity'] = -((temp - p['opt_temp_center']) / 12.0) ** 2
        X['opt_time_proximity'] = -((log_time - np.log1p(p['opt_time_center'])) / 0.7) ** 2
        X['joint_opt_proximity'] = X['opt_temp_proximity'].values + X['opt_time_proximity'].values
        X['opt_window_activation'] = np.exp(X['joint_opt_proximity'].values)

        # 对物理基线进行门控修正的辅助特征
        X['baseline_tensile_x_opt_window'] = baseline_tensile * X['opt_window_activation'].values
        X['baseline_yield_x_opt_window'] = baseline_yield * X['opt_window_activation'].values
        X['baseline_strain_x_opt_window'] = baseline_strain * X['opt_window_activation'].values

        # -------------------------------------------------
        # 7. 机制驱动力特征
        # 方向与数据一致：主趋势是温度/时间提升 -> 性能提升
        # 同时允许局部非单调由模型进一步学习
        # -------------------------------------------------
        temp_norm = np.clip((temp - 420.0) / 60.0, 0.0, 1.5)
        time_norm = np.clip(log_time / np.log1p(24.0), 0.0, 1.5)

        X['solution_progress_index'] = temp_norm + 0.8 * time_norm + 1.2 * temp_norm * time_norm
        X['homogenization_index'] = 0.7 * temp_norm + 1.0 * time_norm
        X['strengthening_potential_index'] = (
            1.0 * temp_norm
            + 0.9 * time_norm
            + 1.4 * temp_norm * time_norm
            + 0.4 * X['opt_window_activation'].values
        )

        # 局部转折/偏离风险：只作次级特征，不作为主方向
        X['local_overprocess_risk'] = (
            self._sigmoid((temp - 470.0) / 4.0) *
            self._sigmoid((time - 18.0) / 3.0)
        )
        X['edge_instability_risk'] = (
            self._sigmoid((temp - 478.0) / 2.0) *
            self._sigmoid((time - 20.0) / 2.0)
        )

        X['net_process_gain_index'] = (
            X['strengthening_potential_index'].values
            - 0.35 * X['local_overprocess_risk'].values
            - 0.25 * X['edge_instability_risk'].values
        )

        # -------------------------------------------------
        # 8. 分区交互：帮助树模型/线性模型学习不同区间斜率
        # -------------------------------------------------
        X['temp_in_low_regime'] = temp * X['is_low_temp_regime'].values
        X['temp_in_transition_regime'] = temp * X['is_transition_temp_regime'].values
        X['temp_in_high_regime'] = temp * X['is_high_temp_regime'].values

        X['log_time_in_low_regime'] = log_time * X['is_low_temp_regime'].values
        X['log_time_in_transition_regime'] = log_time * X['is_transition_temp_regime'].values
        X['log_time_in_high_regime'] = log_time * X['is_high_temp_regime'].values

        X['temp_log_time_low_regime'] = temp * log_time * X['is_low_temp_regime'].values
        X['temp_log_time_transition_regime'] = temp * log_time * X['is_transition_temp_regime'].values
        X['temp_log_time_high_regime'] = temp * log_time * X['is_high_temp_regime'].values

        X['arrhenius_in_high_temp'] = arrhenius * X['is_high_temp_regime'].values
        X['lmp_in_long_time'] = lmp * X['is_long_time_regime'].values

        # -------------------------------------------------
        # 9. 强塑协同相关启发特征
        # 题目明确指出：该数据中不应先验假设强度与塑性必然负相关
        # -------------------------------------------------
        X['strength_ductility_synergy_index'] = (
            0.6 * X['solution_progress_index'].values
            + 0.8 * X['opt_window_activation'].values
            - 0.3 * X['edge_instability_risk'].values
        )

        X['ductility_recovery_like_index'] = (
            0.5 * X['is_high_temp_regime'].values
            + 0.5 * X['is_long_time_regime'].values
            + 0.4 * X['opt_window_activation'].values
            - 0.4 * X['edge_instability_risk'].values
        )

        # -------------------------------------------------
        # 10. 稳健截断与归一化风格特征
        # -------------------------------------------------
        clipped_temp = np.clip(temp, 420.0, 480.0)
        clipped_time = np.clip(time, 1.0, 24.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        X['temp_scaled_420_480'] = (clipped_temp - 420.0) / 60.0
        X['time_scaled_log'] = np.log1p(clipped_time) / np.log1p(24.0)

        # -------------------------------------------------
        # 11. 常数基准特征
        # -------------------------------------------------
        X['base_cast_strain'] = self.base_strain
        X['base_cast_tensile'] = self.base_tensile
        X['base_cast_yield'] = self.base_yield
        X['base_cast_yield_tensile_ratio'] = self._safe_divide(self.base_yield, self.base_tensile)

        # -------------------------------------------------
        # 12. 数值清理
        # -------------------------------------------------
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