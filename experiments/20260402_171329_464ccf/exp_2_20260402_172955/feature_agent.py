import pandas as pd
import numpy as np


# 基于7499铝合金在420–480°C、1–24h附近固溶处理区间的领域先验
# 关键原则：
# 1) 在当前数据范围内，温度升高通常对应更高强度；
# 2) 时间延长多数情况下也有利于强度提升，但存在温度-时间交互与局部非单调；
# 3) 因为样本很少，优先使用低维、稳健、带物理意义的分段与动力学特征；
# 4) 机制边界不设得过“激进”，更多用于帮助模型识别转折区，而不是硬编码绝对规律。
DOMAIN_PARAMS = {
    # 温度机制边界（单位：°C）
    # 结合题述可知420–480°C整体更像固溶处理窗口，因此边界主要表示“处理充分程度”的转折
    'critical_temp_regime_shift': 450.0,      # 从“较弱/中等处理”向“较充分处理”过渡
    'critical_temp_high': 470.0,              # 高温处理区，时间效应更敏感
    'critical_temp_top': 480.0,               # 数据上界附近，高温窗口边界

    # 时间机制边界（单位：h）
    'critical_time_short': 3.0,               # 早期阶段
    'critical_time_mid': 8.0,                 # 中等保温，常见明显提升区
    'critical_time_long': 12.0,               # 长时边界
    'critical_time_very_long': 24.0,          # 数据上界附近

    # 动力学参数
    # 铝合金扩散/固溶相关的经验量级，作为特征构造用，不追求严格定量
    'activation_energy_Q': 115000.0,          # J/mol
    'gas_constant_R': 8.314,                  # J/(mol·K)

    # Larson-Miller风格常数
    'lmp_C': 20.0,

    # 参考温度（用于等效热处理量）
    'reference_temp_c': 420.0,
    'reference_temp_k': 420.0 + 273.15,

    # 机制窗口中心（启发式）
    'solution_center_temp': 460.0,
    'solution_center_time': 12.0,

    # 原始铸态基准性能
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,
}


class FeatureAgent:
    """基于材料热处理机理的特征工程"""

    def __init__(self):
        self.feature_names = []

        self.base_strain = DOMAIN_PARAMS['base_strain']
        self.base_tensile = DOMAIN_PARAMS['base_tensile']
        self.base_yield = DOMAIN_PARAMS['base_yield']

        self.Q = DOMAIN_PARAMS['activation_energy_Q']
        self.R = DOMAIN_PARAMS['gas_constant_R']
        self.lmp_C = DOMAIN_PARAMS['lmp_C']

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
        基于世界知识的基线估计（不依赖训练标签逐点拟合）：
        - 当前区间内，升温通常增强固溶充分程度，强度整体升高；
        - 延时通常也提升处理充分度，但边际效应递减，且高温长时可能出现局部非单调；
        - 应变在本数据中并非与强度必然负相关，中高温/适中-较长时间下可同步改善。
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))

        # 归一化处理程度
        temp_progress = np.clip((temp - 420.0) / 60.0, 0.0, 1.2)     # 420 -> 480
        time_progress = np.clip(log_time / np.log1p(24.0), 0.0, 1.2)  # 1 -> 24 的对数尺度

        # 主导：当前数据下强度总体随温度上升而上升；时间为次主导但有交互
        process_extent = (
            0.55 * temp_progress +
            0.25 * time_progress +
            0.20 * temp_progress * time_progress
        )

        # 高温长时局部“转折/饱和”惩罚：只做轻微修正，不违背整体升高趋势
        high_temp_gate = self._sigmoid((temp - DOMAIN_PARAMS['critical_temp_high']) / 4.0)
        long_time_gate = self._sigmoid((time - DOMAIN_PARAMS['critical_time_long']) / 2.0)
        saturation_penalty = 0.10 * high_temp_gate * long_time_gate

        effective_extent = process_extent - saturation_penalty

        # 强度基线：相对铸态明显提升
        baseline_tensile = self.base_tensile + 105.0 * effective_extent
        baseline_yield = self.base_yield + 88.0 * effective_extent

        # 应变：中高温和适中/较长时间下可同步改善；极端高温长时只做轻微抑制
        ductility_gain = (
            1.2 * temp_progress +
            1.0 * time_progress +
            0.8 * temp_progress * time_progress
        )
        ductility_penalty = 0.45 * high_temp_gate * long_time_gate
        baseline_strain = self.base_strain + ductility_gain - ductility_penalty

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-6, None)

        T1 = DOMAIN_PARAMS['critical_temp_regime_shift']
        T2 = DOMAIN_PARAMS['critical_temp_high']
        T3 = DOMAIN_PARAMS['critical_temp_top']
        t1 = DOMAIN_PARAMS['critical_time_short']
        t2 = DOMAIN_PARAMS['critical_time_mid']
        t3 = DOMAIN_PARAMS['critical_time_long']
        t4 = DOMAIN_PARAMS['critical_time_very_long']
        Tref_c = DOMAIN_PARAMS['reference_temp_c']
        Tref_k = DOMAIN_PARAMS['reference_temp_k']
        Tc = DOMAIN_PARAMS['solution_center_temp']
        tc = DOMAIN_PARAMS['solution_center_time']

        # -----------------------------
        # 1. 原始与基础变换
        # -----------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2

        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # -----------------------------
        # 2. 动力学等效特征
        # -----------------------------
        # Arrhenius 因子：反映热激活程度
        arrhenius = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-6, None)))
        X['arrhenius_factor'] = arrhenius

        # 等效热处理量：时间 * Arrhenius
        thermal_dose = time * arrhenius
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = self._safe_log(thermal_dose)

        # 相对参考温度的“额外热驱动”
        delta_invT = (1.0 / np.clip(Tref_k, 1e-6, None)) - inv_temp_k
        equivalent_rate = np.exp(np.clip((self.Q / self.R) * delta_invT, -50, 50))
        X['equivalent_rate_vs_420C'] = equivalent_rate
        X['equivalent_time_vs_420C'] = time * equivalent_rate
        X['log_equivalent_time_vs_420C'] = self._safe_log(X['equivalent_time_vs_420C'].values)

        # Larson-Miller Parameter 风格
        # 传统常用于蠕变，这里仅作为“温-时合成尺度”帮助模型学习工艺累计作用
        lmp = temp_k * (self.lmp_C + np.log10(np.clip(time, 1e-6, None)))
        X['larson_miller'] = lmp

        # Hollomon/JMAK风格简化指标
        X['kinetic_index'] = self._safe_divide(log_time, temp_k)
        X['temp_centered_log_time'] = (temp - Tref_c) * log_time
        X['arrhenius_log_time'] = arrhenius * log_time

        # -----------------------------
        # 3. 物理机制区间检测特征（关键）
        # -----------------------------
        # 温度区间
        is_low_temp = (temp < T1).astype(float)
        is_mid_temp = ((temp >= T1) & (temp < T2)).astype(float)
        is_high_temp = (temp >= T2).astype(float)
        is_top_temp = (temp >= T3).astype(float)

        X['is_low_temp_regime'] = is_low_temp
        X['is_mid_temp_regime'] = is_mid_temp
        X['is_high_temp_regime'] = is_high_temp
        X['is_top_temp_regime'] = is_top_temp

        # 时间区间
        is_short_time = (time < t1).astype(float)
        is_mid_time = ((time >= t1) & (time < t2)).astype(float)
        is_long_time = ((time >= t2) & (time < t3)).astype(float)
        is_very_long_time = (time >= t3).astype(float)
        is_top_time = (time >= t4).astype(float)

        X['is_short_time_regime'] = is_short_time
        X['is_mid_time_regime'] = is_mid_time
        X['is_long_time_regime'] = is_long_time
        X['is_very_long_time_regime'] = is_very_long_time
        X['is_top_time_regime'] = is_top_time

        # 关键联合机制区
        X['is_high_temp_long_time'] = ((temp >= T2) & (time >= t3)).astype(float)
        X['is_high_temp_mid_time'] = ((temp >= T2) & (time >= t2) & (time < t3)).astype(float)
        X['is_mid_temp_long_time'] = ((temp >= T1) & (temp < T2) & (time >= t3)).astype(float)
        X['is_low_temp_very_long_time'] = ((temp < T1) & (time >= t3)).astype(float)
        X['is_near_regime_shift'] = (np.abs(temp - T1) <= 5.0).astype(float)
        X['is_near_high_temp_shift'] = (np.abs(temp - T2) <= 5.0).astype(float)

        # -----------------------------
        # 4. 相对机制边界的连续特征
        # -----------------------------
        X['temp_relative_to_T1'] = temp - T1
        X['temp_relative_to_T2'] = temp - T2
        X['temp_relative_to_T3'] = temp - T3

        X['time_relative_to_t1'] = time - t1
        X['time_relative_to_t2'] = time - t2
        X['time_relative_to_t3'] = time - t3
        X['time_relative_to_t4'] = time - t4

        X['relu_above_T1'] = np.maximum(temp - T1, 0)
        X['relu_above_T2'] = np.maximum(temp - T2, 0)
        X['relu_above_T3'] = np.maximum(temp - T3, 0)
        X['relu_below_T1'] = np.maximum(T1 - temp, 0)

        X['relu_above_t1'] = np.maximum(time - t1, 0)
        X['relu_above_t2'] = np.maximum(time - t2, 0)
        X['relu_above_t3'] = np.maximum(time - t3, 0)
        X['relu_above_t4'] = np.maximum(time - t4, 0)
        X['relu_below_t1'] = np.maximum(t1 - time, 0)

        # 软门控，捕捉边界附近平滑转折
        X['temp_gate_T1'] = self._sigmoid((temp - T1) / 4.0)
        X['temp_gate_T2'] = self._sigmoid((temp - T2) / 3.0)
        X['time_gate_t2'] = self._sigmoid((time - t2) / 1.5)
        X['time_gate_t3'] = self._sigmoid((time - t3) / 1.5)

        # -----------------------------
        # 5. 机制驱动力特征
        # -----------------------------
        # 处理充分度：与数据趋势一致，温度升高和延时整体正向
        treatment_extent = (
            0.55 * np.clip((temp - 420.0) / 60.0, 0, 1.2) +
            0.25 * np.clip(log_time / np.log1p(24.0), 0, 1.2) +
            0.20 * np.clip((temp - 420.0) / 60.0, 0, 1.2) * np.clip(log_time / np.log1p(24.0), 0, 1.2)
        )
        X['treatment_extent'] = treatment_extent

        # 高温长时下可能出现局部非线性/饱和，仅作轻惩罚
        saturation_risk = self._sigmoid((temp - T2) / 4.0) * self._sigmoid((time - t3) / 2.0)
        X['saturation_risk'] = saturation_risk

        # 净强化潜力
        X['net_solution_strengthening'] = treatment_extent - 0.12 * saturation_risk

        # 塑性恢复/均匀化倾向：中高温与较长时间更可能改善塑性
        ductility_support = (
            0.35 * self._sigmoid((temp - T1) / 6.0) +
            0.30 * self._sigmoid((time - t2) / 2.0) +
            0.20 * self._sigmoid((temp - T2) / 4.0) * self._sigmoid((time - t2) / 2.0)
        )
        X['ductility_support'] = ductility_support

        # 强塑协同指数
        X['strength_ductility_synergy'] = X['net_solution_strengthening'].values + 0.5 * X['ductility_support'].values

        # -----------------------------
        # 6. 分区交互特征
        # -----------------------------
        X['temp_in_low_regime'] = temp * is_low_temp
        X['temp_in_mid_regime'] = temp * is_mid_temp
        X['temp_in_high_regime'] = temp * is_high_temp

        X['log_time_in_low_regime'] = log_time * is_low_temp
        X['log_time_in_mid_regime'] = log_time * is_mid_temp
        X['log_time_in_high_regime'] = log_time * is_high_temp

        X['temp_log_time_low_regime'] = temp * log_time * is_low_temp
        X['temp_log_time_mid_regime'] = temp * log_time * is_mid_temp
        X['temp_log_time_high_regime'] = temp * log_time * is_high_temp

        X['high_temp_long_time_interaction'] = temp * time * X['is_high_temp_long_time'].values
        X['mid_temp_long_time_interaction'] = temp * time * X['is_mid_temp_long_time'].values

        # -----------------------------
        # 7. 以“窗口中心”为参照的邻近度特征
        # -----------------------------
        X['dist_to_solution_center_temp'] = temp - Tc
        X['dist_to_solution_center_time'] = log_time - np.log1p(tc)

        X['solution_center_temp_proximity'] = -((temp - Tc) / 10.0) ** 2
        X['solution_center_time_proximity'] = -((log_time - np.log1p(tc)) / 0.7) ** 2
        X['joint_solution_window_proximity'] = (
            X['solution_center_temp_proximity'].values +
            X['solution_center_time_proximity'].values
        )
        X['solution_window_weight'] = np.exp(X['joint_solution_window_proximity'].values)

        # -----------------------------
        # 8. physics baseline 及残差学习特征
        # -----------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)

        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        # 基线所代表的物理增益
        X['baseline_delta_strain'] = baseline_strain - self.base_strain
        X['baseline_delta_tensile'] = baseline_tensile - self.base_tensile
        X['baseline_delta_yield'] = baseline_yield - self.base_yield

        # 基线比值/差值，有助于模型学习多目标间关系
        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield

        # 相对基线的工艺位置特征（让模型学残差）
        X['temp_minus_baseline_ref'] = temp - Tref_c
        X['time_log_minus_baseline_ref'] = log_time - np.log1p(1.0)
        X['treatment_minus_baseline_extent'] = X['treatment_extent'].values  # 基准铸态处理程度近似为0

        # 基线与机制边界的耦合
        X['baseline_tensile_x_high_temp_gate'] = baseline_tensile * X['temp_gate_T2'].values
        X['baseline_yield_x_long_time_gate'] = baseline_yield * X['time_gate_t3'].values
        X['baseline_strain_x_solution_window'] = baseline_strain * X['solution_window_weight'].values

        # -----------------------------
        # 9. 稳健截断与饱和特征
        # -----------------------------
        clipped_temp = np.clip(temp, 420.0, 480.0)
        clipped_time = np.clip(time, 1.0, 24.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 8.0)
        X['temp_activation'] = self._sigmoid((clipped_temp - T1) / 6.0)
        X['combined_activation'] = X['time_saturation'].values * X['temp_activation'].values

        # -----------------------------
        # 10. 基于原始铸态的潜力指标
        # -----------------------------
        base_strength_sum = self.base_tensile + self.base_yield
        X['base_strength_sum'] = base_strength_sum
        X['base_yield_tensile_ratio'] = self.base_yield / self.base_tensile

        X['process_to_base_potential'] = (
            0.4 * X['combined_activation'].values +
            0.4 * X['net_solution_strengthening'].values +
            0.2 * X['ductility_support'].values
        )

        # -----------------------------
        # 11. 数值清理
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