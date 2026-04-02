import pandas as pd
import numpy as np


# -----------------------------
# Domain knowledge for 7499 Al alloy solution treatment
# -----------------------------
# 任务背景：
# - 温度范围约 420–480°C，时间 1–24h
# - 更接近固溶处理/均匀化窗口，而不是低温人工时效
# - 数据总体趋势：升温、延时通常提升强度；但在 470°C/12h、440°C/24h、460°C/12h 等点
#   出现明显偏差，说明存在温时耦合、局部平台化/组织转折，而非简单线性关系
# - 训练样本仅 29 条，因此特征必须“少而精”，优先保留物理上最有解释性的特征
DOMAIN_PARAMS = {
    # 机制边界 / 工艺区间
    'critical_temp_regime_shift': 450.0,   # 低温固溶不足 -> 中温较充分固溶
    'secondary_temp_boundary': 465.0,      # 中温强化主导区，接近较优窗口中心
    'high_temp_risk_boundary': 470.0,      # 高温区，长时下可能出现平台/粗化/局部非单调
    'critical_time_regime_shift': 6.0,     # 短时 -> 中时，前期动力学主导转折
    'long_time_boundary': 12.0,            # 长时边界，误差集中区之一
    'very_long_time_boundary': 24.0,       # 数据上限附近

    # 动力学参数（启发式）
    'activation_energy_Q': 135000.0,       # J/mol
    'gas_constant_R': 8.314,               # J/mol/K
    'lmp_C': 20.0,                         # Larson-Miller 常数

    # 参考窗口（从数据和材料机理启发，不是硬编码最优）
    'reference_temp_center': 465.0,
    'reference_time_center': 12.0,

    # 基准原始态性能
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,
}


class FeatureAgent:
    """面向 7499 铝合金小样本温度-时间-性能预测的物理启发特征工程"""

    def __init__(self):
        self.feature_names = []

        self.base_strain = DOMAIN_PARAMS['base_strain']
        self.base_tensile = DOMAIN_PARAMS['base_tensile']
        self.base_yield = DOMAIN_PARAMS['base_yield']

        self.Q = DOMAIN_PARAMS['activation_energy_Q']
        self.R = DOMAIN_PARAMS['gas_constant_R']

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
        - 当前窗口以“升温/延时 -> 更充分固溶 -> 强度总体提升”为主
        - 时间效应前快后慢，宜用饱和/对数形式
        - 高温长时只施加轻微软化/平台化修正，避免违背数据总体上升趋势
        """
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15

        # 温度激活：450°C 左右进入更充分固溶区，465°C 左右进一步增强
        temp_act_1 = self._sigmoid((temp - DOMAIN_PARAMS['critical_temp_regime_shift']) / 8.0)
        temp_act_2 = self._sigmoid((temp - DOMAIN_PARAMS['secondary_temp_boundary']) / 6.0)

        # 时间激活：前期快速，随后饱和
        time_act_fast = 1.0 - np.exp(-time / DOMAIN_PARAMS['critical_time_regime_shift'])
        time_act_slow = self._safe_log1p(time) / self._safe_log1p(DOMAIN_PARAMS['very_long_time_boundary'])

        # Arrhenius 动力学等效
        arrhenius = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-6, None)))
        kinetic_dose = time * arrhenius
        kinetic_norm = kinetic_dose / (np.max(kinetic_dose) + 1e-12) if kinetic_dose.size > 0 else kinetic_dose

        # 综合处理充分度
        solution_progress = (
            0.38 * temp_act_1 +
            0.20 * temp_act_2 +
            0.24 * time_act_fast +
            0.08 * time_act_slow +
            0.10 * kinetic_norm
        )

        # 高温长时下局部平台/组织粗化风险
        high_temp_gate = self._sigmoid((temp - DOMAIN_PARAMS['high_temp_risk_boundary']) / 3.5)
        long_time_gate = self._sigmoid((time - DOMAIN_PARAMS['long_time_boundary']) / 1.8)
        overprocess_risk = high_temp_gate * long_time_gate

        # Tensile / Yield：整体增长 + 轻微高温长时修正
        baseline_tensile = (
            self.base_tensile
            + 275.0 * solution_progress
            - 22.0 * overprocess_risk
        )

        baseline_yield = (
            self.base_yield
            + 215.0 * solution_progress
            - 18.0 * overprocess_risk
        )

        # 应变：不强制与强度负相关；组织改善下可同步升高
        baseline_strain = (
            self.base_strain
            - 2.0
            + 5.2 * solution_progress
            + 0.9 * temp_act_2
            - 0.7 * overprocess_risk
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log1p(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-6, None)

        T1 = DOMAIN_PARAMS['critical_temp_regime_shift']
        T2 = DOMAIN_PARAMS['secondary_temp_boundary']
        T3 = DOMAIN_PARAMS['high_temp_risk_boundary']
        t1 = DOMAIN_PARAMS['critical_time_regime_shift']
        t2 = DOMAIN_PARAMS['long_time_boundary']
        t3 = DOMAIN_PARAMS['very_long_time_boundary']

        # -----------------------------
        # 1. 基础工艺特征
        # -----------------------------
        X['temp'] = temp
        X['time'] = time
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # -----------------------------
        # 2. 物理基线特征
        # -----------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield
        X['baseline_strength_gap'] = baseline_tensile - baseline_yield
        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)

        # -----------------------------
        # 3. 非线性与温时耦合
        # -----------------------------
        X['temp_sq_centered'] = ((temp - 450.0) / 20.0) ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time

        # -----------------------------
        # 4. 动力学等效特征
        # -----------------------------
        arrhenius = np.exp(-self.Q / (self.R * np.clip(temp_k, 1e-6, None)))
        thermal_dose = time * arrhenius
        log_thermal_dose = self._safe_log1p(thermal_dose)

        X['arrhenius'] = arrhenius
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = log_thermal_dose

        # Larson-Miller Parameter
        log10_time = np.log10(np.clip(time, 1e-6, None))
        X['larson_miller'] = temp_k * (DOMAIN_PARAMS['lmp_C'] + log10_time)

        # 简化 temper-like parameter
        X['temper_parameter'] = temp_k * log_time

        # -----------------------------
        # 5. 机制区间检测特征（关键）
        # -----------------------------
        X['is_low_temp'] = (temp < T1).astype(float)
        X['is_mid_temp'] = ((temp >= T1) & (temp < T3)).astype(float)
        X['is_high_temp'] = (temp >= T3).astype(float)

        X['is_short_time'] = (time < t1).astype(float)
        X['is_mid_time'] = ((time >= t1) & (time < t2)).astype(float)
        X['is_long_time'] = (time >= t2).astype(float)

        # 重点误差区域对应的机制标记
        X['is_low_temp_long_time'] = ((temp < T1) & (time >= t2)).astype(float)      # 440/24h 类
        X['is_midhigh_temp_long_time'] = ((temp >= T2) & (time >= t2)).astype(float) # 470/12h, 460/12h 类
        X['is_high_temp_long_time'] = ((temp >= T3) & (time >= t2)).astype(float)
        X['is_mid_temp_long_time'] = ((temp >= T1) & (temp < T3) & (time >= t2)).astype(float)
        X['is_high_temp_mid_time'] = ((temp >= T3) & (time >= t1) & (time < t2)).astype(float)

        # -----------------------------
        # 6. 分段距离 / 边界相对位置
        # -----------------------------
        X['temp_rel_T1'] = temp - T1
        X['temp_rel_T2'] = temp - T2
        X['temp_rel_T3'] = temp - T3
        X['time_rel_t1'] = time - t1
        X['time_rel_t2'] = time - t2

        X['relu_above_T1'] = np.maximum(temp - T1, 0.0)
        X['relu_above_T2'] = np.maximum(temp - T2, 0.0)
        X['relu_above_T3'] = np.maximum(temp - T3, 0.0)
        X['relu_below_T1'] = np.maximum(T1 - temp, 0.0)
        X['relu_above_t1'] = np.maximum(time - t1, 0.0)
        X['relu_above_t2'] = np.maximum(time - t2, 0.0)

        # 平滑门控
        X['temp_gate_T1'] = self._sigmoid((temp - T1) / 8.0)
        X['temp_gate_T2'] = self._sigmoid((temp - T2) / 6.0)
        X['temp_gate_T3'] = self._sigmoid((temp - T3) / 4.0)
        X['time_gate_t1'] = self._sigmoid((time - t1) / 1.5)
        X['time_gate_t2'] = self._sigmoid((time - t2) / 2.0)

        # -----------------------------
        # 7. 机制导向特征
        # -----------------------------
        # 低温区：时间补偿更重要
        X['low_temp_time_comp'] = ((temp < T1).astype(float)) * log_time

        # 中高温区：温时协同促进充分固溶
        X['solution_synergy'] = X['temp_gate_T1'].values * (1.0 - np.exp(-time / 8.0))

        # 470°C / 12h 附近：高温长时的窗口敏感性
        X['high_temp_long_risk'] = X['temp_gate_T3'].values * X['time_gate_t2'].values
        X['boundary_coupling'] = np.maximum(temp - T2, 0.0) * np.maximum(time - t1, 0.0)

        # 440/24h 类：低温长时补偿但可能仍未完全达到高温充分固溶
        X['low_temp_long_compensation'] = ((temp < T1).astype(float)) * np.maximum(time - t2, 0.0)

        # 460/12h 类：中高温 + 长时的临界窗口
        X['critical_window_indicator'] = (
            self._sigmoid((temp - 460.0) / 4.0) *
            self._sigmoid((time - 12.0) / 1.5)
        )

        # -----------------------------
        # 8. 接近参考窗口特征
        # -----------------------------
        T_ref = DOMAIN_PARAMS['reference_temp_center']
        t_ref = DOMAIN_PARAMS['reference_time_center']
        log_t_ref = np.log1p(t_ref)

        X['temp_peak_proximity'] = np.exp(-((temp - T_ref) / 12.0) ** 2)
        X['time_peak_proximity'] = np.exp(-((log_time - log_t_ref) / 0.7) ** 2)
        X['joint_peak_window'] = X['temp_peak_proximity'].values * X['time_peak_proximity'].values

        # -----------------------------
        # 9. 相对基线 / 残差风格特征
        # -----------------------------
        X['baseline_tensile_x_temp_rel'] = baseline_tensile * (temp - T1) / 100.0
        X['baseline_yield_x_time_rel'] = baseline_yield * (time - t1) / 10.0
        X['baseline_strain_x_highT'] = baseline_strain * X['temp_gate_T3'].values

        # -----------------------------
        # 10. 原始铸态参考
        # -----------------------------
        X['base_tensile'] = self.base_tensile
        X['base_yield'] = self.base_yield
        X['base_strain'] = self.base_strain

        X['baseline_tensile_gain_over_cast'] = baseline_tensile - self.base_tensile
        X['baseline_yield_gain_over_cast'] = baseline_yield - self.base_yield
        X['baseline_strain_gain_over_cast'] = baseline_strain - self.base_strain

        # -----------------------------
        # 11. 小样本稳健压缩特征
        # -----------------------------
        X['time_saturation'] = 1.0 - np.exp(-time / 6.0)
        X['temp_saturation'] = self._sigmoid((temp - 450.0) / 8.0)
        X['combined_saturation'] = X['time_saturation'].values * X['temp_saturation'].values

        # 清理
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