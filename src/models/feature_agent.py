import pandas as pd
import numpy as np

# =========================
# 领域知识参数
# =========================
# 说明：
# 1) 这里的数据温度为 440/460/470°C，明显不是传统 7xxx 人工时效(100~180°C)窗口，
#    更接近高温均匀化/固溶/高温暴露条件。
# 2) 在这种高温区，主导机制更可能是：
#    - 析出相溶解 / 再分配
#    - 晶界相变化
#    - 回复与再结晶倾向
#    - 长时间暴露导致粗化、软化
# 3) 因此不能沿用低温时效峰值参数，而应围绕“高温暴露剂量 + 阶段转变”构造特征。
DOMAIN_PARAMS = {
    # 基础物理常数
    'gas_constant_J_molK': 8.314,

    # 对 Al-Zn-Mg-Cu(7xxx) 合金，高温扩散/组织演化的经验激活能量级
    # 取中等量级，用于构造相对动力学指标，而非严格材料常数
    'diffusion_activation_energy_J_mol': 115000.0,
    'recovery_activation_energy_J_mol': 105000.0,
    'coarsening_activation_energy_J_mol': 125000.0,

    # 当前数据对应的高温区间边界（按样本温度分布和物理机制划分）
    'regime1_temp_C': 445.0,   # 低高温暴露区：接近 440°C，组织变化相对较弱
    'regime2_temp_C': 465.0,   # 强演化区：460~470°C 对组织更敏感
    'solution_like_temp_C': 475.0,  # 接近更强溶解/高温活化边界的软阈值

    # 时间边界
    'short_time_h': 2.0,
    'medium_time_h': 8.0,
    'long_time_h': 18.0,
    'very_long_time_h': 24.0,

    # 高温暴露下的组织演化拐点
    'recovery_onset_temp_C': 430.0,
    'rapid_softening_temp_C': 460.0,
    'coarsening_sensitive_temp_C': 455.0,

    # 温度/时间软门控尺度
    'temp_transition_width_C': 6.0,
    'time_transition_width_h': 4.0,
    'log_time_width': 0.55,

    # 基准性能：原始凝固样
    'base_strain_pct': 6.94,
    'base_tensile_MPa': 145.83,
    'base_yield_MPa': 96.60,

    # 高温处理后的经验强化/软化幅值上限
    # 启发：处理后强度显著高于铸态，但 470°C-12h 误差很大，说明该区有明显机制跃迁
    'max_strength_increment_MPa': 260.0,
    'max_yield_increment_MPa': 190.0,
    'max_strain_increment_pct': 10.0,

    # 竞争机制权重
    'precipitation_like_weight': 0.75,
    'solution_recovery_weight': 0.95,
    'coarsening_weight': 1.05,

    # 稳健边界
    'min_valid_temp_C': 400.0,
    'max_valid_temp_C': 520.0,
    'min_valid_time_h': 0.0,
    'max_valid_time_h': 48.0,
}


class FeatureAgent:
    """基于高温热暴露/组织演化机理的特征工程"""

    def __init__(self):
        self.feature_names = []
        self.params = DOMAIN_PARAMS

        self.R = self.params['gas_constant_J_molK']
        self.Q_diff = self.params['diffusion_activation_energy_J_mol']
        self.Q_rec = self.params['recovery_activation_energy_J_mol']
        self.Q_coarse = self.params['coarsening_activation_energy_J_mol']

        self.base_strain = self.params['base_strain_pct']
        self.base_tensile = self.params['base_tensile_MPa']
        self.base_yield = self.params['base_yield_MPa']

        self.regime1_temp = self.params['regime1_temp_C']
        self.regime2_temp = self.params['regime2_temp_C']
        self.solution_like_temp = self.params['solution_like_temp_C']

        self.short_time = self.params['short_time_h']
        self.medium_time = self.params['medium_time_h']
        self.long_time = self.params['long_time_h']
        self.very_long_time = self.params['very_long_time_h']

        self.recovery_onset_temp = self.params['recovery_onset_temp_C']
        self.rapid_softening_temp = self.params['rapid_softening_temp_C']
        self.coarsening_sensitive_temp = self.params['coarsening_sensitive_temp_C']

        self.temp_transition_width = self.params['temp_transition_width_C']
        self.time_transition_width = self.params['time_transition_width_h']
        self.log_time_width = self.params['log_time_width']

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
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    def _clip_inputs(self, temp, time):
        temp = np.clip(np.asarray(temp, dtype=float), self.min_temp, self.max_temp)
        time = np.clip(np.asarray(time, dtype=float), self.min_time, self.max_time)
        return temp, time

    # =========================
    # 第二步：physics_baseline(temp, time)
    # =========================
    def physics_baseline(self, temp, time):
        """
        基于世界知识的近似物理基线：
        - 高温下组织演化由热激活控制，速率对温度高度敏感
        - 强度提升来自“高温处理后形成更有利组织状态”的净收益
        - 但 460~470°C 且中长时，会出现回复/粗化/局部软化竞争
        - 延性与强度通常竞争，但在明显软化阶段可回升

        返回:
            baseline_strain, baseline_tensile, baseline_yield
        """
        temp, time = self._clip_inputs(temp, time)
        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # 热激活项：用相对量即可
        diff_rate = np.exp(-self.Q_diff / (self.R * temp_k))
        rec_rate = np.exp(-self.Q_rec / (self.R * temp_k))
        coarse_rate = np.exp(-self.Q_coarse / (self.R * temp_k))

        # 归一化到当前窗口，避免数值过小
        diff_ref = np.exp(-self.Q_diff / (self.R * (460.0 + 273.15)))
        rec_ref = np.exp(-self.Q_rec / (self.R * (460.0 + 273.15)))
        coarse_ref = np.exp(-self.Q_coarse / (self.R * (460.0 + 273.15)))

        diff_norm = diff_rate / diff_ref
        rec_norm = rec_rate / rec_ref
        coarse_norm = coarse_rate / coarse_ref

        # 高温处理“有效剂量”
        effective_dose = diff_norm * log_time
        saturation = 1.0 - np.exp(-np.clip(time, 0, None) / 6.0)

        # 温区机制门控
        regime1_gate = self._sigmoid((temp - self.regime1_temp) / self.temp_transition_width)
        regime2_gate = self._sigmoid((temp - self.regime2_temp) / self.temp_transition_width)
        solution_gate = self._sigmoid((temp - self.solution_like_temp) / self.temp_transition_width)

        # 组织改善项：温度升高 + 时间增加带来更充分组织转变
        strengthening_progress = (
            0.55 * saturation
            + 0.45 * np.tanh(np.clip(effective_dose, 0, None))
        )

        # 470°C-12h 误差最大，说明该处简单平滑不足，需要强调“中高温+中长时”的跃迁
        transition_boost = regime2_gate * self._sigmoid((time - 10.0) / 2.5)

        # 软化/回复/粗化项
        recovery_softening = rec_norm * np.maximum(log_time - np.log1p(4.0), 0.0) * self._sigmoid((temp - self.recovery_onset_temp) / 8.0)
        coarsening_softening = coarse_norm * np.maximum(log_time - np.log1p(8.0), 0.0) * self._sigmoid((temp - self.coarsening_sensitive_temp) / 6.0)
        solution_softening = solution_gate * self._sigmoid((time - 6.0) / 2.0)

        net_strength_index = (
            1.00 * strengthening_progress
            + 0.35 * transition_boost
            - 0.30 * recovery_softening
            - 0.22 * coarsening_softening
            - 0.12 * solution_softening
        )

        net_strength_index = np.clip(net_strength_index, -0.2, 1.4)

        # 基线强度：以原始铸态为起点，加上净强化
        tensile = self.base_tensile + self.params['max_strength_increment_MPa'] * np.clip(net_strength_index, 0, 1.0)
        yield_strength_factor = (
            0.70
            + 0.06 * regime1_gate
            + 0.05 * regime2_gate
            - 0.03 * solution_gate
        )
        yield_ = self.base_yield + self.params['max_yield_increment_MPa'] * np.clip(net_strength_index * yield_strength_factor, 0, 1.0)

        # 延性：与强化竞争，但高温长时软化可回升
        ductility_soft_recovery = (
            0.85 * recovery_softening
            + 0.70 * coarsening_softening
            + 0.35 * solution_softening
        )
        strain = (
            self.base_strain
            + self.params['max_strain_increment_pct'] * (
                0.85 * transition_boost
                + 0.45 * regime2_gate
                + 0.35 * ductility_soft_recovery
                - 0.28 * np.clip(net_strength_index, 0, 1.0)
            )
        )

        # 物理约束
        yield_ = np.minimum(yield_, tensile - 5.0)
        tensile = np.maximum(tensile, self.base_tensile)
        yield_ = np.maximum(yield_, self.base_yield)
        strain = np.maximum(strain, 0.5)

        return strain, tensile, yield_

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        temp, time = self._clip_inputs(temp, time)

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-12, None)

        # 原始输入
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # 基础非线性
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # Arrhenius 热激活特征
        arr_diff = np.exp(-self.Q_diff / (self.R * temp_k))
        arr_rec = np.exp(-self.Q_rec / (self.R * temp_k))
        arr_coarse = np.exp(-self.Q_coarse / (self.R * temp_k))

        ref_diff = np.exp(-self.Q_diff / (self.R * (460.0 + 273.15)))
        ref_rec = np.exp(-self.Q_rec / (self.R * (460.0 + 273.15)))
        ref_coarse = np.exp(-self.Q_coarse / (self.R * (460.0 + 273.15)))

        arr_diff_norm = arr_diff / ref_diff
        arr_rec_norm = arr_rec / ref_rec
        arr_coarse_norm = arr_coarse / ref_coarse

        X['arr_diff'] = arr_diff
        X['arr_rec'] = arr_rec
        X['arr_coarse'] = arr_coarse
        X['arr_diff_norm'] = arr_diff_norm
        X['arr_rec_norm'] = arr_rec_norm
        X['arr_coarse_norm'] = arr_coarse_norm

        X['diffusion_dose'] = time * arr_diff_norm
        X['recovery_dose'] = time * arr_rec_norm
        X['coarsening_dose'] = time * arr_coarse_norm
        X['log_diffusion_dose'] = self._safe_log(X['diffusion_dose'].values)
        X['log_recovery_dose'] = self._safe_log(X['recovery_dose'].values)
        X['log_coarsening_dose'] = self._safe_log(X['coarsening_dose'].values)

        # =========================
        # 第一步要求：基于 DOMAIN_PARAMS 的机制区间检测
        # =========================
        X['is_regime1_or_higher'] = (temp >= self.regime1_temp).astype(float)
        X['is_regime2_or_higher'] = (temp >= self.regime2_temp).astype(float)
        X['is_solution_like_temp'] = (temp >= self.solution_like_temp).astype(float)

        X['is_short_time'] = (time < self.short_time).astype(float)
        X['is_medium_time'] = ((time >= self.short_time) & (time < self.medium_time)).astype(float)
        X['is_long_time'] = ((time >= self.medium_time) & (time < self.long_time)).astype(float)
        X['is_very_long_time'] = (time >= self.very_long_time).astype(float)

        # 相对边界残差特征
        X['temp_relative_to_regime1'] = temp - self.regime1_temp
        X['temp_relative_to_regime2'] = temp - self.regime2_temp
        X['temp_relative_to_solution_like'] = temp - self.solution_like_temp

        X['time_relative_to_short'] = time - self.short_time
        X['time_relative_to_medium'] = time - self.medium_time
        X['time_relative_to_long'] = time - self.long_time
        X['log_time_relative_to_medium'] = log_time - np.log1p(self.medium_time)
        X['log_time_relative_to_long'] = log_time - np.log1p(self.long_time)

        X['relu_above_regime1'] = np.maximum(temp - self.regime1_temp, 0)
        X['relu_above_regime2'] = np.maximum(temp - self.regime2_temp, 0)
        X['relu_above_solution_like'] = np.maximum(temp - self.solution_like_temp, 0)

        X['relu_above_short_time'] = np.maximum(time - self.short_time, 0)
        X['relu_above_medium_time'] = np.maximum(time - self.medium_time, 0)
        X['relu_above_long_time'] = np.maximum(time - self.long_time, 0)
        X['relu_above_very_long_time'] = np.maximum(time - self.very_long_time, 0)

        # 软门控特征
        regime1_gate = self._sigmoid((temp - self.regime1_temp) / self.temp_transition_width)
        regime2_gate = self._sigmoid((temp - self.regime2_temp) / self.temp_transition_width)
        solution_gate = self._sigmoid((temp - self.solution_like_temp) / self.temp_transition_width)

        short_to_medium_gate = self._sigmoid((time - self.short_time) / self.time_transition_width)
        medium_to_long_gate = self._sigmoid((time - self.medium_time) / self.time_transition_width)
        long_to_verylong_gate = self._sigmoid((time - self.long_time) / self.time_transition_width)

        X['regime1_gate'] = regime1_gate
        X['regime2_gate'] = regime2_gate
        X['solution_gate'] = solution_gate
        X['short_to_medium_gate'] = short_to_medium_gate
        X['medium_to_long_gate'] = medium_to_long_gate
        X['long_to_verylong_gate'] = long_to_verylong_gate

        # =========================
        # 第三步：基线预测值本身作为特征
        # =========================
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        # 基线导出的状态量
        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_strength_sum'] = baseline_tensile + baseline_yield
        X['baseline_strength_minus_base_tensile'] = baseline_tensile - self.base_tensile
        X['baseline_yield_minus_base_yield'] = baseline_yield - self.base_yield
        X['baseline_strain_minus_base_strain'] = baseline_strain - self.base_strain

        # “相对基线”的输入偏移特征：让模型学习残差
        X['temp_times_baseline_tensile'] = temp * baseline_tensile
        X['time_times_baseline_tensile'] = time * baseline_tensile
        X['temp_times_baseline_yield'] = temp * baseline_yield
        X['time_times_baseline_yield'] = time * baseline_yield
        X['temp_times_baseline_strain'] = temp * baseline_strain
        X['log_time_times_baseline_tensile'] = log_time * baseline_tensile
        X['log_time_times_baseline_yield'] = log_time * baseline_yield
        X['log_time_times_baseline_strain'] = log_time * baseline_strain

        # 基线相对机制边界的修正项
        X['baseline_tensile_x_regime2'] = baseline_tensile * regime2_gate
        X['baseline_yield_x_regime2'] = baseline_yield * regime2_gate
        X['baseline_strain_x_regime2'] = baseline_strain * regime2_gate

        X['baseline_tensile_x_longtime'] = baseline_tensile * medium_to_long_gate
        X['baseline_yield_x_longtime'] = baseline_yield * medium_to_long_gate
        X['baseline_strain_x_longtime'] = baseline_strain * medium_to_long_gate

        # =========================
        # 重点针对最大误差区：470°C, 12h
        # 该区域可能存在机制跃迁，需显式构造邻域与风险特征
        # =========================
        target_temp = 470.0
        target_time = 12.0
        temp_prox_470 = np.exp(-((temp - target_temp) / 6.0) ** 2)
        time_prox_12 = np.exp(-((log_time - np.log1p(target_time)) / 0.35) ** 2)
        joint_prox_470_12 = temp_prox_470 * time_prox_12

        X['temp_prox_470'] = temp_prox_470
        X['time_prox_12h'] = time_prox_12
        X['joint_prox_470_12'] = joint_prox_470_12

        # 中高温-中长时组织跃迁
        X['transition_risk_460_470_midtime'] = (
            regime2_gate
            * self._sigmoid((time - 9.0) / 2.0)
            * (1.0 - self._sigmoid((time - 20.0) / 3.0))
        )

        # 高温长时软化/粗化
        recovery_drive = arr_rec_norm * np.maximum(log_time - np.log1p(2.0), 0.0) * self._sigmoid((temp - self.recovery_onset_temp) / 8.0)
        coarsening_drive = arr_coarse_norm * np.maximum(log_time - np.log1p(6.0), 0.0) * self._sigmoid((temp - self.coarsening_sensitive_temp) / 6.0)
        solution_drive = solution_gate * short_to_medium_gate

        strengthening_drive = (
            self.params['precipitation_like_weight']
            * (1.0 - np.exp(-time / 6.0))
            * (0.55 + 0.45 * regime1_gate)
            * (0.75 + 0.25 * arr_diff_norm)
        )

        softening_drive = (
            self.params['solution_recovery_weight'] * recovery_drive
            + self.params['coarsening_weight'] * coarsening_drive
            + 0.45 * solution_drive
        )

        net_drive = strengthening_drive - softening_drive

        X['strengthening_drive'] = strengthening_drive
        X['recovery_drive'] = recovery_drive
        X['coarsening_drive'] = coarsening_drive
        X['solution_drive'] = solution_drive
        X['softening_drive'] = softening_drive
        X['net_drive'] = net_drive

        # 动态推理特征
        X['under_processed_tendency'] = (1.0 - regime1_gate) * (1.0 - short_to_medium_gate)
        X['transition_tendency'] = regime2_gate * medium_to_long_gate * (1.0 - long_to_verylong_gate)
        X['overexposure_tendency'] = regime2_gate * long_to_verylong_gate * self._sigmoid((temp - 465.0) / 4.0)

        X['strength_ductility_tradeoff_index'] = net_drive - 0.35 * solution_drive + 0.20 * coarsening_drive
        X['yield_tensile_coupling_index'] = baseline_yield / np.maximum(baseline_tensile, 1e-6) + 0.15 * net_drive
        X['ductility_recovery_index'] = 0.55 * softening_drive + 0.20 * regime2_gate - 0.18 * strengthening_drive

        # 分区交互
        X['temp_in_regime1'] = temp * ((temp >= self.regime1_temp) & (temp < self.regime2_temp)).astype(float)
        X['temp_in_regime2'] = temp * (temp >= self.regime2_temp).astype(float)
        X['log_time_in_regime1'] = log_time * ((temp >= self.regime1_temp) & (temp < self.regime2_temp)).astype(float)
        X['log_time_in_regime2'] = log_time * (temp >= self.regime2_temp).astype(float)

        X['temp_log_time_regime1'] = temp * log_time * ((temp >= self.regime1_temp) & (temp < self.regime2_temp)).astype(float)
        X['temp_log_time_regime2'] = temp * log_time * (temp >= self.regime2_temp).astype(float)

        # 等效暴露参数
        X['larson_miller_like'] = temp_k * (20.0 + log_time)
        X['equivalent_exposure_index'] = temp * log_time
        X['high_temp_exposure_index'] = np.maximum(temp - 440.0, 0) * log_time
        X['high_temp_long_time_penalty'] = np.maximum(temp - 460.0, 0) * np.maximum(time - 8.0, 0)
        X['470_12_penalty_like'] = np.maximum(temp - 465.0, 0) * np.maximum(time - 10.0, 0)

        # 相对原始样品的潜力
        X['base_strength_sum'] = self.base_tensile + self.base_yield
        X['base_strength_ratio'] = self.base_yield / self.base_tensile
        X['process_strength_potential'] = (
            0.55 * strengthening_drive
            + 0.25 * joint_prox_470_12
            + 0.20 * regime2_gate
            - 0.30 * softening_drive
        )
        X['process_ductility_potential'] = (
            0.40 * regime2_gate
            + 0.35 * solution_drive
            + 0.30 * coarsening_drive
            - 0.20 * strengthening_drive
        )

        # 基线残差学习辅助特征（输入相对于典型机制点）
        X['temp_residual_to_470'] = temp - 470.0
        X['time_residual_to_12'] = time - 12.0
        X['log_time_residual_to_12'] = log_time - np.log1p(12.0)
        X['temp_time_residual_product_470_12'] = (temp - 470.0) * (time - 12.0)

        # 稳健清洗
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

    # 示例：物理基线输出
    bs, bt, by = agent.physics_baseline(
        train['temp'].values[:5],
        train['time'].values[:5]
    )
    print("Physics baseline sample:")
    print("strain:", bs)
    print("tensile:", bt)
    print("yield:", by)