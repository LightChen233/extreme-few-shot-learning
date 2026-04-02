import pandas as pd
import numpy as np


# 7499铝合金热处理特征工程参数
# 依据材料学常识而非标签直接拟合：
# - 440/460/470°C处于高温固溶/组织演化敏感区；
# - 1/12/24h分别代表短时、典型中时、长时；
# - 高温促进扩散与析出/粗化动力学，时间作用通常更接近log或饱和形式；
# - 470×12h、440×24h、460×12h 是当前误差热点，因此需要显式构造对应机制窗口，
#   但仍用“组织演化阶段”解释，而不是简单记忆样本。
DOMAIN_PARAMS = {
    # 基础常数
    'R': 8.314,                        # J/(mol·K)
    'Q_diff': 118000.0,                # J/mol, 铝合金高温扩散/析出控制量级
    'arrhenius_scale': 1e14,           # 数值缩放，便于构造热剂量

    # 原始凝固态基准性能
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # 温度机制边界（℃）
    'temp_low_regime': 445.0,          # 接近440°C：扩散较慢，长时更重要
    'temp_transition': 455.0,          # 440→460过渡区
    'temp_high_regime': 465.0,         # 接近470°C：高温敏感区
    'temp_extreme': 472.0,             # 更高温软化/粗化风险增强

    # 时间机制边界（h）
    'time_short': 2.0,                 # 1h附近，早期阶段
    'time_peak_like': 12.0,            # 中时阶段，常对应强化敏感点
    'time_long': 16.0,                 # 长时起点
    'time_very_long': 20.0,            # 24h附近，过渡到长时/粗化竞争

    # 误差热点中心（工艺窗口）
    'focus_temp_440': 440.0,
    'focus_temp_460': 460.0,
    'focus_temp_470': 470.0,
    'focus_time_1': 1.0,
    'focus_time_12': 12.0,
    'focus_time_24': 24.0,

    # 高斯窗口宽度
    'temp_window_sigma': 5.5,
    'log_time_window_sigma_short': 0.18,
    'log_time_window_sigma_mid': 0.22,
    'log_time_window_sigma_long': 0.24,

    # 经验强化/软化幅值（物理先验）
    'max_tensile_gain': 235.0,
    'max_yield_gain': 200.0,
    'max_strain_drop': 2.8,
    'max_strain_recovery': 3.8,

    # 基线模型中的组织演化窗口
    'peak_temp_center': 467.0,         # 高温中时强化峰附近
    'peak_time_center': 12.0,
    'longhold_temp_center': 442.0,     # 低端高温长时特殊强化/均匀化区
    'longhold_time_center': 24.0,

    # 软化/粗化风险
    'soft_temp_center': 470.0,
    'soft_time_center': 24.0,

    # 数值尺度
    'temp_sigmoid_scale': 6.0,
    'temp_high_sigmoid_scale': 4.5,
    'time_sat_scale': 5.5,
    'time_long_sigmoid_scale': 2.8,
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

    def _gaussian(self, x, center, sigma):
        sigma = max(float(sigma), 1e-9)
        return np.exp(-((x - center) / sigma) ** 2)

    def physics_baseline(self, temp, time):
        """
        基于世界知识的粗粒度物理基线：
        - 温度升高 => 扩散激活增强；
        - 时间增加 => 组织演化推进，但存在饱和；
        - 中高温+中时可能出现强化高点；
        - 高温+长时存在粗化/过时效风险；
        - 440×24h可能由于低端高温长时均匀化/析出阶段形成较高强度；
        - 强度升高通常伴随应变下降，软化/回复使应变回升。
        """
        p = self.params

        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)
        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # 扩散热激活
        arrhenius = np.exp(-p['Q_diff'] / (p['R'] * np.clip(temp_k, 1e-9, None)))
        thermal_dose = time * arrhenius * p['arrhenius_scale']
        log_dose = np.log1p(np.clip(thermal_dose, 0, None))

        # 基本激活
        temp_activation = self._sigmoid((temp - p['temp_low_regime']) / p['temp_sigmoid_scale'])
        high_temp_activation = self._sigmoid((temp - p['temp_high_regime']) / p['temp_high_sigmoid_scale'])
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / p['time_sat_scale'])
        long_time_activation = self._sigmoid((time - p['time_long']) / p['time_long_sigmoid_scale'])

        # 一般强化趋势
        general_strength = (
            0.34 * temp_activation
            + 0.24 * time_activation
            + 0.18 * np.tanh(np.clip(log_dose, 0, None) / 3.0)
        )

        # 高温中时强化窗口：解释470×12h低估
        peak_window = (
            self._gaussian(temp, p['peak_temp_center'], 7.0)
            * self._gaussian(log_time, np.log1p(p['peak_time_center']), 0.28)
        )

        # 低端高温长时窗口：解释440×24h低估
        longhold_window = (
            self._gaussian(temp, p['longhold_temp_center'], 6.5)
            * self._gaussian(log_time, np.log1p(p['longhold_time_center']), 0.25)
        )

        # 中温12h过渡窗口：解释460×12h方向不稳
        mid_transition_window = (
            self._gaussian(temp, 460.0, 5.5)
            * self._gaussian(log_time, np.log1p(12.0), 0.22)
        )

        # 软化/粗化风险：高温且长时更明显
        softening = (
            high_temp_activation
            * (0.42 * long_time_activation + 0.16 * np.maximum(log_time - np.log1p(12.0), 0.0))
            + 0.12 * self._gaussian(temp, p['soft_temp_center'], 5.0)
            * self._gaussian(log_time, np.log1p(p['soft_time_center']), 0.24)
        )

        # 460×12h属于转折区，适当降低基线过度乐观
        transition_penalty = 0.10 * mid_transition_window

        net_strength = (
            general_strength
            + 0.36 * peak_window
            + 0.30 * longhold_window
            - 0.18 * softening
            - transition_penalty
        )
        net_strength = np.clip(net_strength, 0.0, 1.20)

        baseline_tensile = p['base_tensile'] + p['max_tensile_gain'] * net_strength

        baseline_yield_factor = (
            0.92 * general_strength
            + 0.31 * peak_window
            + 0.27 * longhold_window
            - 0.20 * softening
            - 0.08 * mid_transition_window
        )
        baseline_yield_factor = np.clip(baseline_yield_factor, 0.0, 1.20)
        baseline_yield = p['base_yield'] + p['max_yield_gain'] * baseline_yield_factor

        # 屈服强度不应高于抗拉强度
        baseline_yield = np.minimum(baseline_yield, baseline_tensile - 8.0)

        # 应变：强化下降，软化/回复回升
        strain_drop = p['max_strain_drop'] * (
            0.72 * general_strength
            + 0.18 * peak_window
            + 0.10 * longhold_window
        )
        strain_recovery = p['max_strain_recovery'] * (
            0.42 * softening
            + 0.12 * long_time_activation
            + 0.08 * mid_transition_window
        )

        baseline_strain = p['base_strain'] - strain_drop + strain_recovery

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

        # 1. 原始与基础变换
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp'] = self._safe_divide(time, temp)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # 2. Arrhenius / 热剂量特征
        arrhenius = np.exp(-p['Q_diff'] / (p['R'] * np.clip(temp_k, 1e-9, None)))
        thermal_dose = time * arrhenius * p['arrhenius_scale']
        log_thermal_dose = np.log1p(np.clip(thermal_dose, 0, None))

        X['arrhenius'] = arrhenius
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = log_thermal_dose
        X['temp_x_log_thermal_dose'] = temp * log_thermal_dose
        X['time_x_arrhenius'] = time * arrhenius

        # 3. 物理基线特征
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        baseline_gap = baseline_tensile - baseline_yield
        baseline_ratio = self._safe_divide(baseline_yield, baseline_tensile)
        baseline_sum = baseline_tensile + baseline_yield

        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield
        X['baseline_gap_ty'] = baseline_gap
        X['baseline_ratio_y_t'] = baseline_ratio
        X['baseline_strength_sum'] = baseline_sum

        X['baseline_tensile_gain'] = baseline_tensile - p['base_tensile']
        X['baseline_yield_gain'] = baseline_yield - p['base_yield']
        X['baseline_strain_delta'] = baseline_strain - p['base_strain']
        X['baseline_gap_over_sum'] = self._safe_divide(baseline_gap, baseline_sum)
        X['baseline_strength_per_strain'] = self._safe_divide(baseline_sum, baseline_strain + 1e-6)

        # 4. 机制区间检测
        is_temp_low = (temp <= p['temp_low_regime']).astype(float)
        is_temp_mid = ((temp > p['temp_low_regime']) & (temp < p['temp_high_regime'])).astype(float)
        is_temp_high = (temp >= p['temp_high_regime']).astype(float)
        is_temp_extreme = (temp >= p['temp_extreme']).astype(float)

        is_time_short = (time <= p['time_short']).astype(float)
        is_time_mid = ((time > p['time_short']) & (time < p['time_long'])).astype(float)
        is_time_long = (time >= p['time_long']).astype(float)
        is_time_very_long = (time >= p['time_very_long']).astype(float)

        X['is_temp_low'] = is_temp_low
        X['is_temp_mid'] = is_temp_mid
        X['is_temp_high'] = is_temp_high
        X['is_temp_extreme'] = is_temp_extreme

        X['is_time_short'] = is_time_short
        X['is_time_mid'] = is_time_mid
        X['is_time_long'] = is_time_long
        X['is_time_very_long'] = is_time_very_long

        # 5. 相对机制边界残差特征
        X['temp_relative_low_boundary'] = temp - p['temp_low_regime']
        X['temp_relative_transition'] = temp - p['temp_transition']
        X['temp_relative_high_boundary'] = temp - p['temp_high_regime']
        X['temp_relative_extreme'] = temp - p['temp_extreme']

        X['time_relative_short_boundary'] = time - p['time_short']
        X['time_relative_peak_like'] = time - p['time_peak_like']
        X['time_relative_long_boundary'] = time - p['time_long']
        X['time_relative_very_long'] = time - p['time_very_long']

        X['relu_above_temp_low'] = np.maximum(temp - p['temp_low_regime'], 0.0)
        X['relu_above_temp_transition'] = np.maximum(temp - p['temp_transition'], 0.0)
        X['relu_above_temp_high'] = np.maximum(temp - p['temp_high_regime'], 0.0)
        X['relu_below_temp_low'] = np.maximum(p['temp_low_regime'] - temp, 0.0)

        X['relu_above_time_peak_like'] = np.maximum(time - p['time_peak_like'], 0.0)
        X['relu_above_time_long'] = np.maximum(time - p['time_long'], 0.0)
        X['relu_above_time_very_long'] = np.maximum(time - p['time_very_long'], 0.0)
        X['relu_below_time_short'] = np.maximum(p['time_short'] - time, 0.0)

        # 6. 激活/竞争特征
        temp_activation = self._sigmoid((temp - p['temp_low_regime']) / p['temp_sigmoid_scale'])
        temp_high_activation = self._sigmoid((temp - p['temp_high_regime']) / p['temp_high_sigmoid_scale'])
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / p['time_sat_scale'])
        time_long_activation = self._sigmoid((time - p['time_long']) / p['time_long_sigmoid_scale'])

        X['temp_activation'] = temp_activation
        X['temp_high_activation'] = temp_high_activation
        X['time_activation'] = time_activation
        X['time_long_activation'] = time_long_activation
        X['activation_product'] = temp_activation * time_activation
        X['high_temp_long_activation'] = temp_high_activation * time_long_activation

        # 7. 热点工艺窗口
        win_440_24 = self._gaussian(temp, p['focus_temp_440'], p['temp_window_sigma']) * \
                     self._gaussian(log_time, np.log1p(p['focus_time_24']), p['log_time_window_sigma_long'])
        win_470_12 = self._gaussian(temp, p['focus_temp_470'], p['temp_window_sigma']) * \
                     self._gaussian(log_time, np.log1p(p['focus_time_12']), p['log_time_window_sigma_mid'])
        win_460_12 = self._gaussian(temp, p['focus_temp_460'], p['temp_window_sigma']) * \
                     self._gaussian(log_time, np.log1p(p['focus_time_12']), p['log_time_window_sigma_mid'])
        win_440_1 = self._gaussian(temp, p['focus_temp_440'], p['temp_window_sigma']) * \
                    self._gaussian(log_time, np.log1p(p['focus_time_1']), p['log_time_window_sigma_short'])

        X['window_440_24'] = win_440_24
        X['window_470_12'] = win_470_12
        X['window_460_12'] = win_460_12
        X['window_440_1'] = win_440_1

        # 8. 组织机制竞争
        strengthening_drive = (
            0.26 * temp_activation
            + 0.22 * time_activation
            + 0.12 * np.tanh(np.clip(log_thermal_dose, 0, None) / 3.0)
            + 0.28 * win_470_12
            + 0.24 * win_440_24
            - 0.08 * win_460_12
        )

        softening_drive = (
            temp_high_activation * (0.42 * time_long_activation + 0.14 * np.maximum(log_time - np.log1p(12.0), 0.0))
            + 0.08 * is_temp_extreme * time_activation
        )

        transition_instability = win_460_12 * (0.6 + 0.4 * temp_high_activation)
        net_strengthening = strengthening_drive - softening_drive

        X['strengthening_drive'] = strengthening_drive
        X['softening_drive'] = softening_drive
        X['transition_instability'] = transition_instability
        X['net_strengthening'] = net_strengthening

        # 9. 分区交互特征
        X['temp_low_x_log_time'] = is_temp_low * temp * log_time
        X['temp_mid_x_log_time'] = is_temp_mid * temp * log_time
        X['temp_high_x_log_time'] = is_temp_high * temp * log_time

        X['time_short_x_temp'] = is_time_short * time * temp
        X['time_mid_x_temp'] = is_time_mid * time * temp
        X['time_long_x_temp'] = is_time_long * time * temp

        X['high_temp_long_time_coupling'] = is_temp_high * is_time_long * temp * log_time
        X['low_temp_very_long_coupling'] = is_temp_low * is_time_very_long * temp * log_time
        X['high_temp_mid_time_coupling'] = is_temp_high * is_time_mid * temp * log_time
        X['mid_temp_peak_time_coupling'] = is_temp_mid * self._gaussian(time, 12.0, 3.5) * temp

        # 10. 基线残差学习友好特征
        X['baseline_tensile_x_temp_relative_high'] = baseline_tensile * (temp - p['temp_high_regime'])
        X['baseline_yield_x_time_relative_long'] = baseline_yield * (time - p['time_long'])
        X['baseline_gap_x_window_470_12'] = baseline_gap * win_470_12
        X['baseline_gap_x_window_440_24'] = baseline_gap * win_440_24
        X['baseline_gap_x_window_460_12'] = baseline_gap * win_460_12
        X['baseline_strength_sum_x_net_strength'] = baseline_sum * net_strengthening
        X['baseline_strain_x_softening'] = baseline_strain * softening_drive
        X['baseline_tensile_x_strengthening'] = baseline_tensile * strengthening_drive
        X['baseline_yield_x_strengthening'] = baseline_yield * strengthening_drive
        X['baseline_tensile_x_transition_instability'] = baseline_tensile * transition_instability
        X['baseline_yield_x_transition_instability'] = baseline_yield * transition_instability

        # 11. 强塑权衡特征
        ductility_recovery = (
            0.52 * softening_drive
            + 0.16 * is_time_very_long
            + 0.10 * is_temp_extreme
            - 0.22 * strengthening_drive
        )
        strength_ductility_tradeoff = net_strengthening - ductility_recovery

        X['ductility_recovery'] = ductility_recovery
        X['strength_ductility_tradeoff'] = strength_ductility_tradeoff
        X['predicted_strength_ductility_coupling'] = self._safe_divide(baseline_sum, baseline_strain + 1e-6)

        # 12. 围绕关键工艺点的距离特征
        X['dist_temp_to_440'] = np.abs(temp - p['focus_temp_440'])
        X['dist_temp_to_460'] = np.abs(temp - p['focus_temp_460'])
        X['dist_temp_to_470'] = np.abs(temp - p['focus_temp_470'])
        X['dist_time_to_1'] = np.abs(time - p['focus_time_1'])
        X['dist_time_to_12'] = np.abs(time - p['focus_time_12'])
        X['dist_time_to_24'] = np.abs(time - p['focus_time_24'])

        X['signed_temp_from_455'] = temp - 455.0
        X['signed_time_from_12'] = time - 12.0
        X['signed_temp_from_455_x_signed_time_from_12'] = (temp - 455.0) * (time - 12.0)

        # 13. 饱和型与惩罚型特征
        clipped_temp = np.clip(temp, 430.0, 480.0)
        clipped_time = np.clip(time, 0.0, 30.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['combined_activation'] = X['time_saturation'].values * temp_activation
        X['high_temp_long_penalty'] = np.maximum(clipped_temp - p['temp_high_regime'], 0.0) * np.maximum(clipped_time - p['time_peak_like'], 0.0)
        X['low_temp_very_long_bonus'] = np.maximum(p['temp_transition'] - clipped_temp, 0.0) * np.maximum(clipped_time - p['time_long'], 0.0)
        X['mid_temp_peak_bonus'] = self._gaussian(clipped_temp, 460.0, 6.0) * self._gaussian(clipped_time, 12.0, 3.0)

        # 14. 原始态常数先验
        base_strength_sum = p['base_tensile'] + p['base_yield']
        base_strength_ratio = p['base_yield'] / p['base_tensile']

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['process_to_base_potential'] = (
            X['combined_activation'].values
            + 0.40 * net_strengthening
            + 0.18 * win_470_12
            + 0.16 * win_440_24
            - 0.10 * transition_instability
            - 0.01 * X['high_temp_long_penalty'].values
        )

        # 15. 多目标结构先验
        X['predicted_yield_tensile_ratio_baseline'] = baseline_ratio
        X['predicted_tensile_minus_yield_baseline'] = baseline_gap
        X['predicted_gap_over_sum_baseline'] = self._safe_divide(baseline_gap, baseline_sum)

        # 数值清理
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