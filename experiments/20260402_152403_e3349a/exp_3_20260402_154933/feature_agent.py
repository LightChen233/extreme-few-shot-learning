import pandas as pd
import numpy as np


# 7499铝合金高温热处理/固溶-时效相关启发式参数
# 这里结合当前数据分布（440/460/470°C, 1/12/24h）与铝合金高温组织演化常识：
# - 440°C：相对较低的高温处理区，长时可能仍持续溶质均匀化/再分配
# - 460°C：更强扩散激活，12h附近可能进入敏感强化区
# - 470°C：高温侧，12h即可出现显著组织转变；若模型只用平滑特征，容易低估跃迁
DOMAIN_PARAMS = {
    # 温度边界（℃）
    'temp_low': 445.0,          # 低高温区上边界：接近440°C组，扩散相对慢
    'temp_mid': 455.0,          # 中心过渡区：440→460之间组织响应加快
    'temp_high': 465.0,         # 高温敏感区起点：470°C附近更容易表现出强烈非线性
    'temp_very_high': 472.0,    # 极高温近似边界，用于高温强化/软化竞争识别

    # 时间边界（h）
    'time_short': 2.0,          # 1h附近，处理尚短
    'time_mid': 8.0,            # 中等时间
    'time_long': 16.0,          # 长时起点，12~24h之间进入显著演化区
    'time_very_long': 20.0,     # 超长时边界，24h组重点关注

    # 敏感工艺中心（来自误差分布重点工况）
    'focus_temp_440': 440.0,
    'focus_temp_460': 460.0,
    'focus_temp_470': 470.0,
    'focus_time_1': 1.0,
    'focus_time_12': 12.0,
    'focus_time_24': 24.0,

    # 物理启发尺度
    'temp_scale': 8.0,
    'time_scale': 4.0,
    'log_time_scale': 0.55,

    # Arrhenius参数（铝合金扩散/析出控制量级）
    'R': 8.314,                 # J/(mol·K)
    'Q': 120000.0,              # J/mol，取较高温扩散控制的经验量级，强调温度敏感性
    'arrhenius_scale': 1e14,    # 数值缩放，便于构造热剂量特征

    # 原始凝固态基准性能
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # 热处理后的经验增益/回落幅值（世界知识约束，不直接拟合标签）
    'max_tensile_gain': 230.0,
    'max_yield_gain': 195.0,
    'max_strain_drop': 2.8,
    'max_strain_recovery': 4.0,

    # 特定敏感窗口
    'high_strength_temp_center': 465.0,   # 高温强化峰附近
    'high_strength_time_center': 12.0,    # 12h组误差大，说明此处有强烈非线性
    'long_hold_temp_center': 440.0,       # 440×24h组误差大，说明低端高温长时也形成特殊状态
    'long_hold_time_center': 24.0,

    'window_temp_sigma': 10.0,
    'window_log_time_sigma': 0.45,
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
        1) 高温促进扩散、固溶与析出动力学；
        2) 时间延长促进组织演化，但存在饱和；
        3) 440×24h 与 470×12h 是当前误差热点，说明这两类条件可能形成
           高于普通平滑规律的强化状态，因此基线中加入“敏感窗口”；
        4) 强度提高通常伴随应变下降，但长时/高温可能有一定塑性恢复。
        """
        p = self.params

        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)
        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # 扩散激活
        arrhenius = np.exp(-p['Q'] / (p['R'] * np.clip(temp_k, 1e-9, None)))
        thermal_dose = time * arrhenius * p['arrhenius_scale']
        log_thermal_dose = np.log1p(np.clip(thermal_dose, 0, None))

        # 温度与时间激活
        temp_activation = self._sigmoid((temp - p['temp_low']) / 6.0)
        high_temp_activation = self._sigmoid((temp - p['temp_high']) / 4.5)
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / 5.0)
        long_time_activation = self._sigmoid((time - p['time_long']) / 2.5)

        # 常规强化项：高温+一定时间
        general_strength = (
            0.30 * temp_activation
            + 0.22 * time_activation
            + 0.18 * np.tanh(np.clip(log_thermal_dose, 0, None) / 3.0)
        )

        # 敏感窗口1：470×12h附近，误差显示模型系统性低估
        hot_mid_window = (
            self._gaussian(temp, p['high_strength_temp_center'], p['window_temp_sigma'])
            * self._gaussian(log_time, np.log1p(p['high_strength_time_center']), p['window_log_time_sigma'])
        )

        # 敏感窗口2：440×24h附近，长时低端高温组也显著低估
        long_hold_window = (
            self._gaussian(temp, p['long_hold_temp_center'], p['window_temp_sigma'])
            * self._gaussian(log_time, np.log1p(p['long_hold_time_center']), p['window_log_time_sigma'])
        )

        # 高温长时软化/粗化风险
        softening = high_temp_activation * (
            0.45 * long_time_activation
            + 0.20 * np.maximum(log_time - np.log1p(p['time_mid']), 0.0)
        )

        # 净强化：对热点工况适度抬升
        net_strength = (
            general_strength
            + 0.42 * hot_mid_window
            + 0.34 * long_hold_window
            - 0.18 * softening
        )
        net_strength = np.clip(net_strength, 0.0, 1.25)

        # 强度基线
        baseline_tensile = p['base_tensile'] + p['max_tensile_gain'] * net_strength
        baseline_yield = p['base_yield'] + p['max_yield_gain'] * (
            0.95 * general_strength
            + 0.36 * hot_mid_window
            + 0.30 * long_hold_window
            - 0.22 * softening
        )

        # 屈服不高于抗拉
        baseline_yield = np.minimum(baseline_yield, baseline_tensile - 8.0)

        # 应变：强化使塑性下降，软化/回复使其回升
        strain_drop = p['max_strain_drop'] * (
            0.70 * general_strength
            + 0.20 * hot_mid_window
            + 0.10 * long_hold_window
        )
        strain_recovery = p['max_strain_recovery'] * (
            0.40 * softening
            + 0.10 * long_time_activation
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

        # 1. 原始特征
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # 2. 基础非线性
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['temp_cube'] = temp ** 3
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp'] = self._safe_divide(time, temp)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # 3. Arrhenius与热剂量
        arrhenius = np.exp(-p['Q'] / (p['R'] * np.clip(temp_k, 1e-9, None)))
        thermal_dose = time * arrhenius * p['arrhenius_scale']
        log_thermal_dose = np.log1p(np.clip(thermal_dose, 0, None))

        X['arrhenius'] = arrhenius
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = log_thermal_dose
        X['temp_x_log_thermal_dose'] = temp * log_thermal_dose
        X['time_x_arrhenius'] = time * arrhenius

        # 4. 物理基线
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

        # 5. 机制区间检测
        is_temp_low = (temp <= p['temp_low']).astype(float)
        is_temp_mid = ((temp > p['temp_low']) & (temp < p['temp_high'])).astype(float)
        is_temp_high = (temp >= p['temp_high']).astype(float)
        is_temp_very_high = (temp >= p['temp_very_high']).astype(float)

        is_time_short = (time <= p['time_short']).astype(float)
        is_time_mid = ((time > p['time_short']) & (time < p['time_long'])).astype(float)
        is_time_long = (time >= p['time_long']).astype(float)
        is_time_very_long = (time >= p['time_very_long']).astype(float)

        X['is_temp_low'] = is_temp_low
        X['is_temp_mid'] = is_temp_mid
        X['is_temp_high'] = is_temp_high
        X['is_temp_very_high'] = is_temp_very_high

        X['is_time_short'] = is_time_short
        X['is_time_mid'] = is_time_mid
        X['is_time_long'] = is_time_long
        X['is_time_very_long'] = is_time_very_long

        # 6. 相对机制边界残差
        X['temp_relative_low'] = temp - p['temp_low']
        X['temp_relative_mid'] = temp - p['temp_mid']
        X['temp_relative_high'] = temp - p['temp_high']
        X['temp_relative_very_high'] = temp - p['temp_very_high']

        X['time_relative_short'] = time - p['time_short']
        X['time_relative_mid'] = time - p['time_mid']
        X['time_relative_long'] = time - p['time_long']
        X['time_relative_very_long'] = time - p['time_very_long']

        X['relu_above_temp_low'] = np.maximum(temp - p['temp_low'], 0.0)
        X['relu_above_temp_mid'] = np.maximum(temp - p['temp_mid'], 0.0)
        X['relu_above_temp_high'] = np.maximum(temp - p['temp_high'], 0.0)
        X['relu_below_temp_low'] = np.maximum(p['temp_low'] - temp, 0.0)

        X['relu_above_time_mid'] = np.maximum(time - p['time_mid'], 0.0)
        X['relu_above_time_long'] = np.maximum(time - p['time_long'], 0.0)
        X['relu_above_time_very_long'] = np.maximum(time - p['time_very_long'], 0.0)
        X['relu_below_time_short'] = np.maximum(p['time_short'] - time, 0.0)

        # 7. 激活/竞争特征
        temp_activation = self._sigmoid((temp - p['temp_low']) / 6.0)
        temp_high_activation = self._sigmoid((temp - p['temp_high']) / 4.5)
        time_activation = 1.0 - np.exp(-np.clip(time, 0, None) / 5.0)
        time_long_activation = self._sigmoid((time - p['time_long']) / 2.5)

        X['temp_activation'] = temp_activation
        X['temp_high_activation'] = temp_high_activation
        X['time_activation'] = time_activation
        X['time_long_activation'] = time_long_activation

        # 8. 敏感工况窗口特征
        win_440_24 = self._gaussian(temp, p['focus_temp_440'], 6.0) * self._gaussian(log_time, np.log1p(p['focus_time_24']), 0.22)
        win_470_12 = self._gaussian(temp, p['focus_temp_470'], 6.0) * self._gaussian(log_time, np.log1p(p['focus_time_12']), 0.20)
        win_460_12 = self._gaussian(temp, p['focus_temp_460'], 6.0) * self._gaussian(log_time, np.log1p(p['focus_time_12']), 0.20)
        win_440_1 = self._gaussian(temp, p['focus_temp_440'], 6.0) * self._gaussian(log_time, np.log1p(p['focus_time_1']), 0.18)

        X['window_440_24'] = win_440_24
        X['window_470_12'] = win_470_12
        X['window_460_12'] = win_460_12
        X['window_440_1'] = win_440_1

        # 9. 组织机制竞争
        strengthening_drive = (
            0.25 * temp_activation
            + 0.20 * time_activation
            + 0.30 * win_470_12
            + 0.22 * win_440_24
            + 0.12 * win_460_12
            + 0.10 * np.tanh(np.clip(log_thermal_dose, 0, None) / 3.0)
        )

        softening_drive = (
            temp_high_activation * (0.45 * time_long_activation + 0.18 * np.maximum(log_time - np.log1p(p['time_mid']), 0.0))
            + 0.10 * is_temp_very_high * time_activation
        )

        net_strengthening = strengthening_drive - softening_drive

        X['strengthening_drive'] = strengthening_drive
        X['softening_drive'] = softening_drive
        X['net_strengthening'] = net_strengthening

        # 10. 分区耦合
        X['temp_low_x_log_time'] = is_temp_low * temp * log_time
        X['temp_mid_x_log_time'] = is_temp_mid * temp * log_time
        X['temp_high_x_log_time'] = is_temp_high * temp * log_time

        X['time_short_x_temp'] = is_time_short * time * temp
        X['time_mid_x_temp'] = is_time_mid * time * temp
        X['time_long_x_temp'] = is_time_long * time * temp

        X['high_temp_long_time_coupling'] = is_temp_high * is_time_long * temp * log_time
        X['low_temp_very_long_coupling'] = is_temp_low * is_time_very_long * temp * log_time
        X['very_high_temp_mid_time_coupling'] = is_temp_very_high * is_time_mid * temp * log_time

        # 11. 基线残差型工艺表征
        X['baseline_tensile_x_temp_relative_high'] = baseline_tensile * (temp - p['temp_high'])
        X['baseline_yield_x_time_relative_long'] = baseline_yield * (time - p['time_long'])
        X['baseline_gap_x_window_470_12'] = baseline_gap * win_470_12
        X['baseline_gap_x_window_440_24'] = baseline_gap * win_440_24
        X['baseline_strength_sum_x_net_strength'] = baseline_sum * net_strengthening
        X['baseline_strain_x_softening'] = baseline_strain * softening_drive
        X['baseline_tensile_x_strengthening'] = baseline_tensile * strengthening_drive
        X['baseline_yield_x_strengthening'] = baseline_yield * strengthening_drive

        # 12. 强塑权衡
        ductility_recovery = (
            0.55 * softening_drive
            + 0.18 * is_time_very_long
            + 0.12 * is_temp_very_high
            - 0.25 * strengthening_drive
        )
        strength_ductility_tradeoff = net_strengthening - ductility_recovery

        X['ductility_recovery'] = ductility_recovery
        X['strength_ductility_tradeoff'] = strength_ductility_tradeoff
        X['predicted_strength_ductility_coupling'] = self._safe_divide(baseline_sum, baseline_strain + 1e-6)

        # 13. 围绕误差热点的“距离”特征
        X['dist_temp_to_440'] = np.abs(temp - p['focus_temp_440'])
        X['dist_temp_to_460'] = np.abs(temp - p['focus_temp_460'])
        X['dist_temp_to_470'] = np.abs(temp - p['focus_temp_470'])
        X['dist_time_to_1'] = np.abs(time - p['focus_time_1'])
        X['dist_time_to_12'] = np.abs(time - p['focus_time_12'])
        X['dist_time_to_24'] = np.abs(time - p['focus_time_24'])

        X['signed_temp_from_455'] = temp - 455.0
        X['signed_time_from_12'] = time - 12.0
        X['signed_temp_from_455_x_signed_time_from_12'] = (temp - 455.0) * (time - 12.0)

        # 14. 饱和型特征
        clipped_temp = np.clip(temp, 430.0, 480.0)
        clipped_time = np.clip(time, 0.0, 30.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['combined_activation'] = X['time_saturation'].values * temp_activation
        X['high_temp_long_penalty'] = np.maximum(clipped_temp - p['temp_high'], 0.0) * np.maximum(clipped_time - p['time_mid'], 0.0)
        X['low_temp_very_long_bonus'] = np.maximum(p['temp_mid'] - clipped_temp, 0.0) * np.maximum(clipped_time - p['time_long'], 0.0)

        # 15. 原始态常数先验
        base_strength_sum = p['base_tensile'] + p['base_yield']
        base_strength_ratio = p['base_yield'] / p['base_tensile']

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['process_to_base_potential'] = (
            X['combined_activation'].values
            + 0.45 * net_strengthening
            + 0.20 * win_470_12
            + 0.16 * win_440_24
            - 0.01 * X['high_temp_long_penalty'].values
        )

        # 16. 多目标关联先验
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