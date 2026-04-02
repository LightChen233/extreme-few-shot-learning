import pandas as pd
import numpy as np


# 7499铝合金热处理-性能预测的物理启发参数
# 依据：
# 1) 7499 属于高强 Al-Zn-Mg-Cu 系，强烈受温度-时间耦合控制；
# 2) 440/460/470°C 与 1/12/24h 落在“高温处理/组织快速演化”区间；
# 3) 高温下扩散、固溶、析出相粗化、晶界变化会导致强度存在明显非线性拐点；
# 4) 当前最大误差集中于 470×12h、460×12h、440×24h，说明这些点可能对应不同机制窗口。
DOMAIN_PARAMS = {
    # -------------------------
    # 基础常数
    # -------------------------
    'R': 8.314,                  # J/(mol·K), 气体常数
    'Q_diff': 115000.0,          # J/mol, 铝合金中高温扩散/组织演化的经验活化能量级
    'Q_alt': 128000.0,           # J/mol, 较高激活能，用于增强高温敏感性
    'arr_scale': 1e14,           # Arrhenius量缩放，避免数值过小
    'arr_scale_alt': 1e16,

    # -------------------------
    # 原始凝固态基准性能
    # -------------------------
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # -------------------------
    # 温度机制边界（℃）
    # 440: 低端高温区
    # 460: 中高温敏感区
    # 470: 高温快速演化区
    # -------------------------
    'temp_low_boundary': 445.0,
    'temp_mid_boundary': 455.0,
    'temp_high_boundary': 465.0,
    'temp_overheat_boundary': 472.0,

    # -------------------------
    # 时间机制边界（h）
    # 1h: 短时
    # 12h: 中时/敏感平台
    # 24h: 长时/深度演化
    # -------------------------
    'time_short_boundary': 2.0,
    'time_mid_boundary': 8.0,
    'time_sensitive_boundary': 12.0,
    'time_long_boundary': 16.0,
    'time_overage_boundary': 20.0,

    # -------------------------
    # 强化/软化窗口中心
    # -------------------------
    'win_440_temp': 440.0,
    'win_460_temp': 460.0,
    'win_470_temp': 470.0,
    'win_1_time': 1.0,
    'win_12_time': 12.0,
    'win_24_time': 24.0,

    # -------------------------
    # 高斯窗口宽度
    # -------------------------
    'temp_sigma_narrow': 5.5,
    'temp_sigma_wide': 9.0,
    'log_time_sigma_narrow': 0.18,
    'log_time_sigma_mid': 0.28,
    'log_time_sigma_wide': 0.42,

    # -------------------------
    # 基线预测的最大增益/损失幅度
    # 只体现世界知识量级，不直接拟合标签
    # -------------------------
    'max_tensile_gain': 235.0,
    'max_yield_gain': 200.0,
    'max_strain_drop': 3.2,
    'max_strain_recovery': 4.2,

    # -------------------------
    # 敏感窗口
    # 当前误差显示：
    # 470×12h：模型常低估 -> 可能是高温中时窗口强化跃迁
    # 460×12h：模型出现高估 -> 可能是竞争机制区，非单调
    # 440×24h：模型低估 -> 低温长时可能形成充分均匀化/析出有利状态
    # -------------------------
    'hot_mid_temp_center': 468.0,
    'hot_mid_time_center': 12.0,
    'mid_competition_temp_center': 460.0,
    'mid_competition_time_center': 12.0,
    'low_long_temp_center': 440.0,
    'low_long_time_center': 24.0,

    # -------------------------
    # 数值稳定
    # -------------------------
    'eps': 1e-12,
}


class FeatureAgent:
    """基于材料热处理机理的动态特征工程"""

    def __init__(self):
        self.feature_names = []
        self.params = DOMAIN_PARAMS

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < self.params['eps'], self.params['eps'], b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def _gaussian(self, x, center, sigma):
        sigma = max(float(sigma), self.params['eps'])
        return np.exp(-((x - center) / sigma) ** 2)

    def physics_baseline(self, temp, time):
        """
        基于领域知识的粗粒度物理基线：
        1) 温度升高 -> 扩散与组织演化加速；
        2) 时间延长 -> 强化先增强后可能进入粗化/过度演化；
        3) 470×12h 可能处于高温敏感强化窗口；
        4) 460×12h 可能处于强化与软化竞争区；
        5) 440×24h 可能处于低温长时充分演化窗口；
        6) 强度提高通常伴随塑性下降，但软化/回复会使应变回升。
        """
        p = self.params

        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)
        time = np.clip(time, 0, None)
        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(time)

        # -------------------------
        # 扩散/热剂量特征
        # -------------------------
        arr = np.exp(-p['Q_diff'] / (p['R'] * np.clip(temp_k, 1.0, None)))
        arr_alt = np.exp(-p['Q_alt'] / (p['R'] * np.clip(temp_k, 1.0, None)))

        thermal_dose = time * arr * p['arr_scale']
        thermal_dose_alt = sqrt_time * arr_alt * p['arr_scale_alt']

        log_dose = np.log1p(np.clip(thermal_dose, 0, None))
        log_dose_alt = np.log1p(np.clip(thermal_dose_alt, 0, None))

        # -------------------------
        # 温度/时间激活
        # -------------------------
        temp_act = self._sigmoid((temp - p['temp_low_boundary']) / 5.5)
        temp_high_act = self._sigmoid((temp - p['temp_high_boundary']) / 4.0)
        temp_overheat_act = self._sigmoid((temp - p['temp_overheat_boundary']) / 2.5)

        time_act = 1.0 - np.exp(-time / 5.0)
        time_mid_act = self._sigmoid((time - p['time_mid_boundary']) / 2.0)
        time_long_act = self._sigmoid((time - p['time_long_boundary']) / 2.4)
        time_overage_act = self._sigmoid((time - p['time_overage_boundary']) / 1.8)

        # -------------------------
        # 窗口机制
        # -------------------------
        win_470_12 = (
            self._gaussian(temp, p['hot_mid_temp_center'], p['temp_sigma_wide']) *
            self._gaussian(log_time, np.log1p(p['hot_mid_time_center']), p['log_time_sigma_mid'])
        )
        win_460_12 = (
            self._gaussian(temp, p['mid_competition_temp_center'], p['temp_sigma_wide']) *
            self._gaussian(log_time, np.log1p(p['mid_competition_time_center']), p['log_time_sigma_mid'])
        )
        win_440_24 = (
            self._gaussian(temp, p['low_long_temp_center'], p['temp_sigma_wide']) *
            self._gaussian(log_time, np.log1p(p['low_long_time_center']), p['log_time_sigma_mid'])
        )

        # -------------------------
        # 强化驱动力
        # -------------------------
        general_strength = (
            0.28 * temp_act +
            0.22 * time_act +
            0.16 * np.tanh(log_dose / 3.0) +
            0.10 * np.tanh(log_dose_alt / 2.5)
        )

        # 470×12h：高温中时强化峰附近
        hot_mid_boost = 0.36 * win_470_12

        # 440×24h：低温长时充分演化
        low_long_boost = 0.30 * win_440_24

        # 460×12h：竞争区，给基线少量“折返/抑制”
        competition_penalty = 0.16 * win_460_12 * (0.55 + 0.45 * temp_high_act)

        # 高温长时软化/粗化风险
        softening = (
            temp_high_act * (0.26 * time_mid_act + 0.42 * time_long_act) +
            0.18 * temp_overheat_act * (0.60 * time_mid_act + 0.80 * time_overage_act)
        )

        net_strength = general_strength + hot_mid_boost + low_long_boost - competition_penalty - 0.22 * softening
        net_strength = np.clip(net_strength, 0.0, 1.25)

        # -------------------------
        # 强度基线
        # -------------------------
        baseline_tensile = p['base_tensile'] + p['max_tensile_gain'] * net_strength

        baseline_yield_factor = (
            0.93 * general_strength +
            0.30 * win_470_12 +
            0.26 * win_440_24 -
            0.18 * win_460_12 -
            0.24 * softening
        )
        baseline_yield_factor = np.clip(baseline_yield_factor, 0.0, 1.20)
        baseline_yield = p['base_yield'] + p['max_yield_gain'] * baseline_yield_factor

        # 屈服一般不高于抗拉，保留最小间隔
        baseline_yield = np.minimum(baseline_yield, baseline_tensile - 8.0)

        # -------------------------
        # 应变基线
        # 强化增强 -> 应变下降
        # 长时/高温回复软化 -> 应变回升
        # -------------------------
        strain_drop = p['max_strain_drop'] * (
            0.68 * general_strength +
            0.18 * win_470_12 +
            0.14 * win_440_24
        )

        strain_recovery = p['max_strain_recovery'] * (
            0.42 * softening +
            0.10 * time_long_act +
            0.08 * temp_overheat_act
        )

        # 460×12h 竞争区，可能强度不及470×12h，但塑性略保留
        competition_recovery = 0.55 * win_460_12

        baseline_strain = p['base_strain'] - strain_drop + strain_recovery + competition_recovery

        # 合理裁剪
        baseline_tensile = np.clip(baseline_tensile, 80.0, 650.0)
        baseline_yield = np.clip(baseline_yield, 50.0, 600.0)
        baseline_strain = np.clip(baseline_strain, 1.0, 20.0)

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)
        p = self.params

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        time = np.clip(time, 0, None)

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(time)
        inv_temp_k = 1.0 / np.clip(temp_k, p['eps'], None)

        # 1. 原始与基础变换
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['temp_cube'] = temp ** 3
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp'] = self._safe_divide(time, temp)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # 2. Arrhenius / 热剂量
        arr = np.exp(-p['Q_diff'] / (p['R'] * np.clip(temp_k, 1.0, None)))
        arr_alt = np.exp(-p['Q_alt'] / (p['R'] * np.clip(temp_k, 1.0, None)))
        thermal_dose = time * arr * p['arr_scale']
        thermal_dose_alt = sqrt_time * arr_alt * p['arr_scale_alt']
        log_thermal_dose = np.log1p(np.clip(thermal_dose, 0, None))
        log_thermal_dose_alt = np.log1p(np.clip(thermal_dose_alt, 0, None))

        X['arrhenius'] = arr
        X['arrhenius_alt'] = arr_alt
        X['thermal_dose'] = thermal_dose
        X['thermal_dose_alt'] = thermal_dose_alt
        X['log_thermal_dose'] = log_thermal_dose
        X['log_thermal_dose_alt'] = log_thermal_dose_alt
        X['temp_x_log_thermal_dose'] = temp * log_thermal_dose
        X['temp_x_log_thermal_dose_alt'] = temp * log_thermal_dose_alt
        X['time_x_arrhenius'] = time * arr
        X['sqrt_time_x_arrhenius_alt'] = sqrt_time * arr_alt

        # 3. 物理基线
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

        # 4. 机制区间检测
        is_temp_low = (temp <= p['temp_low_boundary']).astype(float)
        is_temp_mid = ((temp > p['temp_low_boundary']) & (temp < p['temp_high_boundary'])).astype(float)
        is_temp_high = (temp >= p['temp_high_boundary']).astype(float)
        is_temp_overheat = (temp >= p['temp_overheat_boundary']).astype(float)

        is_time_short = (time <= p['time_short_boundary']).astype(float)
        is_time_mid = ((time > p['time_short_boundary']) & (time < p['time_long_boundary'])).astype(float)
        is_time_sensitive = (time >= p['time_sensitive_boundary']).astype(float)
        is_time_long = (time >= p['time_long_boundary']).astype(float)
        is_time_overage = (time >= p['time_overage_boundary']).astype(float)

        X['is_temp_low'] = is_temp_low
        X['is_temp_mid'] = is_temp_mid
        X['is_temp_high'] = is_temp_high
        X['is_temp_overheat'] = is_temp_overheat

        X['is_time_short'] = is_time_short
        X['is_time_mid'] = is_time_mid
        X['is_time_sensitive'] = is_time_sensitive
        X['is_time_long'] = is_time_long
        X['is_time_overage'] = is_time_overage

        # 5. 相对边界残差特征
        X['temp_relative_low_boundary'] = temp - p['temp_low_boundary']
        X['temp_relative_mid_boundary'] = temp - p['temp_mid_boundary']
        X['temp_relative_high_boundary'] = temp - p['temp_high_boundary']
        X['temp_relative_overheat_boundary'] = temp - p['temp_overheat_boundary']

        X['time_relative_short_boundary'] = time - p['time_short_boundary']
        X['time_relative_mid_boundary'] = time - p['time_mid_boundary']
        X['time_relative_sensitive_boundary'] = time - p['time_sensitive_boundary']
        X['time_relative_long_boundary'] = time - p['time_long_boundary']
        X['time_relative_overage_boundary'] = time - p['time_overage_boundary']

        X['relu_above_temp_low'] = np.maximum(temp - p['temp_low_boundary'], 0.0)
        X['relu_above_temp_mid'] = np.maximum(temp - p['temp_mid_boundary'], 0.0)
        X['relu_above_temp_high'] = np.maximum(temp - p['temp_high_boundary'], 0.0)
        X['relu_above_temp_overheat'] = np.maximum(temp - p['temp_overheat_boundary'], 0.0)
        X['relu_below_temp_low'] = np.maximum(p['temp_low_boundary'] - temp, 0.0)

        X['relu_above_time_mid'] = np.maximum(time - p['time_mid_boundary'], 0.0)
        X['relu_above_time_sensitive'] = np.maximum(time - p['time_sensitive_boundary'], 0.0)
        X['relu_above_time_long'] = np.maximum(time - p['time_long_boundary'], 0.0)
        X['relu_above_time_overage'] = np.maximum(time - p['time_overage_boundary'], 0.0)
        X['relu_below_time_short'] = np.maximum(p['time_short_boundary'] - time, 0.0)

        # 6. 激活与竞争特征
        temp_act = self._sigmoid((temp - p['temp_low_boundary']) / 5.5)
        temp_high_act = self._sigmoid((temp - p['temp_high_boundary']) / 4.0)
        temp_overheat_act = self._sigmoid((temp - p['temp_overheat_boundary']) / 2.5)
        time_act = 1.0 - np.exp(-time / 5.0)
        time_mid_act = self._sigmoid((time - p['time_mid_boundary']) / 2.0)
        time_long_act = self._sigmoid((time - p['time_long_boundary']) / 2.4)
        time_overage_act = self._sigmoid((time - p['time_overage_boundary']) / 1.8)

        X['temp_activation'] = temp_act
        X['temp_high_activation'] = temp_high_act
        X['temp_overheat_activation'] = temp_overheat_act
        X['time_activation'] = time_act
        X['time_mid_activation'] = time_mid_act
        X['time_long_activation'] = time_long_act
        X['time_overage_activation'] = time_overage_act

        # 7. 敏感工况窗口
        win_440_24 = self._gaussian(temp, p['win_440_temp'], p['temp_sigma_narrow']) * \
                     self._gaussian(log_time, np.log1p(p['win_24_time']), p['log_time_sigma_narrow'])
        win_470_12 = self._gaussian(temp, p['win_470_temp'], p['temp_sigma_narrow']) * \
                     self._gaussian(log_time, np.log1p(p['win_12_time']), p['log_time_sigma_narrow'])
        win_460_12 = self._gaussian(temp, p['win_460_temp'], p['temp_sigma_narrow']) * \
                     self._gaussian(log_time, np.log1p(p['win_12_time']), p['log_time_sigma_narrow'])
        win_440_1 = self._gaussian(temp, p['win_440_temp'], p['temp_sigma_narrow']) * \
                    self._gaussian(log_time, np.log1p(p['win_1_time']), p['log_time_sigma_narrow'])

        X['window_440_24'] = win_440_24
        X['window_470_12'] = win_470_12
        X['window_460_12'] = win_460_12
        X['window_440_1'] = win_440_1

        # 8. 组织机制驱动力
        strengthening_drive = (
            0.24 * temp_act +
            0.20 * time_act +
            0.10 * np.tanh(log_thermal_dose / 3.0) +
            0.08 * np.tanh(log_thermal_dose_alt / 2.5) +
            0.30 * win_470_12 +
            0.22 * win_440_24
        )

        competition_drive = (
            0.20 * win_460_12 +
            0.10 * temp_high_act * time_mid_act
        )

        softening_drive = (
            temp_high_act * (0.18 * time_mid_act + 0.36 * time_long_act) +
            temp_overheat_act * (0.14 * time_mid_act + 0.28 * time_overage_act)
        )

        net_strengthening = strengthening_drive - competition_drive - softening_drive

        X['strengthening_drive'] = strengthening_drive
        X['competition_drive'] = competition_drive
        X['softening_drive'] = softening_drive
        X['net_strengthening'] = net_strengthening

        # 9. 分区耦合
        X['temp_low_x_log_time'] = is_temp_low * temp * log_time
        X['temp_mid_x_log_time'] = is_temp_mid * temp * log_time
        X['temp_high_x_log_time'] = is_temp_high * temp * log_time

        X['time_short_x_temp'] = is_time_short * time * temp
        X['time_mid_x_temp'] = is_time_mid * time * temp
        X['time_long_x_temp'] = is_time_long * time * temp

        X['high_temp_long_time_coupling'] = is_temp_high * is_time_long * temp * log_time
        X['low_temp_very_long_coupling'] = is_temp_low * is_time_overage * temp * log_time
        X['overheat_sensitive_coupling'] = is_temp_overheat * is_time_sensitive * temp * log_time
        X['competition_zone_coupling'] = win_460_12 * temp * log_time
        X['hot_peak_zone_coupling'] = win_470_12 * temp * log_time
        X['low_long_zone_coupling'] = win_440_24 * temp * log_time

        # 10. 基线相关残差型特征
        X['baseline_tensile_x_temp_relative_high'] = baseline_tensile * (temp - p['temp_high_boundary'])
        X['baseline_yield_x_time_relative_long'] = baseline_yield * (time - p['time_long_boundary'])
        X['baseline_gap_x_window_470_12'] = baseline_gap * win_470_12
        X['baseline_gap_x_window_460_12'] = baseline_gap * win_460_12
        X['baseline_gap_x_window_440_24'] = baseline_gap * win_440_24
        X['baseline_strength_sum_x_net_strength'] = baseline_sum * net_strengthening
        X['baseline_strain_x_softening'] = baseline_strain * softening_drive
        X['baseline_strain_x_competition'] = baseline_strain * competition_drive
        X['baseline_tensile_x_strengthening'] = baseline_tensile * strengthening_drive
        X['baseline_yield_x_strengthening'] = baseline_yield * strengthening_drive
        X['baseline_yield_x_softening'] = baseline_yield * softening_drive
        X['baseline_tensile_minus_base'] = baseline_tensile - p['base_tensile']
        X['baseline_yield_minus_base'] = baseline_yield - p['base_yield']
        X['baseline_strain_minus_base'] = baseline_strain - p['base_strain']

        # 11. 强塑权衡
        ductility_recovery = (
            0.50 * softening_drive +
            0.16 * is_time_overage +
            0.12 * is_temp_overheat +
            0.15 * win_460_12 -
            0.24 * strengthening_drive
        )
        strength_ductility_tradeoff = net_strengthening - ductility_recovery

        X['ductility_recovery'] = ductility_recovery
        X['strength_ductility_tradeoff'] = strength_ductility_tradeoff
        X['predicted_strength_ductility_coupling'] = self._safe_divide(baseline_sum, baseline_strain + 1e-6)
        X['predicted_gap_over_sum_baseline'] = self._safe_divide(baseline_gap, baseline_sum)

        # 12. 围绕误差热点的距离特征
        X['dist_temp_to_440'] = np.abs(temp - p['win_440_temp'])
        X['dist_temp_to_460'] = np.abs(temp - p['win_460_temp'])
        X['dist_temp_to_470'] = np.abs(temp - p['win_470_temp'])
        X['dist_time_to_1'] = np.abs(time - p['win_1_time'])
        X['dist_time_to_12'] = np.abs(time - p['win_12_time'])
        X['dist_time_to_24'] = np.abs(time - p['win_24_time'])

        X['signed_temp_from_455'] = temp - 455.0
        X['signed_temp_from_465'] = temp - 465.0
        X['signed_time_from_12'] = time - 12.0
        X['signed_time_from_24'] = time - 24.0
        X['signed_temp_from_455_x_signed_time_from_12'] = (temp - 455.0) * (time - 12.0)
        X['signed_temp_from_465_x_signed_time_from_12'] = (temp - 465.0) * (time - 12.0)

        # 13. 饱和型/惩罚型特征
        clipped_temp = np.clip(temp, 430.0, 480.0)
        clipped_time = np.clip(time, 0.0, 30.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['combined_activation'] = X['time_saturation'].values * temp_act
        X['high_temp_long_penalty'] = np.maximum(clipped_temp - p['temp_high_boundary'], 0.0) * np.maximum(clipped_time - p['time_mid_boundary'], 0.0)
        X['overheat_penalty'] = np.maximum(clipped_temp - p['temp_overheat_boundary'], 0.0) * np.maximum(clipped_time - p['time_sensitive_boundary'], 0.0)
        X['low_temp_very_long_bonus'] = np.maximum(p['temp_mid_boundary'] - clipped_temp, 0.0) * np.maximum(clipped_time - p['time_long_boundary'], 0.0)

        # 14. 原始态先验
        base_strength_sum = p['base_tensile'] + p['base_yield']
        base_strength_ratio = p['base_yield'] / p['base_tensile']

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['process_to_base_potential'] = (
            X['combined_activation'].values +
            0.40 * net_strengthening +
            0.18 * win_470_12 +
            0.14 * win_440_24 -
            0.12 * win_460_12 -
            0.008 * X['high_temp_long_penalty'].values
        )

        # 15. 多目标关联先验
        X['predicted_yield_tensile_ratio_baseline'] = baseline_ratio
        X['predicted_tensile_minus_yield_baseline'] = baseline_gap
        X['predicted_gap_over_strain'] = self._safe_divide(baseline_gap, baseline_strain + 1e-6)
        X['predicted_sum_over_strain'] = self._safe_divide(baseline_sum, baseline_strain + 1e-6)

        # 16. 对热点工艺更敏感的组合特征
        X['win47012_x_baseline_tensile'] = win_470_12 * baseline_tensile
        X['win47012_x_baseline_yield'] = win_470_12 * baseline_yield
        X['win46012_x_baseline_tensile'] = win_460_12 * baseline_tensile
        X['win46012_x_baseline_yield'] = win_460_12 * baseline_yield
        X['win44024_x_baseline_tensile'] = win_440_24 * baseline_tensile
        X['win44024_x_baseline_yield'] = win_440_24 * baseline_yield
        X['win4401_x_baseline_strain'] = win_440_1 * baseline_strain

        X['competition_minus_hotpeak'] = win_460_12 - win_470_12
        X['hotpeak_minus_lowlong'] = win_470_12 - win_440_24
        X['lowlong_minus_short'] = win_440_24 - win_440_1

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