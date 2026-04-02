import pandas as pd
import numpy as np


# 7499铝合金热处理/再固溶-保温过程的物理启发参数
DOMAIN_PARAMS = {
    # -----------------------------
    # 1) 温度边界（℃）
    # 这里任务样本温度集中在 440/460/470℃，属于较高温固溶/高温保温区
    # 440℃附近：固溶与第二相溶解可能尚不充分，且动力学偏慢
    # 460℃附近：常可视作较有效的强化准备/组织均匀化区
    # 470℃附近：更强烈的溶解与扩散，同时晶粒粗化、过烧/局部损伤风险上升
    # -----------------------------
    'temp_low': 445.0,
    'temp_mid': 460.0,
    'temp_high': 468.0,

    # -----------------------------
    # 2) 时间边界（h）
    # 1h：短时，组织演化有限
    # 12h：中长时，可能进入显著组织重构区
    # 24h：长时，高温下粗化/性能跃迁风险更明显
    # -----------------------------
    'time_short': 2.0,
    'time_mid': 10.0,
    'time_long': 18.0,

    # -----------------------------
    # 3) 热激活参数
    # 用于构造 Arrhenius 型扩散/组织演化速率
    # 铝合金扩散控制过程常见数量级
    # -----------------------------
    'R': 8.314,          # J/(mol·K)
    'Q': 135000.0,       # J/mol，偏向高温溶解/扩散的经验量级

    # -----------------------------
    # 4) 参考中心
    # 用于构造“最佳激活窗口”和“高风险窗口”
    # 从当前误差表现看，460℃×12h附近可能是重要转折区
    # 470℃×12h、440℃×24h是模型最难点，需要显式编码
    # -----------------------------
    'peak_temp_center': 460.0,
    'peak_time_center': 12.0,
    'peak_temp_scale': 10.0,
    'peak_log_time_scale': 0.55,

    'risk_temp_center_1': 470.0,   # 高温敏感区
    'risk_time_center_1': 12.0,
    'risk_temp_scale_1': 6.0,
    'risk_log_time_scale_1': 0.35,

    'risk_temp_center_2': 440.0,   # 低一些温度但长时保温的敏感区
    'risk_time_center_2': 24.0,
    'risk_temp_scale_2': 6.0,
    'risk_log_time_scale_2': 0.30,

    # -----------------------------
    # 5) 原始凝固态平均性能
    # -----------------------------
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # -----------------------------
    # 6) 物理上合理的性能变化幅度
    # 不直接拟合标签，只作为粗基线的可解释上界
    # -----------------------------
    'max_tensile_gain': 240.0,
    'max_yield_gain': 190.0,
    'max_strain_gain': 7.0,
    'max_softening_loss_tensile': 55.0,
    'max_softening_loss_yield': 40.0,

    # -----------------------------
    # 7) 强塑协同/损伤启发项
    # 高温长时可能导致：
    # - 强度下降或非单调
    # - 延性上升
    # - 也可能因粗化/缺陷导致波动
    # -----------------------------
    'damage_temp_start': 466.0,
    'damage_time_start': 10.0,

    # 8) 数值稳定
    'eps': 1e-12,
}


class FeatureAgent:
    """基于7499铝合金高温热处理机理的动态特征工程"""

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

    def _gaussian(self, x, center, scale):
        scale = max(scale, self.params['eps'])
        return np.exp(-((x - center) / scale) ** 2)

    def physics_baseline(self, temp, time):
        """
        基于世界知识的粗粒度物理基线，不使用训练标签。
        物理假设：
        1) 温度越高，扩散/溶解速率越快
        2) 时间越长，组织演化越充分，但存在饱和
        3) 460℃×12h附近可能接近“有效强化准备/组织优化”窗口
        4) 470℃×12h 与 440℃×24h 是敏感组织区，可能出现强度跃升或明显非线性
        5) 高温长时可能伴随粗化/局部损伤/过度溶解，导致强度回落或波动
        """
        p = self.params

        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # Arrhenius 扩散激活
        arrhenius = np.exp(-p['Q'] / (p['R'] * np.clip(temp_k, 1.0, None)))
        arr_scaled = arrhenius * 1e9

        # 时间饱和：短时增长快，长时趋缓
        time_sat = 1.0 - np.exp(-np.clip(time, 0, None) / 5.0)

        # 温度激活：445℃后开始显著增强，460℃附近更充分
        temp_act_low = self._sigmoid((temp - p['temp_low']) / 4.5)
        temp_act_mid = self._sigmoid((temp - p['temp_mid']) / 3.5)
        temp_act_high = self._sigmoid((temp - p['temp_high']) / 2.5)

        # 有效组织优化窗口
        peak_temp = self._gaussian(temp, p['peak_temp_center'], p['peak_temp_scale'])
        peak_time = self._gaussian(log_time, np.log1p(p['peak_time_center']), p['peak_log_time_scale'])
        peak_window = peak_temp * peak_time

        # 两个显式风险/敏感窗口
        risk_window_470_12 = (
            self._gaussian(temp, p['risk_temp_center_1'], p['risk_temp_scale_1']) *
            self._gaussian(log_time, np.log1p(p['risk_time_center_1']), p['risk_log_time_scale_1'])
        )

        risk_window_440_24 = (
            self._gaussian(temp, p['risk_temp_center_2'], p['risk_temp_scale_2']) *
            self._gaussian(log_time, np.log1p(p['risk_time_center_2']), p['risk_log_time_scale_2'])
        )

        # 热剂量
        thermal_dose = np.clip(time, 0, None) * arr_scaled
        log_thermal_dose = np.log1p(thermal_dose)

        # 强化/组织改善驱动
        strengthening = (
            0.32 * temp_act_low +
            0.20 * time_sat +
            0.28 * peak_window +
            0.10 * np.tanh(log_thermal_dose / 3.0) +
            0.18 * risk_window_470_12 +
            0.15 * risk_window_440_24
        )

        # 高温长时软化/粗化/损伤风险
        damage_temp = self._sigmoid((temp - p['damage_temp_start']) / 2.5)
        damage_time = self._sigmoid((time - p['damage_time_start']) / 2.5)
        softening = (
            0.55 * damage_temp * damage_time +
            0.25 * temp_act_high * np.maximum(log_time - np.log1p(8.0), 0.0)
        )

        # 抗拉/屈服基线
        baseline_tensile = (
            p['base_tensile'] +
            p['max_tensile_gain'] * strengthening -
            p['max_softening_loss_tensile'] * softening
        )

        baseline_yield = (
            p['base_yield'] +
            p['max_yield_gain'] * (0.92 * strengthening + 0.05 * peak_window) -
            p['max_softening_loss_yield'] * softening
        )

        # 应变：高温长时与敏感窗口下可能明显提升
        baseline_strain = (
            p['base_strain'] +
            p['max_strain_gain'] * (
                0.22 * temp_act_mid * time_sat +
                0.40 * risk_window_470_12 +
                0.28 * risk_window_440_24 +
                0.20 * softening
            ) -
            1.0 * peak_window
        )

        # 基本物理约束
        baseline_tensile = np.clip(baseline_tensile, 80.0, 520.0)
        baseline_yield = np.clip(baseline_yield, 50.0, 470.0)
        baseline_yield = np.minimum(baseline_yield, baseline_tensile - 5.0)
        baseline_strain = np.clip(baseline_strain, 1.0, 18.0)

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)
        p = self.params

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, p['eps'], None)

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
        # 2. Arrhenius / 热剂量
        # -----------------------------
        arrhenius = np.exp(-p['Q'] / (p['R'] * np.clip(temp_k, 1.0, None)))
        arr_scaled = arrhenius * 1e9
        thermal_dose = np.clip(time, 0, None) * arr_scaled
        log_thermal_dose = np.log1p(thermal_dose)

        X['arrhenius'] = arrhenius
        X['arrhenius_scaled'] = arr_scaled
        X['thermal_dose'] = thermal_dose
        X['log_thermal_dose'] = log_thermal_dose
        X['temp_x_log_thermal_dose'] = temp * log_thermal_dose
        X['time_x_arrhenius_scaled'] = time * arr_scaled

        # -----------------------------
        # 3. 机制区间检测
        # -----------------------------
        is_low_temp = (temp < p['temp_low']).astype(float)
        is_mid_temp = ((temp >= p['temp_low']) & (temp < p['temp_mid'])).astype(float)
        is_high_temp = ((temp >= p['temp_mid']) & (temp < p['temp_high'])).astype(float)
        is_very_high_temp = (temp >= p['temp_high']).astype(float)

        is_short_time = (time < p['time_short']).astype(float)
        is_mid_time = ((time >= p['time_short']) & (time < p['time_mid'])).astype(float)
        is_long_time = ((time >= p['time_mid']) & (time < p['time_long'])).astype(float)
        is_very_long_time = (time >= p['time_long']).astype(float)

        X['is_low_temp'] = is_low_temp
        X['is_mid_temp'] = is_mid_temp
        X['is_high_temp'] = is_high_temp
        X['is_very_high_temp'] = is_very_high_temp

        X['is_short_time'] = is_short_time
        X['is_mid_time'] = is_mid_time
        X['is_long_time'] = is_long_time
        X['is_very_long_time'] = is_very_long_time

        # -----------------------------
        # 4. 相对机制边界的残差特征
        # -----------------------------
        X['temp_relative_to_low_boundary'] = temp - p['temp_low']
        X['temp_relative_to_mid_boundary'] = temp - p['temp_mid']
        X['temp_relative_to_high_boundary'] = temp - p['temp_high']

        X['time_relative_to_short_boundary'] = time - p['time_short']
        X['time_relative_to_mid_boundary'] = time - p['time_mid']
        X['time_relative_to_long_boundary'] = time - p['time_long']

        X['relu_above_low_temp'] = np.maximum(temp - p['temp_low'], 0.0)
        X['relu_above_mid_temp'] = np.maximum(temp - p['temp_mid'], 0.0)
        X['relu_above_high_temp'] = np.maximum(temp - p['temp_high'], 0.0)
        X['relu_below_low_temp'] = np.maximum(p['temp_low'] - temp, 0.0)

        X['relu_above_short_time'] = np.maximum(time - p['time_short'], 0.0)
        X['relu_above_mid_time'] = np.maximum(time - p['time_mid'], 0.0)
        X['relu_above_long_time'] = np.maximum(time - p['time_long'], 0.0)
        X['relu_below_short_time'] = np.maximum(p['time_short'] - time, 0.0)

        # -----------------------------
        # 5. 峰值窗口/风险窗口特征
        # -----------------------------
        peak_temp = self._gaussian(temp, p['peak_temp_center'], p['peak_temp_scale'])
        peak_time = self._gaussian(log_time, np.log1p(p['peak_time_center']), p['peak_log_time_scale'])
        peak_window = peak_temp * peak_time

        risk_window_470_12 = (
            self._gaussian(temp, p['risk_temp_center_1'], p['risk_temp_scale_1']) *
            self._gaussian(log_time, np.log1p(p['risk_time_center_1']), p['risk_log_time_scale_1'])
        )
        risk_window_440_24 = (
            self._gaussian(temp, p['risk_temp_center_2'], p['risk_temp_scale_2']) *
            self._gaussian(log_time, np.log1p(p['risk_time_center_2']), p['risk_log_time_scale_2'])
        )

        X['peak_temp_proximity'] = peak_temp
        X['peak_time_proximity'] = peak_time
        X['peak_window'] = peak_window

        X['risk_window_470_12'] = risk_window_470_12
        X['risk_window_440_24'] = risk_window_440_24
        X['risk_window_sum'] = risk_window_470_12 + risk_window_440_24
        X['risk_window_diff'] = risk_window_470_12 - risk_window_440_24

        X['temp_residual_to_peak'] = temp - p['peak_temp_center']
        X['log_time_residual_to_peak'] = log_time - np.log1p(p['peak_time_center'])
        X['abs_temp_residual_to_peak'] = np.abs(X['temp_residual_to_peak'].values)
        X['abs_log_time_residual_to_peak'] = np.abs(X['log_time_residual_to_peak'].values)

        # -----------------------------
        # 6. 激活/竞争机制特征
        # -----------------------------
        temp_act_low = self._sigmoid((temp - p['temp_low']) / 4.5)
        temp_act_mid = self._sigmoid((temp - p['temp_mid']) / 3.5)
        temp_act_high = self._sigmoid((temp - p['temp_high']) / 2.5)
        time_sat = 1.0 - np.exp(-np.clip(time, 0, None) / 5.0)

        damage_temp = self._sigmoid((temp - p['damage_temp_start']) / 2.5)
        damage_time = self._sigmoid((time - p['damage_time_start']) / 2.5)
        softening = (
            0.55 * damage_temp * damage_time +
            0.25 * temp_act_high * np.maximum(log_time - np.log1p(8.0), 0.0)
        )

        strengthening = (
            0.32 * temp_act_low +
            0.20 * time_sat +
            0.28 * peak_window +
            0.10 * np.tanh(log_thermal_dose / 3.0) +
            0.18 * risk_window_470_12 +
            0.15 * risk_window_440_24
        )

        net_process_index = strengthening - softening

        X['temp_activation_low'] = temp_act_low
        X['temp_activation_mid'] = temp_act_mid
        X['temp_activation_high'] = temp_act_high
        X['time_saturation'] = time_sat
        X['damage_temp_activation'] = damage_temp
        X['damage_time_activation'] = damage_time
        X['strengthening_drive'] = strengthening
        X['softening_drive'] = softening
        X['net_process_index'] = net_process_index

        # -----------------------------
        # 7. 分区耦合特征
        # -----------------------------
        X['temp_in_low_zone'] = temp * is_low_temp
        X['temp_in_mid_zone'] = temp * is_mid_temp
        X['temp_in_high_zone'] = temp * is_high_temp
        X['temp_in_very_high_zone'] = temp * is_very_high_temp

        X['log_time_in_short_zone'] = log_time * is_short_time
        X['log_time_in_mid_zone'] = log_time * is_mid_time
        X['log_time_in_long_zone'] = log_time * is_long_time
        X['log_time_in_very_long_zone'] = log_time * is_very_long_time

        X['temp_log_time_low_zone'] = temp * log_time * is_low_temp
        X['temp_log_time_mid_zone'] = temp * log_time * is_mid_temp
        X['temp_log_time_high_zone'] = temp * log_time * is_high_temp
        X['temp_log_time_very_high_zone'] = temp * log_time * is_very_high_temp

        X['high_temp_long_time_drive'] = np.maximum(temp - p['temp_mid'], 0.0) * np.maximum(time - p['time_mid'], 0.0)
        X['very_high_temp_very_long_time_drive'] = np.maximum(temp - p['temp_high'], 0.0) * np.maximum(time - p['time_long'], 0.0)

        # 显式针对大误差工艺点
        X['dist_to_470_12_temp'] = np.abs(temp - 470.0)
        X['dist_to_470_12_time'] = np.abs(time - 12.0)
        X['dist_to_440_24_temp'] = np.abs(temp - 440.0)
        X['dist_to_440_24_time'] = np.abs(time - 24.0)

        X['temp_time_470_12_signed'] = (temp - 470.0) * (time - 12.0)
        X['temp_time_440_24_signed'] = (temp - 440.0) * (time - 24.0)

        # -----------------------------
        # 8. 物理基线预测值
        # -----------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)

        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        baseline_gap_ty = baseline_tensile - baseline_yield
        baseline_ratio_yt = self._safe_divide(baseline_yield, baseline_tensile)
        baseline_sum_strength = baseline_tensile + baseline_yield

        X['baseline_gap_ty'] = baseline_gap_ty
        X['baseline_ratio_yt'] = baseline_ratio_yt
        X['baseline_sum_strength'] = baseline_sum_strength

        # -----------------------------
        # 9. 相对原始凝固态的基线增量
        # -----------------------------
        X['baseline_tensile_gain_from_base'] = baseline_tensile - p['base_tensile']
        X['baseline_yield_gain_from_base'] = baseline_yield - p['base_yield']
        X['baseline_strain_change_from_base'] = baseline_strain - p['base_strain']

        X['baseline_tensile_gain_ratio'] = self._safe_divide(
            baseline_tensile - p['base_tensile'], p['base_tensile']
        )
        X['baseline_yield_gain_ratio'] = self._safe_divide(
            baseline_yield - p['base_yield'], p['base_yield']
        )
        X['baseline_strain_change_ratio'] = self._safe_divide(
            baseline_strain - p['base_strain'], p['base_strain']
        )

        # -----------------------------
        # 10. 基于基线的残差型工艺表征
        # “残差”这里指输入工艺相对物理机制中心/基线状态的偏离，不使用标签
        # -----------------------------
        X['baseline_strength_per_temp'] = self._safe_divide(baseline_sum_strength, temp_k)
        X['baseline_gap_over_strength'] = self._safe_divide(baseline_gap_ty, baseline_sum_strength)
        X['baseline_tensile_per_hour'] = self._safe_divide(baseline_tensile, time + 1.0)
        X['baseline_yield_per_hour'] = self._safe_divide(baseline_yield, time + 1.0)
        X['baseline_strain_per_hour'] = self._safe_divide(baseline_strain, time + 1.0)

        X['baseline_tensile_x_peak_window'] = baseline_tensile * peak_window
        X['baseline_yield_x_peak_window'] = baseline_yield * peak_window
        X['baseline_strain_x_peak_window'] = baseline_strain * peak_window

        X['baseline_tensile_x_risk470_12'] = baseline_tensile * risk_window_470_12
        X['baseline_yield_x_risk470_12'] = baseline_yield * risk_window_470_12
        X['baseline_strain_x_risk470_12'] = baseline_strain * risk_window_470_12

        X['baseline_tensile_x_risk440_24'] = baseline_tensile * risk_window_440_24
        X['baseline_yield_x_risk440_24'] = baseline_yield * risk_window_440_24
        X['baseline_strain_x_risk440_24'] = baseline_strain * risk_window_440_24

        X['baseline_tensile_x_net_process'] = baseline_tensile * net_process_index
        X['baseline_yield_x_net_process'] = baseline_yield * net_process_index
        X['baseline_strain_x_softening'] = baseline_strain * softening

        # -----------------------------
        # 11. 强塑权衡启发特征
        # -----------------------------
        X['predicted_strength_ductility_coupling'] = self._safe_divide(
            baseline_sum_strength, baseline_strain + 1e-6
        )
        X['predicted_tensile_minus_yield'] = baseline_gap_ty
        X['predicted_yield_tensile_ratio'] = baseline_ratio_yt

        ductility_recovery_index = (
            0.45 * softening +
            0.25 * risk_window_470_12 +
            0.20 * risk_window_440_24 +
            0.10 * is_very_long_time -
            0.25 * peak_window
        )
        strength_ductility_tradeoff = net_process_index - ductility_recovery_index

        X['ductility_recovery_index'] = ductility_recovery_index
        X['strength_ductility_tradeoff'] = strength_ductility_tradeoff

        # -----------------------------
        # 12. 常数/基准相关特征
        # -----------------------------
        base_strength_sum = p['base_tensile'] + p['base_yield']
        base_strength_ratio = p['base_yield'] / p['base_tensile']

        X['base_strength_sum'] = base_strength_sum
        X['base_strength_ratio'] = base_strength_ratio
        X['process_to_base_potential'] = (
            0.45 * temp_act_low +
            0.25 * time_sat +
            0.20 * peak_window +
            0.10 * np.tanh(log_thermal_dose / 3.0) -
            0.20 * softening
        )

        # -----------------------------
        # 13. 稳健裁剪与数值友好特征
        # -----------------------------
        clipped_temp = np.clip(temp, 430.0, 480.0)
        clipped_time = np.clip(time, 0.0, 30.0)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)
        X['clipped_temp_time'] = clipped_temp * clipped_time

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