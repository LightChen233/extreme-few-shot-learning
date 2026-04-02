import pandas as pd
import numpy as np


# 7499 铝合金在给定温度/时间热处理后的性能预测特征工程
# 重点针对当前高误差区域：
# 1) 440°C / 1h ：短时高温，可能处于“快速激活但尚未稳定”的欠时效/瞬态区
# 2) 460°C / 12h：中高温中长时，可能接近强化-粗化竞争区
# 3) 470°C / 12h：更高温中长时，可能进入过时效/粗化/局部溶解敏感区
# 4) 440°C / 24h：较低温长时，可能对应“低温长时等效峰值或后峰值”区域
#
# 因此本版特征工程核心：
# - 显式构造 440/460/470 与 1/12/24 的“误差热点邻域”特征
# - 强化 temp-time 耦合与边界跨越特征
# - 构建更符合热激活/析出-粗化竞争的 baseline
# - 用基线 + 邻域修正让模型学残差


DOMAIN_PARAMS = {
    # ---------------------------
    # 基础常数
    # ---------------------------
    'gas_constant_R': 8.314,                    # J/(mol*K)
    'activation_energy_Q': 118000.0,           # J/mol，析出/扩散控制过程经验量级
    'coarsening_energy_Qc': 132000.0,          # J/mol，粗化/过时效过程经验量级

    # ---------------------------
    # 原始铸态基线
    # ---------------------------
    'base_strain': 6.94,
    'base_tensile': 145.83,
    'base_yield': 96.60,

    # ---------------------------
    # 温度机制边界（针对当前数据范围，直接以实际 temp 标度定义）
    # 这些不是严格相图点，而是建模上的“组织演化转折点”
    # ---------------------------
    'critical_temp_regime_shift': 435.0,       # 从弱响应到明显热处理响应
    'critical_temp_fast_precip': 445.0,        # 析出动力学明显加快
    'critical_temp_competition': 455.0,        # 强化/粗化竞争开始增强
    'critical_temp_overage': 465.0,            # 过时效风险显著
    'critical_temp_severe_overage': 472.0,     # 严重过时效/组织不稳定敏感区

    # ---------------------------
    # 时间机制边界
    # ---------------------------
    'critical_time_transient': 1.5,            # 短时瞬态区，1h 附近很关键
    'critical_time_growth': 6.0,               # 析出增长主导
    'critical_time_competition': 10.0,         # 强化与粗化竞争窗口开始
    'critical_time_overage': 14.0,             # 中长时过时效风险
    'critical_time_long': 20.0,                # 长时区
    'critical_time_severe_overage': 24.0,      # 严重长时区

    # ---------------------------
    # 经验峰值/中心区
    # 结合现有误差分布，峰值不设在最高温，而偏向中温中时
    # ---------------------------
    'peak_temp_center': 452.0,
    'peak_temp_width': 10.0,
    'peak_time_center': 10.0,
    'peak_log_time_width': 0.62,

    # ---------------------------
    # 误差热点工艺点（显式建特征）
    # ---------------------------
    'hot_temp_1': 440.0,
    'hot_temp_2': 460.0,
    'hot_temp_3': 470.0,
    'hot_time_1': 1.0,
    'hot_time_2': 12.0,
    'hot_time_3': 24.0,

    # ---------------------------
    # 基线幅值
    # ---------------------------
    'max_tensile_increment': 215.0,
    'max_yield_increment': 165.0,
    'max_strain_drop': 2.8,
    'max_strain_recovery': 3.1,

    # ---------------------------
    # Larson-Miller 风格
    # ---------------------------
    'lmp_constant_C': 20.0,

    # ---------------------------
    # 邻域尺度
    # ---------------------------
    'temp_local_width': 6.0,
    'time_local_width': 2.5,
    'log_time_local_width': 0.35
}


class FeatureAgent:
    """针对热处理边界条件、时效竞争机制和误差热点的特征工程"""

    def __init__(self):
        self.feature_names = []
        self.params = DOMAIN_PARAMS
        self.R = self.params['gas_constant_R']
        self.Q = self.params['activation_energy_Q']
        self.Qc = self.params['coarsening_energy_Qc']

    def _safe_log(self, x):
        return np.log1p(np.clip(x, 0, None))

    def _safe_divide(self, a, b):
        b = np.where(np.abs(b) < 1e-12, 1e-12, b)
        return a / b

    def _sigmoid(self, x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    def _gaussian(self, x, center, width):
        width = max(float(width), 1e-12)
        return np.exp(-((x - center) / width) ** 2)

    def physics_baseline(self, temp, time):
        """
        基于热处理/析出强化的一般规律构建近似基线：
        - 初始阶段：强化随温度激活、随时间推进
        - 中间阶段：接近峰值时效，强度最高
        - 高温中长时：粗化/过时效/局部回复，强度下降、塑性恢复
        """
        p = self.params
        temp = np.asarray(temp, dtype=float)
        time = np.asarray(time, dtype=float)

        temp_k = temp + 273.15
        log_time = self._safe_log(time)

        # 热激活/等效暴露
        arr = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        carr = np.exp(-self.Qc / np.clip(self.R * temp_k, 1e-12, None))

        dose = time * arr
        coarsen_dose = time * carr

        log_dose = np.log1p(np.clip(dose * 1e18, 0, None))
        log_coarsen = np.log1p(np.clip(coarsen_dose * 1e20, 0, None))

        # 温度激活
        temp_activate = self._sigmoid((temp - p['critical_temp_regime_shift']) / 5.5)
        temp_fast = self._sigmoid((temp - p['critical_temp_fast_precip']) / 4.5)
        temp_over = self._sigmoid((temp - p['critical_temp_overage']) / 3.8)
        temp_severe = self._sigmoid((temp - p['critical_temp_severe_overage']) / 2.5)

        # 时间推进
        time_short = 1.0 - np.exp(-np.clip(time, 0, None) / p['critical_time_growth'])
        time_mid = self._sigmoid((time - p['critical_time_growth']) / 2.0)
        time_over = self._sigmoid((time - p['critical_time_overage']) / 2.2)
        time_long = self._sigmoid((time - p['critical_time_long']) / 2.0)

        # 峰值窗口
        peak_temp = self._gaussian(temp, p['peak_temp_center'], p['peak_temp_width'])
        peak_time = self._gaussian(log_time, np.log1p(p['peak_time_center']), p['peak_log_time_width'])
        peak_window = peak_temp * peak_time

        # 短时瞬态区（解释 440°C/1h 类误差）
        transient_short = self._gaussian(time, p['hot_time_1'], 0.9) * self._gaussian(temp, p['hot_temp_1'], 7.0)

        # 低温长时等效峰值倾向（解释 440°C/24h）
        lowtemp_longtime_peak_like = (
            self._gaussian(temp, p['hot_temp_1'], 7.0) *
            self._gaussian(time, p['hot_time_3'], 5.0)
        )

        # 中高温 12h 竞争区（解释 460/470°C,12h）
        competition_window = self._gaussian(time, p['hot_time_2'], 3.0) * (
            0.55 * self._gaussian(temp, p['hot_temp_2'], 6.0) +
            0.45 * self._gaussian(temp, p['hot_temp_3'], 5.5)
        )

        # 强化驱动
        strengthening = (
            0.30 * temp_activate * time_short +
            0.20 * temp_fast * time_mid +
            0.30 * peak_window +
            0.12 * self._sigmoid(log_dose - 2.4) +
            0.08 * lowtemp_longtime_peak_like
        )
        strengthening = np.clip(strengthening, 0.0, 1.25)

        # 软化/过时效驱动
        softening = (
            0.38 * temp_over * time_over +
            0.26 * temp_severe * time_over +
            0.18 * temp_over * time_long +
            0.10 * self._sigmoid(log_coarsen - 1.8) +
            0.08 * competition_window
        )
        softening = np.clip(softening, 0.0, 1.35)

        # 净强化
        net = strengthening - 0.85 * softening

        baseline_tensile = (
            p['base_tensile']
            + p['max_tensile_increment'] * strengthening
            - 118.0 * softening
            + 10.0 * lowtemp_longtime_peak_like
            - 8.0 * transient_short
        )

        baseline_yield = (
            p['base_yield']
            + p['max_yield_increment'] * strengthening
            - 98.0 * softening
            + 6.0 * lowtemp_longtime_peak_like
            - 10.0 * transient_short
        )

        baseline_strain = (
            p['base_strain']
            - p['max_strain_drop'] * np.clip(net, 0, None)
            + p['max_strain_recovery'] * softening
            + 0.65 * transient_short
            + 0.55 * temp_over * time_over
            + 0.35 * temp_severe
        )

        baseline_tensile = np.clip(baseline_tensile, 80.0, 420.0)
        baseline_yield = np.clip(baseline_yield, 50.0, 340.0)
        baseline_strain = np.clip(baseline_strain, 2.0, 18.0)

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        X = pd.DataFrame(index=df.index)
        p = self.params

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        temp_k = temp + 273.15
        log_time = self._safe_log(time)
        sqrt_time = np.sqrt(np.clip(time, 0, None))
        inv_temp_k = 1.0 / np.clip(temp_k, 1e-12, None)

        # 基础阈值
        T1 = p['critical_temp_regime_shift']
        T2 = p['critical_temp_fast_precip']
        T3 = p['critical_temp_competition']
        T4 = p['critical_temp_overage']
        T5 = p['critical_temp_severe_overage']

        t1 = p['critical_time_transient']
        t2 = p['critical_time_growth']
        t3 = p['critical_time_competition']
        t4 = p['critical_time_overage']
        t5 = p['critical_time_long']
        t6 = p['critical_time_severe_overage']

        # ---------------------------
        # 1. 原始基础特征
        # ---------------------------
        X['temp'] = temp
        X['time'] = time
        X['temp_k'] = temp_k
        X['log_time'] = log_time
        X['sqrt_time'] = sqrt_time
        X['inv_temp_k'] = inv_temp_k

        # ---------------------------
        # 2. 常规非线性与耦合
        # ---------------------------
        X['temp_sq'] = temp ** 2
        X['time_sq'] = time ** 2
        X['temp_cube_scaled'] = (temp ** 3) / 1e5
        X['log_time_sq'] = log_time ** 2
        X['temp_time'] = temp * time
        X['temp_log_time'] = temp * log_time
        X['temp_sqrt_time'] = temp * sqrt_time
        X['time_over_temp'] = self._safe_divide(time, temp)
        X['time_over_temp_k'] = self._safe_divide(time, temp_k)
        X['log_time_over_temp_k'] = self._safe_divide(log_time, temp_k)

        # ---------------------------
        # 3. 动力学等效特征
        # ---------------------------
        arrhenius = np.exp(-self.Q / np.clip(self.R * temp_k, 1e-12, None))
        coarsening_arrhenius = np.exp(-self.Qc / np.clip(self.R * temp_k, 1e-12, None))

        thermal_dose = time * arrhenius
        coarsening_dose = time * coarsening_arrhenius

        X['arrhenius_factor'] = arrhenius
        X['coarsening_arrhenius'] = coarsening_arrhenius
        X['thermal_dose'] = thermal_dose
        X['coarsening_dose'] = coarsening_dose
        X['log_thermal_dose'] = np.log1p(np.clip(thermal_dose * 1e18, 0, None))
        X['log_coarsening_dose'] = np.log1p(np.clip(coarsening_dose * 1e20, 0, None))
        X['dose_ratio_precip_to_coarsen'] = self._safe_divide(
            X['thermal_dose'].values, X['coarsening_dose'].values + 1e-18
        )

        # Larson-Miller 风格
        lmp = temp_k * (p['lmp_constant_C'] + np.log10(np.clip(time, 1e-6, None)))
        X['larson_miller_param'] = lmp
        X['lmp_scaled'] = lmp / 1000.0

        # Zener-Hollomon/JMAK 风格
        X['jmak_t05'] = np.power(np.clip(time, 0, None) + 1e-9, 0.5)
        X['jmak_t067'] = np.power(np.clip(time, 0, None) + 1e-9, 0.67)
        X['jmak_drive_05'] = arrhenius * X['jmak_t05'].values
        X['jmak_drive_067'] = arrhenius * X['jmak_t067'].values
        X['log_jmak_drive_05'] = np.log1p(X['jmak_drive_05'].values * 1e18)
        X['log_jmak_drive_067'] = np.log1p(X['jmak_drive_067'].values * 1e18)

        # ---------------------------
        # 4. 机制区间检测特征（关键）
        # ---------------------------
        X['is_low_response_temp'] = (temp < T1).astype(float)
        X['is_activated_temp'] = ((temp >= T1) & (temp < T2)).astype(float)
        X['is_fast_precip_temp'] = ((temp >= T2) & (temp < T3)).astype(float)
        X['is_competition_temp'] = ((temp >= T3) & (temp < T4)).astype(float)
        X['is_overage_temp'] = ((temp >= T4) & (temp < T5)).astype(float)
        X['is_severe_overage_temp'] = (temp >= T5).astype(float)

        X['is_transient_time'] = (time < t1).astype(float)
        X['is_growth_time'] = ((time >= t1) & (time < t2)).astype(float)
        X['is_mid_time'] = ((time >= t2) & (time < t3)).astype(float)
        X['is_competition_time'] = ((time >= t3) & (time < t4)).astype(float)
        X['is_overage_time'] = ((time >= t4) & (time < t6)).astype(float)
        X['is_severe_overage_time'] = (time >= t6).astype(float)

        X['is_under_aged'] = ((temp < T3) & (time < t2)).astype(float)
        X['is_near_peak'] = ((temp >= T2) & (temp < T4) & (time >= t2) & (time < t4)).astype(float)
        X['is_competition_zone'] = ((temp >= T3) & (time >= t3) & (time < t5)).astype(float)
        X['is_overaged'] = ((temp >= T4) & (time >= t4)).astype(float)
        X['is_severely_overaged'] = ((temp >= T5) & (time >= t6)).astype(float)

        # 重点边界型组合
        X['is_high_temp_short_time'] = ((temp >= T2) & (time <= t1)).astype(float)
        X['is_low_temp_long_time'] = ((temp <= T2) & (time >= t5)).astype(float)
        X['is_460_470_and_12h_like'] = (((temp >= 456) & (temp <= 472) & (time >= 9) & (time <= 15))).astype(float)
        X['is_440_and_1h_like'] = (((temp >= 436) & (temp <= 444) & (time >= 0.5) & (time <= 1.5))).astype(float)
        X['is_440_and_24h_like'] = (((temp >= 436) & (temp <= 444) & (time >= 20) & (time <= 28))).astype(float)

        # ---------------------------
        # 5. 相对边界距离特征
        # ---------------------------
        X['temp_relative_to_regime_shift'] = temp - T1
        X['temp_relative_to_fast_precip'] = temp - T2
        X['temp_relative_to_competition'] = temp - T3
        X['temp_relative_to_overage'] = temp - T4
        X['temp_relative_to_severe_overage'] = temp - T5

        X['time_relative_to_transient'] = time - t1
        X['time_relative_to_growth'] = time - t2
        X['time_relative_to_competition'] = time - t3
        X['time_relative_to_overage'] = time - t4
        X['time_relative_to_long'] = time - t5
        X['time_relative_to_severe_overage'] = time - t6

        X['relu_temp_above_regime_shift'] = np.maximum(temp - T1, 0)
        X['relu_temp_above_fast_precip'] = np.maximum(temp - T2, 0)
        X['relu_temp_above_competition'] = np.maximum(temp - T3, 0)
        X['relu_temp_above_overage'] = np.maximum(temp - T4, 0)
        X['relu_temp_above_severe_overage'] = np.maximum(temp - T5, 0)

        X['relu_time_above_growth'] = np.maximum(time - t2, 0)
        X['relu_time_above_competition'] = np.maximum(time - t3, 0)
        X['relu_time_above_overage'] = np.maximum(time - t4, 0)
        X['relu_time_above_long'] = np.maximum(time - t5, 0)
        X['relu_time_above_severe_overage'] = np.maximum(time - t6, 0)

        # ---------------------------
        # 6. 平滑门控特征
        # ---------------------------
        X['temp_activation_regime'] = self._sigmoid((temp - T1) / 5.5)
        X['temp_activation_fast_precip'] = self._sigmoid((temp - T2) / 4.5)
        X['temp_activation_competition'] = self._sigmoid((temp - T3) / 4.0)
        X['temp_activation_overage'] = self._sigmoid((temp - T4) / 3.8)
        X['temp_activation_severe_overage'] = self._sigmoid((temp - T5) / 2.5)

        X['time_activation_transient_end'] = self._sigmoid((time - t1) / 0.7)
        X['time_activation_growth'] = self._sigmoid((time - t2) / 2.0)
        X['time_activation_competition'] = self._sigmoid((time - t3) / 1.8)
        X['time_activation_overage'] = self._sigmoid((time - t4) / 2.2)
        X['time_activation_long'] = self._sigmoid((time - t5) / 2.0)
        X['time_activation_severe_overage'] = self._sigmoid((time - t6) / 2.0)

        # ---------------------------
        # 7. 峰值邻近与误差热点邻近（关键增强）
        # ---------------------------
        peak_temp_prox = self._gaussian(temp, p['peak_temp_center'], p['peak_temp_width'])
        peak_time_prox = self._gaussian(log_time, np.log1p(p['peak_time_center']), p['peak_log_time_width'])
        X['peak_temp_proximity'] = peak_temp_prox
        X['peak_time_proximity'] = peak_time_prox
        X['peak_window'] = peak_temp_prox * peak_time_prox

        # 对高误差点显式建邻域
        X['near_temp_440'] = self._gaussian(temp, p['hot_temp_1'], p['temp_local_width'])
        X['near_temp_460'] = self._gaussian(temp, p['hot_temp_2'], p['temp_local_width'])
        X['near_temp_470'] = self._gaussian(temp, p['hot_temp_3'], p['temp_local_width'])

        X['near_time_1h'] = self._gaussian(time, p['hot_time_1'], 0.8)
        X['near_time_12h'] = self._gaussian(time, p['hot_time_2'], p['time_local_width'])
        X['near_time_24h'] = self._gaussian(time, p['hot_time_3'], 4.0)

        X['near_log_time_1h'] = self._gaussian(log_time, np.log1p(p['hot_time_1']), 0.22)
        X['near_log_time_12h'] = self._gaussian(log_time, np.log1p(p['hot_time_2']), p['log_time_local_width'])
        X['near_log_time_24h'] = self._gaussian(log_time, np.log1p(p['hot_time_3']), 0.25)

        X['near_440_1'] = X['near_temp_440'].values * X['near_time_1h'].values
        X['near_440_24'] = X['near_temp_440'].values * X['near_time_24h'].values
        X['near_460_12'] = X['near_temp_460'].values * X['near_time_12h'].values
        X['near_470_12'] = X['near_temp_470'].values * X['near_time_12h'].values

        X['near_460_470_12_band'] = X['near_time_12h'].values * np.maximum(
            X['near_temp_460'].values, X['near_temp_470'].values
        )

        # 曼哈顿/径向距离型
        X['dist_440_1'] = np.abs(temp - 440.0) / 6.0 + np.abs(np.log1p(time) - np.log1p(1.0)) / 0.25
        X['dist_440_24'] = np.abs(temp - 440.0) / 6.0 + np.abs(np.log1p(time) - np.log1p(24.0)) / 0.25
        X['dist_460_12'] = np.abs(temp - 460.0) / 6.0 + np.abs(np.log1p(time) - np.log1p(12.0)) / 0.25
        X['dist_470_12'] = np.abs(temp - 470.0) / 5.0 + np.abs(np.log1p(time) - np.log1p(12.0)) / 0.25

        X['inv_dist_440_1'] = 1.0 / (1.0 + X['dist_440_1'].values)
        X['inv_dist_440_24'] = 1.0 / (1.0 + X['dist_440_24'].values)
        X['inv_dist_460_12'] = 1.0 / (1.0 + X['dist_460_12'].values)
        X['inv_dist_470_12'] = 1.0 / (1.0 + X['dist_470_12'].values)

        # ---------------------------
        # 8. 强化-软化竞争特征
        # ---------------------------
        precipitation_drive = (
            0.22 * X['temp_activation_regime'].values * (1.0 - X['temp_activation_overage'].values) +
            0.16 * X['temp_activation_fast_precip'].values * (1.0 - X['time_activation_severe_overage'].values) +
            0.18 * (1.0 - np.exp(-np.clip(time, 0, None) / t2)) +
            0.22 * X['peak_window'].values +
            0.12 * self._sigmoid(X['log_thermal_dose'].values - 2.4) +
            0.10 * X['near_440_24'].values
        )

        softening_drive = (
            0.24 * X['temp_activation_overage'].values * X['time_activation_overage'].values +
            0.20 * X['temp_activation_severe_overage'].values * X['time_activation_overage'].values +
            0.12 * X['temp_activation_overage'].values * X['time_activation_long'].values +
            0.12 * self._sigmoid(X['log_coarsening_dose'].values - 1.8) +
            0.16 * X['near_460_12'].values +
            0.16 * X['near_470_12'].values
        )

        transient_instability = (
            0.45 * X['near_440_1'].values +
            0.25 * X['is_high_temp_short_time'].values +
            0.15 * X['is_transient_time'].values * X['temp_activation_fast_precip'].values
        )

        competition_index = (
            X['temp_activation_competition'].values *
            X['time_activation_competition'].values *
            (1.0 - 0.5 * X['temp_activation_severe_overage'].values)
        )

        net_strengthening = precipitation_drive - 0.9 * softening_drive - 0.22 * transient_instability

        X['precipitation_drive'] = precipitation_drive
        X['softening_drive'] = softening_drive
        X['transient_instability'] = transient_instability
        X['competition_index'] = competition_index
        X['net_strengthening_index'] = net_strengthening
        X['overaging_severity'] = (
            X['temp_activation_overage'].values * X['time_activation_overage'].values +
            X['temp_activation_severe_overage'].values * X['time_activation_severe_overage'].values
        )

        # ---------------------------
        # 9. 分段交互
        # ---------------------------
        X['temp_in_fast_precip_zone'] = temp * X['is_fast_precip_temp'].values
        X['temp_in_competition_zone'] = temp * X['is_competition_temp'].values
        X['temp_in_overage_zone'] = temp * X['is_overage_temp'].values
        X['temp_in_severe_overage_zone'] = temp * X['is_severe_overage_temp'].values

        X['log_time_in_transient_zone'] = log_time * X['is_transient_time'].values
        X['log_time_in_growth_zone'] = log_time * X['is_growth_time'].values
        X['log_time_in_competition_zone'] = log_time * X['is_competition_time'].values
        X['log_time_in_overage_zone'] = log_time * X['is_overage_time'].values

        X['temp_log_time_near_peak'] = temp * log_time * X['is_near_peak'].values
        X['temp_log_time_overaged'] = temp * log_time * X['is_overaged'].values
        X['temp_log_time_competition_zone'] = temp * log_time * X['is_competition_zone'].values

        X['temp_x_near_440_1'] = temp * X['near_440_1'].values
        X['temp_x_near_460_12'] = temp * X['near_460_12'].values
        X['temp_x_near_470_12'] = temp * X['near_470_12'].values
        X['time_x_near_440_24'] = time * X['near_440_24'].values

        # ---------------------------
        # 10. 强塑性权衡特征
        # ---------------------------
        X['ductility_recovery_index'] = (
            0.55 * softening_drive
            + 0.20 * X['overaging_severity'].values
            + 0.18 * X['transient_instability'].values
            - 0.32 * precipitation_drive
        )
        X['strength_ductility_tradeoff'] = (
            X['net_strengthening_index'].values - X['ductility_recovery_index'].values
        )
        X['yield_sensitivity_index'] = (
            0.42 * precipitation_drive - 0.48 * softening_drive - 0.28 * transient_instability
        )
        X['strain_sensitivity_index'] = (
            0.55 * softening_drive + 0.35 * transient_instability - 0.30 * precipitation_drive
        )

        # ---------------------------
        # 11. 物理基线作为特征
        # ---------------------------
        baseline_strain, baseline_tensile, baseline_yield = self.physics_baseline(temp, time)
        X['baseline_strain'] = baseline_strain
        X['baseline_tensile'] = baseline_tensile
        X['baseline_yield'] = baseline_yield

        X['baseline_yield_tensile_ratio'] = self._safe_divide(baseline_yield, baseline_tensile)
        X['baseline_strength_sum'] = baseline_tensile + baseline_yield
        X['baseline_strength_diff'] = baseline_tensile - baseline_yield
        X['baseline_strength_mean'] = (baseline_tensile + baseline_yield) / 2.0

        X['baseline_tensile_x_peak'] = baseline_tensile * X['peak_window'].values
        X['baseline_yield_x_peak'] = baseline_yield * X['peak_window'].values
        X['baseline_strain_x_overaging'] = baseline_strain * X['overaging_severity'].values

        X['baseline_tensile_x_near_470_12'] = baseline_tensile * X['near_470_12'].values
        X['baseline_yield_x_near_460_12'] = baseline_yield * X['near_460_12'].values
        X['baseline_strain_x_near_440_1'] = baseline_strain * X['near_440_1'].values
        X['baseline_tensile_x_near_440_24'] = baseline_tensile * X['near_440_24'].values

        # ---------------------------
        # 12. 相对基线与相对机制偏移
        # ---------------------------
        X['temp_minus_peak_center'] = temp - p['peak_temp_center']
        X['log_time_minus_peak_center'] = log_time - np.log1p(p['peak_time_center'])

        X['relative_peak_temp_offset'] = self._safe_divide(
            temp - p['peak_temp_center'], p['peak_temp_width']
        )
        X['relative_peak_time_offset'] = self._safe_divide(
            log_time - np.log1p(p['peak_time_center']), p['peak_log_time_width']
        )

        X['relative_regime_temp'] = self._safe_divide(temp - T1, T1)
        X['relative_overage_temp'] = self._safe_divide(temp - T4, T4)
        X['relative_overage_time'] = self._safe_divide(time - t4, t4)
        X['relative_severe_overage_temp'] = self._safe_divide(temp - T5, T5)

        X['process_minus_baseline_strength_scale'] = (
            X['net_strengthening_index'].values * X['baseline_strength_mean'].values
        )
        X['process_minus_baseline_ductility_scale'] = (
            X['ductility_recovery_index'].values * baseline_strain
        )

        # ---------------------------
        # 13. 相对铸态潜力特征
        # ---------------------------
        base_tensile = p['base_tensile']
        base_yield = p['base_yield']
        base_strain = p['base_strain']

        X['base_tensile'] = base_tensile
        X['base_yield'] = base_yield
        X['base_strain'] = base_strain

        X['baseline_tensile_gain_over_cast'] = baseline_tensile - base_tensile
        X['baseline_yield_gain_over_cast'] = baseline_yield - base_yield
        X['baseline_strain_change_over_cast'] = baseline_strain - base_strain

        X['tensile_gain_ratio_over_cast'] = self._safe_divide(
            baseline_tensile - base_tensile, base_tensile
        )
        X['yield_gain_ratio_over_cast'] = self._safe_divide(
            baseline_yield - base_yield, base_yield
        )
        X['strain_change_ratio_over_cast'] = self._safe_divide(
            baseline_strain - base_strain, base_strain
        )

        # ---------------------------
        # 14. 热点局部校正特征（专门针对大误差点）
        # ---------------------------
        # 470/12：高温中长时，可能强度波动大、屈服偏高估或低估交替
        X['hotspot_470_12_softening_bias'] = (
            X['near_470_12'].values *
            X['temp_activation_overage'].values *
            X['time_activation_competition'].values
        )

        # 460/12：强化/粗化竞争
        X['hotspot_460_12_competition_bias'] = (
            X['near_460_12'].values *
            X['competition_index'].values
        )

        # 440/1：短时瞬态，对应应变与强度关系不稳定
        X['hotspot_440_1_transient_bias'] = (
            X['near_440_1'].values *
            X['transient_instability'].values
        )

        # 440/24：低温长时等效峰值/后峰值
        X['hotspot_440_24_delayed_peak_bias'] = (
            X['near_440_24'].values *
            (0.6 * X['peak_window'].values + 0.4 * X['precipitation_drive'].values)
        )

        # ---------------------------
        # 15. 稳健截断特征
        # ---------------------------
        clipped_temp = np.clip(temp, 430, 475)
        clipped_time = np.clip(time, 0, 30)

        X['clipped_temp'] = clipped_temp
        X['clipped_time'] = clipped_time
        X['clipped_temp_log_time'] = clipped_temp * np.log1p(clipped_time)

        X['time_saturation'] = 1.0 - np.exp(-clipped_time / 6.0)
        X['temp_saturation_regime'] = self._sigmoid((clipped_temp - T1) / 5.5)
        X['temp_saturation_overage'] = self._sigmoid((clipped_temp - T4) / 3.8)
        X['combined_saturation'] = X['time_saturation'].values * X['temp_saturation_regime'].values

        X['high_temp_long_time_penalty'] = (
            np.maximum(clipped_temp - T4, 0) * np.maximum(clipped_time - t4, 0)
        )
        X['severe_overage_penalty'] = (
            np.maximum(clipped_temp - T5, 0) * np.maximum(clipped_time - t6, 0)
        )
        X['short_time_high_temp_penalty'] = (
            np.maximum(clipped_temp - T2, 0) * np.maximum(t1 - clipped_time, 0)
        )

        # ---------------------------
        # 16. 最终清理
        # ---------------------------
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