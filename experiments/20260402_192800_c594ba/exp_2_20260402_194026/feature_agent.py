import pandas as pd
import numpy as np


# 7499铝合金热处理的经验物理参数（基于铝合金析出/回复动力学的常识性估计）
DOMAIN_PARAMS = {
    # 机制边界：从当前误差分布看，440℃附近与470℃附近表现出明显不同响应
    'critical_temp_regime_shift': 455.0,   # 析出强化主导 -> 回复/粗化竞争增强的转折区
    'high_temp_boundary': 468.0,           # 高温敏感区，470℃/12h误差大，需单独刻画
    'low_temp_boundary': 442.0,            # 低温敏感区，440℃/1h与24h误差大

    'critical_time_regime_shift': 10.0,    # 短时/中长时机制切换
    'long_time_boundary': 18.0,            # 长时暴露边界
    'short_time_boundary': 2.0,            # 极短时边界

    # 热激活动力学参数（铝合金析出/扩散的量级估计）
    'activation_energy_Q': 120000.0,       # J/mol，经验量级
    'gas_constant_R': 8.314,               # J/mol/K

    # Larson-Miller 型参数常数
    'lmp_C': 20.0,

    # 参考条件
    'ref_temp_c': 440.0,
    'ref_time_h': 12.0,

    # 原始铸态基线
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 响应窗口中心：依据题述趋势，性能在中高温+中等时间附近存在优值窗口
    'peak_temp_strength': 462.0,
    'peak_temp_strain': 468.0,
    'peak_time_strength': 14.0,
    'peak_time_strain': 12.0,
}


class FeatureAgent:
    """
    面向小样本材料热处理问题的物理启发特征工程。
    设计原则：
    1. 少而精，避免高维泛滥；
    2. 显式编码温度-时间交互、动力学累计量、机制边界；
    3. 加入 physics baseline，让下游模型学习残差。
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于世界知识的粗略基线：
        - 相比原始铸态，当前热处理区间整体提升强度与应变；
        - 温度/时间响应非单调，存在工艺窗口；
        - 强度与屈服强相关，应变峰值略偏高温/中等时间。
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = max(float(time), 1e-6)

        # 归一化工艺变量
        t_norm = (temp - 440.0) / 40.0
        log_time = np.log1p(time)
        log_time_norm = np.log1p(time) / np.log(25.0)

        # 高于铸态的整体增益：在当前实验窗口内应整体上升
        heat_gain = 1.0 / (1.0 + np.exp(-(
            1.8 * (temp - 430.0) / 20.0 +
            1.6 * (log_time - np.log1p(2.0))
        )))

        # 强度窗口：中高温 + 中等/偏长时间
        strength_window_temp = np.exp(-((temp - p['peak_temp_strength']) / 18.0) ** 2)
        strength_window_time = np.exp(-((np.log1p(time) - np.log1p(p['peak_time_strength'])) / 0.75) ** 2)
        strength_window = strength_window_temp * strength_window_time

        # 应变窗口：峰值略偏高温、时间偏中等
        strain_window_temp = np.exp(-((temp - p['peak_temp_strain']) / 16.0) ** 2)
        strain_window_time = np.exp(-((np.log1p(time) - np.log1p(p['peak_time_strain'])) / 0.7) ** 2)
        strain_window = strain_window_temp * strain_window_time

        # 长时/高温下的轻微回落项：仅作弱修正，避免违背“整体高于原始样品”
        overexposure = max(0.0, temp - 465.0) / 20.0 * max(0.0, np.log1p(time) - np.log1p(12.0))

        baseline_tensile = (
            p['as_cast_tensile']
            + 75.0 * heat_gain
            + 85.0 * strength_window
            - 12.0 * overexposure
        )

        baseline_yield = (
            p['as_cast_yield']
            + 55.0 * heat_gain
            + 55.0 * strength_window
            - 10.0 * overexposure
        )

        baseline_strain = (
            p['as_cast_strain']
            + 2.2 * heat_gain
            + 9.0 * strain_window
            - 1.0 * overexposure
        )

        # 物理合理下限：热处理后整体不应比铸态差太多，且题述显示整体提升
        baseline_tensile = max(baseline_tensile, p['as_cast_tensile'] + 5.0)
        baseline_yield = max(baseline_yield, p['as_cast_yield'] + 3.0)
        baseline_strain = max(baseline_strain, p['as_cast_strain'] + 0.2)

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        p = DOMAIN_PARAMS
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        time_clip = np.clip(time, 1e-6, None)

        temp_k = temp + 273.15
        log_time = np.log1p(time_clip)
        sqrt_time = np.sqrt(time_clip)

        # -------------------------
        # 1) 基础少量非线性
        # -------------------------
        cols['temp'] = temp
        cols['time'] = time_clip
        cols['temp_centered'] = temp - p['critical_temp_regime_shift']
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time

        cols['temp_sq_scaled'] = ((temp - p['ref_temp_c']) / 20.0) ** 2
        cols['log_time_sq'] = (log_time - np.log1p(p['ref_time_h'])) ** 2

        # -------------------------
        # 2) 温度-时间交互 / 热暴露
        # -------------------------
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_x_time_scaled'] = temp * time_clip / 100.0
        cols['centered_temp_x_log_time'] = (temp - p['critical_temp_regime_shift']) * log_time

        # 简化热暴露：适合小样本
        cols['thermal_dose'] = (temp_k / 1000.0) * log_time
        cols['thermal_dose_linear'] = (temp - 400.0) * (time_clip / 24.0)

        # -------------------------
        # 3) 动力学等效特征
        # -------------------------
        # Arrhenius 指数项：反映扩散/析出动力学
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * temp_k))
        cols['arrhenius'] = arrhenius
        cols['arrhenius_x_time'] = arrhenius * time_clip
        cols['arrhenius_x_log_time'] = arrhenius * log_time

        # JMAK风格转化分数（简化）
        # 使用小指数避免数值过饱和
        k_eff = arrhenius * 1e9
        cols['transform_fraction_n1'] = 1.0 - np.exp(-k_eff * time_clip)
        cols['transform_fraction_n2'] = 1.0 - np.exp(-k_eff * (time_clip ** 2) / 24.0)

        # Larson-Miller Parameter（小时单位常见经验形式）
        cols['lmp'] = temp_k * (p['lmp_C'] + np.log10(time_clip))
        cols['lmp_centered'] = cols['lmp'] - np.mean(cols['lmp'])

        # Hollomon/Jaffe风格 temper parameter 简化
        cols['temper_param'] = temp_k * (10.0 + log_time)

        # -------------------------
        # 4) 机制区间检测特征（关键）
        # -------------------------
        is_low_temp = (temp <= p['low_temp_boundary']).astype(float)
        is_mid_temp = ((temp > p['low_temp_boundary']) & (temp < p['high_temp_boundary'])).astype(float)
        is_high_temp = (temp >= p['high_temp_boundary']).astype(float)

        is_short_time = (time_clip <= p['short_time_boundary']).astype(float)
        is_mid_time = ((time_clip > p['short_time_boundary']) & (time_clip < p['long_time_boundary'])).astype(float)
        is_long_time = (time_clip >= p['long_time_boundary']).astype(float)

        cols['is_low_temp'] = is_low_temp
        cols['is_mid_temp'] = is_mid_temp
        cols['is_high_temp'] = is_high_temp

        cols['is_short_time'] = is_short_time
        cols['is_mid_time'] = is_mid_time
        cols['is_long_time'] = is_long_time

        # 关键组合区：直接针对大误差点
        cols['regime_lowT_shortt'] = ((temp <= p['low_temp_boundary']) & (time_clip <= p['short_time_boundary'])).astype(float)
        cols['regime_lowT_longt'] = ((temp <= p['low_temp_boundary']) & (time_clip >= p['long_time_boundary'])).astype(float)
        cols['regime_highT_midt'] = ((temp >= p['high_temp_boundary']) &
                                     (time_clip >= p['critical_time_regime_shift'] - 3) &
                                     (time_clip <= p['critical_time_regime_shift'] + 4)).astype(float)
        cols['regime_highT_longt'] = ((temp >= p['high_temp_boundary']) & (time_clip >= p['long_time_boundary'])).astype(float)

        # 软分段激活：比硬阈值更平滑
        cols['temp_above_shift'] = np.maximum(0.0, temp - p['critical_temp_regime_shift'])
        cols['temp_below_low_boundary'] = np.maximum(0.0, p['low_temp_boundary'] - temp)
        cols['temp_above_high_boundary'] = np.maximum(0.0, temp - p['high_temp_boundary'])
        cols['time_above_shift'] = np.maximum(0.0, time_clip - p['critical_time_regime_shift'])
        cols['time_above_long_boundary'] = np.maximum(0.0, time_clip - p['long_time_boundary'])

        cols['highT_x_midtime_activation'] = cols['temp_above_high_boundary'] * np.exp(
            -((log_time - np.log1p(12.0)) / 0.45) ** 2
        )
        cols['lowT_x_shorttime_activation'] = cols['temp_below_low_boundary'] * np.exp(
            -((log_time - np.log1p(1.0)) / 0.35) ** 2
        )
        cols['lowT_x_longtime_activation'] = cols['temp_below_low_boundary'] * np.exp(
            -((log_time - np.log1p(24.0)) / 0.35) ** 2
        )

        # -------------------------
        # 5) 工艺窗口距离特征
        # -------------------------
        cols['dist_to_strength_peak'] = np.sqrt(
            ((temp - p['peak_temp_strength']) / 15.0) ** 2 +
            ((log_time - np.log1p(p['peak_time_strength'])) / 0.55) ** 2
        )

        cols['dist_to_strain_peak'] = np.sqrt(
            ((temp - p['peak_temp_strain']) / 15.0) ** 2 +
            ((log_time - np.log1p(p['peak_time_strain'])) / 0.50) ** 2
        )

        cols['strength_peak_proximity'] = np.exp(-cols['dist_to_strength_peak'] ** 2)
        cols['strain_peak_proximity'] = np.exp(-cols['dist_to_strain_peak'] ** 2)

        # -------------------------
        # 6) physics baseline 作为特征
        # -------------------------
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []

        for tt, tm in zip(temp, time_clip):
            bs, bu, by = self.physics_baseline(tt, tm)
            baseline_strain.append(bs)
            baseline_tensile.append(bu)
            baseline_yield.append(by)

        baseline_strain = np.array(baseline_strain)
        baseline_tensile = np.array(baseline_tensile)
        baseline_yield = np.array(baseline_yield)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # -------------------------
        # 7) 相对基线 / 相对机制边界特征
        # -------------------------
        cols['baseline_strength_sum'] = baseline_tensile + baseline_yield
        cols['baseline_yield_tensile_ratio'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)

        cols['temp_relative_to_shift'] = (temp - p['critical_temp_regime_shift']) / p['critical_temp_regime_shift']
        cols['time_relative_to_shift'] = (time_clip - p['critical_time_regime_shift']) / p['critical_time_regime_shift']

        cols['baseline_tensile_x_highT'] = baseline_tensile * is_high_temp
        cols['baseline_yield_x_lowT'] = baseline_yield * is_low_temp
        cols['baseline_strain_x_midtime'] = baseline_strain * is_mid_time

        # 相对于关键边界的“残差坐标”
        cols['delta_temp_to_low_boundary'] = temp - p['low_temp_boundary']
        cols['delta_temp_to_high_boundary'] = temp - p['high_temp_boundary']
        cols['delta_logtime_to_12h'] = log_time - np.log1p(12.0)
        cols['delta_logtime_to_24h'] = log_time - np.log1p(24.0)

        # -------------------------
        # 8) 少量有物理意义的比值特征
        # -------------------------
        cols['log_time_over_temp'] = log_time / temp_k
        cols['temp_over_log_time'] = temp / np.clip(log_time, 1e-6, None)
        cols['arrhenius_over_time'] = arrhenius / np.clip(time_clip, 1e-6, None)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names