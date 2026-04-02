import pandas as pd
import numpy as np


# 领域参数：针对 7xxx（7499）铝合金热处理/时效行为给出的低维物理先验
# 注意：这里不是精确材料常数数据库值，而是用于构造“机制边界 + 动力学尺度”的工程近似
DOMAIN_PARAMS = {
    # 温度机制边界：结合当前误差集中区（440/460/470℃）与铝合金高温组织演化常识设置
    # 低于该区间：组织演化较慢，短时处理不足
    'critical_temp_regime_shift': 450.0,
    # 中高温敏感区中心：470×12h 是最大误差点，说明此附近可能存在强化/粗化竞争拐点
    'critical_temp_peak_window': 468.0,
    # 更高温区：可能进入更强回复/粗化主导
    'critical_temp_high': 475.0,

    # 时间机制边界
    'critical_time_short': 3.0,
    'critical_time_regime_shift': 12.0,
    'critical_time_long': 18.0,

    # 热激活近似参数（J/mol）
    # 7xxx 铝合金中溶质扩散/析出过程量级常在 80~140 kJ/mol，取中间工程值
    'activation_energy_Q': 110000.0,
    'gas_constant_R': 8.314,

    # Arrhenius/LMP 相关常数
    'kelvin_offset': 273.15,
    'lmp_C': 20.0,

    # 原始铸态样品基准性能
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 当前实验窗口，用于构造相对位置特征
    'temp_ref_center': 450.0,
    'time_ref_center': 12.0,
    'temp_scale': 20.0,
    'time_scale': 12.0,
}


class FeatureAgent:
    """
    面向小样本（29条）的物理启发特征工程：
    1) 控制特征数，避免高维过拟合
    2) 显式编码温度-时间机制边界
    3) 使用热激活动力学等效特征
    4) 引入 physics baseline，让模型学习残差
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的近似基线：
        - 相比铸态，当前热处理窗口整体提升三项性能
        - 温度/时间作用非单调，但在本数据范围内总体存在“处理中等到中高热暴露更优”的趋势
        - 470×12h 附近视作高响应窗口之一
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = float(time)

        # 归一化工艺位置
        t_norm = (temp - 420.0) / 60.0          # 大致覆盖 420~480℃
        h_norm = np.log1p(time) / np.log1p(24.0)

        # 热暴露强度：随温度和时间增加而增加，但时间采用对数压缩
        exposure = 0.58 * t_norm + 0.42 * h_norm

        # 针对中高温-中等时间窗口设置“最优区”增益，避免纯单调假设
        temp_peak = np.exp(-((temp - 468.0) / 14.0) ** 2)
        time_peak = np.exp(-((np.log1p(time) - np.log1p(12.0)) / 0.65) ** 2)
        process_window = temp_peak * time_peak

        # 长时间可能出现响应分化：强度不一定同步下降，故仅给出温和惩罚
        long_time_soft = 1.0 / (1.0 + np.exp(-(time - 18.0) / 3.0))

        # 应变：在当前数据描述下整体提升，且中高温/中等时间更可能较优
        baseline_strain = (
            p['as_cast_strain']
            + 4.2 * exposure
            + 5.2 * process_window
            - 0.8 * long_time_soft
        )

        # 抗拉强度：整体显著高于铸态；中高温/12h 左右可能更高
        baseline_tensile = (
            p['as_cast_tensile']
            + 78.0 * exposure
            + 36.0 * process_window
            - 8.0 * long_time_soft
        )

        # 屈服强度：与抗拉强度高度协同，但对组织敏感，给稍弱窗口增益
        baseline_yield = (
            p['as_cast_yield']
            + 52.0 * exposure
            + 26.0 * process_window
            - 6.0 * long_time_soft
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        p = DOMAIN_PARAMS
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        T_k = temp + p['kelvin_offset']
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ========= 1) 原始与低阶基础特征 =========
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time

        # 小样本下只保留必要低阶非线性
        cols['temp_centered'] = (temp - p['temp_ref_center']) / p['temp_scale']
        cols['time_centered'] = (time - p['time_ref_center']) / p['time_scale']
        cols['temp_sq_scaled'] = cols['temp_centered'] ** 2
        cols['log_time_sq'] = log_time ** 2

        # ========= 2) 温度-时间交互 =========
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_x_time_scaled'] = temp * time / 100.0
        cols['temp_centered_x_log_time'] = cols['temp_centered'] * log_time

        # ========= 3) 动力学等效特征 =========
        # Arrhenius 速率项（越大表示热激活过程越容易）
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['arrhenius'] = arrhenius

        # 等效热暴露：时间 * Arrhenius
        cols['eq_thermal_dose'] = time * arrhenius
        cols['log_eq_thermal_dose'] = np.log1p(cols['eq_thermal_dose'] * 1e12)

        # Larson-Miller-like 参数（工程化使用，不追求绝对物理精确）
        cols['lmp'] = T_k * (p['lmp_C'] + np.log10(np.clip(time, 1e-6, None)))

        # 温度倒数：热激活常见线性化变量
        cols['inv_temp_k'] = 1.0 / T_k

        # ========= 4) 机制区间检测特征（关键） =========
        cols['is_high_temp'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_very_high_temp'] = (temp >= p['critical_temp_high']).astype(float)
        cols['is_short_time'] = (time <= p['critical_time_short']).astype(float)
        cols['is_mid_time_or_longer'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_long_time'] = (time >= p['critical_time_long']).astype(float)

        # 误差热点/机制敏感区：440×1h、440×24h、460×12h、470×12h
        cols['is_lowT_short'] = ((temp <= 445.0) & (time <= 3.0)).astype(float)
        cols['is_lowT_long'] = ((temp <= 445.0) & (time >= 18.0)).astype(float)
        cols['is_midhighT_midtime'] = ((temp >= 455.0) & (temp <= 472.0) & (time >= 10.0) & (time <= 14.0)).astype(float)
        cols['is_470_12_window'] = ((temp >= 465.0) & (temp <= 475.0) & (time >= 10.0) & (time <= 14.0)).astype(float)

        # 分段激活：距离边界有多远
        cols['temp_above_shift'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_above_high'] = np.clip(temp - p['critical_temp_high'], 0, None)
        cols['time_above_12'] = np.clip(time - p['critical_time_regime_shift'], 0, None)
        cols['time_above_18'] = np.clip(time - p['critical_time_long'], 0, None)
        cols['time_below_3'] = np.clip(p['critical_time_short'] - time, 0, None)

        # 组合边界：模拟“机制突变只在温度和时间同时越界时出现”
        cols['highT_and_midlong_time'] = cols['is_high_temp'] * cols['is_mid_time_or_longer']
        cols['veryhighT_and_long_time'] = cols['is_very_high_temp'] * cols['is_long_time']
        cols['lowT_and_short_time'] = cols['is_lowT_short']
        cols['lowT_and_long_time'] = cols['is_lowT_long']

        # ========= 5) 最优工艺窗口/峰值响应特征 =========
        # 用平滑窗口表达“非单调 + 工艺最优区”
        temp_peak = np.exp(-((temp - p['critical_temp_peak_window']) / 12.0) ** 2)
        time_peak = np.exp(-((log_time - np.log1p(p['critical_time_regime_shift'])) / 0.55) ** 2)
        cols['temp_peak_window'] = temp_peak
        cols['time_peak_window'] = time_peak
        cols['process_peak_window'] = temp_peak * time_peak

        # 440/1h 和 440/24h 说明 440℃ 区域随时间有明显分化，加入局部窗口
        cols['window_440'] = np.exp(-((temp - 440.0) / 8.0) ** 2)
        cols['window_440_shorttime'] = cols['window_440'] * np.exp(-((log_time - np.log1p(1.0)) / 0.35) ** 2)
        cols['window_440_longtime'] = cols['window_440'] * np.exp(-((log_time - np.log1p(24.0)) / 0.35) ** 2)

        # ========= 6) 物理基线预测作为特征 =========
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []
        for t, h in zip(temp, time):
            bs, bt, by = self.physics_baseline(t, h)
            baseline_strain.append(bs)
            baseline_tensile.append(bt)
            baseline_yield.append(by)

        baseline_strain = np.array(baseline_strain)
        baseline_tensile = np.array(baseline_tensile)
        baseline_yield = np.array(baseline_yield)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # ========= 7) 相对基线/相对机制边界特征 =========
        # 让模型更容易学习“相对于物理先验的偏差”
        cols['strain_gain_over_cast'] = baseline_strain - p['as_cast_strain']
        cols['tensile_gain_over_cast'] = baseline_tensile - p['as_cast_tensile']
        cols['yield_gain_over_cast'] = baseline_yield - p['as_cast_yield']

        cols['temp_relative_to_shift'] = (temp - p['critical_temp_regime_shift']) / p['temp_scale']
        cols['time_relative_to_12'] = (time - p['critical_time_regime_shift']) / p['time_scale']
        cols['distance_to_peak_temp'] = np.abs(temp - p['critical_temp_peak_window']) / 10.0
        cols['distance_to_peak_time_log'] = np.abs(log_time - np.log1p(p['critical_time_regime_shift']))

        # 基线与动力学特征交互：帮助下游模型学习“同样baseline下，不同动力学路径导致的差异”
        cols['baseline_tensile_x_arrhenius'] = baseline_tensile * arrhenius * 1e12
        cols['baseline_yield_x_arrhenius'] = baseline_yield * arrhenius * 1e12
        cols['baseline_strain_x_log_time'] = baseline_strain * log_time

        # ========= 输出 =========
        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names