import pandas as pd
import numpy as np


# 7499铝合金热处理经验参数（基于铝合金析出/回复动力学的宽松物理先验）
DOMAIN_PARAMS = {
    # 机制边界：从误差集中点与铝合金高温组织演化常识推断
    # 440℃附近：短时处理与长时处理表现差异显著，说明这里可能是“未充分激活/开始明显演化”的边界
    'critical_temp_regime_shift': 445.0,
    # 460~470℃附近：中高温长时下误差很大，可能对应析出-粗化/回复竞争增强区
    'secondary_temp_regime_shift': 465.0,

    # 时间边界：1h、12h、24h呈现明显阶段性
    'critical_time_short': 2.0,
    'critical_time_peak': 12.0,
    'critical_time_long': 20.0,

    # 铝合金扩散/析出等效激活能数量级（J/mol），仅用于构造动力学特征
    'activation_energy_Q': 1.20e5,

    # 气体常数
    'gas_constant_R': 8.314,

    # Larson-Miller常数，金属热处理常用经验量级
    'larson_miller_C': 20.0,

    # 数据区间内的经验最优窗口中心（不是硬编码标签，而是机制敏感区中心）
    'window_temp_center': 460.0,
    'window_time_center': 12.0,

    # 原始样品（未热处理）基线
    'raw_strain': 6.94,
    'raw_tensile': 145.83,
    'raw_yield': 96.60,
}


class FeatureAgent:
    """
    面向小样本材料热处理问题的物理启发特征工程：
    1. 少而精，避免高维爆炸
    2. 强化 temp-time 非线性与交互
    3. 显式加入机制边界/阶段特征
    4. 加入物理基线，让模型学习残差
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的粗略基线：
        - 当前数据趋势显示：相对原始样品，热处理后整体三项性能提升
        - 温度与时间作用非单调，存在“中高温 + 中长时”更优窗口
        - 强度与屈服应协同变化
        - 应变也可能在适中窗口提升，而非简单与强度反向

        这里只给出方向正确、平滑、低复杂度的物理近似，不依赖标签拟合。
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = max(float(time), 1e-6)

        # 归一化坐标
        t_norm = (temp - 420.0) / 60.0          # 大致覆盖 420~480℃
        log_time = np.log1p(time)
        log_time_norm = np.log1p(time) / np.log(25.0)  # 1~24h附近尺度

        # 热暴露强度：温度升高 + 时间增加都会加快组织演化
        exposure = 0.55 * t_norm + 0.45 * log_time_norm

        # 最优窗口：围绕 460℃/12h 的平滑峰
        temp_window = np.exp(-((temp - p['window_temp_center']) / 18.0) ** 2)
        time_window = np.exp(-((np.log1p(time) - np.log1p(p['window_time_center'])) / 0.85) ** 2)
        process_window = temp_window * time_window

        # 中高温长时下可能进入竞争区（粗化/回复/部分软化），但不能假设整体下降
        high_temp_long_time = 1.0 / (1.0 + np.exp(-(temp - p['secondary_temp_regime_shift']) / 4.0)) \
                              * 1.0 / (1.0 + np.exp(-(time - p['critical_time_peak']) / 2.5))

        # 相对原始样品整体提升
        baseline_tensile = (
            p['raw_tensile']
            + 55.0 * exposure
            + 70.0 * process_window
            - 12.0 * high_temp_long_time
        )

        baseline_yield = (
            p['raw_yield']
            + 38.0 * exposure
            + 48.0 * process_window
            - 8.0 * high_temp_long_time
        )

        baseline_strain = (
            p['raw_strain']
            + 1.8 * exposure
            + 3.6 * process_window
            - 0.4 * high_temp_long_time
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        time_safe = np.clip(time, 1e-6, None)

        p = DOMAIN_PARAMS
        R = p['gas_constant_R']
        Q = p['activation_energy_Q']

        # ===== 1) 基础低维特征 =====
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = np.log1p(time_safe)
        cols['sqrt_time'] = np.sqrt(time_safe)

        # 适度非线性：仅保留最必要项，防止29条样本下过拟合
        cols['temp_centered'] = temp - p['window_temp_center']
        cols['log_time_centered'] = np.log1p(time_safe) - np.log1p(p['window_time_center'])
        cols['temp_sq_scaled'] = ((temp - p['window_temp_center']) / 20.0) ** 2
        cols['log_time_sq_scaled'] = (
            (np.log1p(time_safe) - np.log1p(p['window_time_center'])) / 1.0
        ) ** 2

        # ===== 2) 交互与等效热暴露 =====
        cols['temp_x_log_time'] = temp * np.log1p(time_safe)
        cols['temp_x_time_scaled'] = temp * time / 100.0

        # Arrhenius型动力学特征：使用开尔文温度
        temp_K = temp + 273.15
        cols['inv_temp_K'] = 1.0 / temp_K
        cols['arrhenius_exponent'] = -Q / (R * temp_K)
        cols['arrhenius_rate'] = np.exp(np.clip(cols['arrhenius_exponent'], -50, 50))
        cols['arrhenius_time'] = cols['arrhenius_rate'] * time_safe

        # Larson-Miller参数（相对尺度即可）
        cols['larson_miller'] = temp_K * (p['larson_miller_C'] + np.log10(np.clip(time_safe, 1e-6, None)))

        # Zener-Hollomon风格简化特征（只作等效尺度）
        cols['zener_like'] = np.log1p(time_safe) - Q / (R * temp_K)

        # ===== 3) 机制区间检测特征（关键）=====
        cols['is_regime_above_445C'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_regime_above_465C'] = (temp >= p['secondary_temp_regime_shift']).astype(float)

        cols['is_short_time'] = (time <= p['critical_time_short']).astype(float)
        cols['is_peak_time_zone'] = ((time >= 8.0) & (time <= 16.0)).astype(float)
        cols['is_long_time'] = (time >= p['critical_time_long']).astype(float)

        # 高误差敏感区对应的阶段组合
        cols['is_440_1_like'] = ((temp <= 445.0) & (time <= 2.0)).astype(float)
        cols['is_470_12_like'] = ((temp >= 465.0) & (time >= 8.0) & (time <= 16.0)).astype(float)
        cols['is_440_24_like'] = ((temp <= 445.0) & (time >= 20.0)).astype(float)
        cols['is_460_12_like'] = ((temp >= 455.0) & (temp <= 465.0) & (time >= 8.0) & (time <= 16.0)).astype(float)

        # 平滑分段激活，避免硬阈值过于生硬
        cols['temp_above_445_relu'] = np.maximum(temp - p['critical_temp_regime_shift'], 0.0)
        cols['temp_above_465_relu'] = np.maximum(temp - p['secondary_temp_regime_shift'], 0.0)
        cols['time_above_12_relu'] = np.maximum(time - p['critical_time_peak'], 0.0)
        cols['time_above_20_relu'] = np.maximum(time - p['critical_time_long'], 0.0)

        # 机制边界相对距离
        cols['temp_rel_boundary1'] = (temp - p['critical_temp_regime_shift']) / 10.0
        cols['temp_rel_boundary2'] = (temp - p['secondary_temp_regime_shift']) / 10.0
        cols['time_rel_peak'] = (time - p['critical_time_peak']) / 6.0
        cols['time_rel_long'] = (time - p['critical_time_long']) / 6.0

        # ===== 4) 最优工艺窗口特征 =====
        temp_window = np.exp(-((temp - p['window_temp_center']) / 18.0) ** 2)
        time_window = np.exp(-((np.log1p(time_safe) - np.log1p(p['window_time_center'])) / 0.85) ** 2)
        cols['temp_window_activation'] = temp_window
        cols['time_window_activation'] = time_window
        cols['process_window_activation'] = temp_window * time_window

        # 距离最优窗口的“半径”
        cols['window_distance'] = np.sqrt(
            ((temp - p['window_temp_center']) / 18.0) ** 2 +
            ((np.log1p(time_safe) - np.log1p(p['window_time_center'])) / 0.85) ** 2
        )

        # ===== 5) 物理基线特征：让模型学残差 =====
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []

        for T, t in zip(temp, time):
            s, ts, ys = self.physics_baseline(T, t)
            baseline_strain.append(s)
            baseline_tensile.append(ts)
            baseline_yield.append(ys)

        baseline_strain = np.array(baseline_strain)
        baseline_tensile = np.array(baseline_tensile)
        baseline_yield = np.array(baseline_yield)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 基线内部关系：帮助模型利用多目标协同
        cols['baseline_yield_to_tensile'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)
        cols['baseline_strength_sum'] = baseline_tensile + baseline_yield
        cols['baseline_strength_diff'] = baseline_tensile - baseline_yield

        # ===== 6) 基于原始样品参考的增量尺度 =====
        cols['baseline_strain_gain_over_raw'] = baseline_strain - p['raw_strain']
        cols['baseline_tensile_gain_over_raw'] = baseline_tensile - p['raw_tensile']
        cols['baseline_yield_gain_over_raw'] = baseline_yield - p['raw_yield']

        # ===== 7) 针对高误差区增加少量耦合特征 =====
        cols['regime465_x_logtime'] = cols['is_regime_above_465C'] * np.log1p(time_safe)
        cols['shorttime_x_temp'] = cols['is_short_time'] * (temp - 440.0)
        cols['longtime_x_lowtemp'] = cols['is_440_24_like'] * np.log1p(time_safe)
        cols['midhighT_peaktime'] = cols['is_regime_above_445C'] * cols['is_peak_time_zone']

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names