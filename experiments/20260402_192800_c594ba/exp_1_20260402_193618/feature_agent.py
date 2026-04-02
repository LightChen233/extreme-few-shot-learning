import pandas as pd
import numpy as np


# 7499铝合金热处理问题的领域参数
# 说明：
# - 数据很少（29条），因此只保留少量、强物理含义的特征
# - 温度范围约在 420~480℃，时间范围约在 1~24 h
# - 从误差分布看，440℃/1h 与 470℃/12h 是关键难点，提示存在明显机制切换/最优窗口
# - 这里不强行假设“高温长时一定软化”，而是采用“强化启动区 -> 协同优化区 -> 高热暴露粗化区”的温和分段

DOMAIN_PARAMS = {
    # 机制边界：低于此温度，强化/组织转变不足；高于此温度后热激活显著增强
    'critical_temp_regime_shift': 450.0,

    # 中高温窗口中心：数据中 460~470℃、12h 附近误差大，说明此处可能接近组织最敏感区
    'optimal_temp_center': 462.0,

    # 高热暴露边界：接近此温度且时间较长时，可能出现粗化/过热暴露效应
    'high_temp_boundary': 468.0,

    # 时间边界：1h 往往不足，12h 左右是显著演化节点，24h 可能进入高暴露区
    'critical_time_regime_shift': 6.0,
    'optimal_time_center': 12.0,
    'long_time_boundary': 18.0,

    # 激活能：铝合金中析出/扩散控制过程常见量级，取一个温和经验值
    # 仅用于构造动力学特征，不追求精确热力学拟合
    'activation_energy_Q': 120000.0,  # J/mol

    # 气体常数
    'gas_constant_R': 8.314,

    # Larson-Miller 常数，工程上常见经验值
    'larson_miller_C': 20.0,

    # 用于构造“最佳热暴露窗口”的等效热暴露中心
    # 采用 temp * log1p(time) 的简单经验尺度
    'exposure_center': 462.0 * np.log1p(12.0),
}


class FeatureAgent:
    """
    面向小样本材料热处理问题的特征工程：
    1. 少而精，避免高维过拟合
    2. 强调温度-时间交互
    3. 引入机制边界、动力学等效量、物理基线
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于材料热处理常识的弱物理基线，不依赖训练标签。
        设计原则：
        - 相比原始样品，热处理后整体三项性能倾向于提升
        - 1h 为不足时效/不足组织演化区
        - 约 450~465℃、12h 附近接近协同优化区
        - 更高热暴露下不强行假设剧烈下降，只施加轻微平台/钝化
        """
        # 原始样品
        base_strain = 6.94
        base_tensile = 145.83
        base_yield = 96.60

        T = float(temp)
        t = max(float(time), 1e-6)

        # 温度、时间归一化
        temp_norm = (T - 420.0) / 60.0          # roughly 0~1 in current range
        time_log = np.log1p(t) / np.log1p(24.0) # 0~1

        # 强化启动：温度和时间共同促进
        activation = 1.0 / (1.0 + np.exp(-(T - 448.0) / 7.0))
        time_progress = 1.0 - np.exp(-t / 5.5)

        # 最优窗口增强：460~465℃、约12h附近
        temp_peak = np.exp(-((T - DOMAIN_PARAMS['optimal_temp_center']) / 11.0) ** 2)
        time_peak = np.exp(-((np.log1p(t) - np.log1p(DOMAIN_PARAMS['optimal_time_center'])) / 0.65) ** 2)
        synergy_peak = temp_peak * time_peak

        # 高热暴露轻微钝化：只做温和修正，避免与数据趋势冲突
        high_exposure = 1.0 / (1.0 + np.exp(-(T - 470.0) / 4.0)) * \
                        1.0 / (1.0 + np.exp(-(t - 16.0) / 3.0))

        # 应变：数据提示并非与强度对立，常在中高热处理下同步改善
        strain = (
            base_strain
            + 4.2 * activation
            + 3.2 * time_progress
            + 2.2 * synergy_peak
            - 0.6 * high_exposure
        )

        # 抗拉强度：整体升高，在最优窗口附近更强，高暴露区轻微平台化
        tensile = (
            base_tensile
            + 52.0 * activation
            + 38.0 * time_progress
            + 24.0 * synergy_peak
            - 8.0 * high_exposure
        )

        # 屈服强度：与抗拉强度强相关，但对组织状态更敏感
        yld = (
            base_yield
            + 34.0 * activation
            + 28.0 * time_progress
            + 18.0 * synergy_peak
            - 7.0 * high_exposure
        )

        return strain, tensile, yld

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        # 防止数值问题
        time_safe = np.clip(time, 1e-6, None)
        temp_K = temp + 273.15

        # -------------------------
        # 1) 基础工艺特征（少量保留）
        # -------------------------
        cols['temp'] = temp
        cols['time'] = time
        cols['log_time'] = np.log1p(time_safe)
        cols['sqrt_time'] = np.sqrt(time_safe)

        # 非线性项：仅保留最必要的二次项
        cols['temp_dev_opt'] = temp - DOMAIN_PARAMS['optimal_temp_center']
        cols['log_time_dev_opt'] = np.log1p(time_safe) - np.log1p(DOMAIN_PARAMS['optimal_time_center'])
        cols['temp_dev_opt_sq'] = cols['temp_dev_opt'] ** 2
        cols['log_time_dev_opt_sq'] = cols['log_time_dev_opt'] ** 2

        # 温度-时间交互
        cols['temp_x_log_time'] = temp * np.log1p(time_safe)

        # -------------------------
        # 2) 机制区间检测特征（关键）
        # -------------------------
        Tc = DOMAIN_PARAMS['critical_temp_regime_shift']
        Th = DOMAIN_PARAMS['high_temp_boundary']
        tc = DOMAIN_PARAMS['critical_time_regime_shift']
        tl = DOMAIN_PARAMS['long_time_boundary']

        cols['is_low_temp_regime'] = (temp < Tc).astype(float)
        cols['is_activated_temp_regime'] = (temp >= Tc).astype(float)
        cols['is_high_temp_regime'] = (temp >= Th).astype(float)

        cols['is_short_time_regime'] = (time <= tc).astype(float)
        cols['is_mid_time_regime'] = ((time > tc) & (time < tl)).astype(float)
        cols['is_long_time_regime'] = (time >= tl).astype(float)

        cols['is_underprocessed'] = ((temp < Tc) & (time <= tc)).astype(float)
        cols['is_synergy_window'] = (
            (temp >= 455.0) & (temp <= 468.0) &
            (time >= 8.0) & (time <= 16.0)
        ).astype(float)
        cols['is_high_exposure_window'] = ((temp >= Th) & (time >= tl)).astype(float)

        # 到边界的相对距离/分段激活
        cols['temp_above_critical'] = np.maximum(temp - Tc, 0.0)
        cols['temp_above_high'] = np.maximum(temp - Th, 0.0)
        cols['time_above_critical'] = np.maximum(time - tc, 0.0)
        cols['time_above_long'] = np.maximum(time - tl, 0.0)

        cols['temp_below_critical'] = np.maximum(Tc - temp, 0.0)
        cols['time_below_critical'] = np.maximum(tc - time, 0.0)

        # -------------------------
        # 3) 动力学等效特征
        # -------------------------
        R = DOMAIN_PARAMS['gas_constant_R']
        Q = DOMAIN_PARAMS['activation_energy_Q']
        C = DOMAIN_PARAMS['larson_miller_C']

        # Arrhenius型速率尺度：exp(-Q/RT)
        arrhenius_rate = np.exp(-Q / (R * temp_K))
        cols['arrhenius_rate'] = arrhenius_rate

        # 时间-温度综合热暴露
        cols['arrhenius_exposure'] = arrhenius_rate * time_safe
        cols['log_arrhenius_exposure'] = np.log1p(cols['arrhenius_exposure'])

        # Larson-Miller Parameter
        lmp = temp_K * (C + np.log10(time_safe))
        cols['larson_miller'] = lmp / 1000.0  # 缩放便于模型学习

        # 简化热暴露指标：适合小样本
        exposure = temp * np.log1p(time_safe)
        cols['thermal_exposure'] = exposure
        cols['exposure_dev_center'] = exposure - DOMAIN_PARAMS['exposure_center']
        cols['exposure_dev_center_sq'] = cols['exposure_dev_center'] ** 2

        # 最优窗口型软激活特征
        cols['opt_window_temp_kernel'] = np.exp(-((temp - DOMAIN_PARAMS['optimal_temp_center']) / 10.0) ** 2)
        cols['opt_window_time_kernel'] = np.exp(
            -((np.log1p(time_safe) - np.log1p(DOMAIN_PARAMS['optimal_time_center'])) / 0.6) ** 2
        )
        cols['opt_window_synergy'] = cols['opt_window_temp_kernel'] * cols['opt_window_time_kernel']

        # -------------------------
        # 4) 物理基线特征（让模型学残差）
        # -------------------------
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

        # 基线派生：输出间物理关系
        cols['baseline_yield_to_tensile'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)
        cols['baseline_strength_sum'] = baseline_tensile + baseline_yield

        # 基线与机制边界结合的残差驱动特征（不是标签残差）
        cols['baseline_tensile_x_synergy'] = baseline_tensile * cols['is_synergy_window']
        cols['baseline_yield_x_high_exposure'] = baseline_yield * cols['is_high_exposure_window']
        cols['baseline_strain_x_underprocessed'] = baseline_strain * cols['is_underprocessed']

        # -------------------------
        # 5) 针对高误差条件的局部敏感特征
        # -------------------------
        # 440℃/1h：显著高估，提示“低温短时不足演化”需更明确编码
        cols['dist_to_440_1'] = np.sqrt(((temp - 440.0) / 10.0) ** 2 + ((time - 1.0) / 3.0) ** 2)
        cols['near_440_1'] = np.exp(-cols['dist_to_440_1'] ** 2)

        # 470℃/12h：显著低估，提示“中高温-中时协同峰值”需单独编码
        cols['dist_to_470_12'] = np.sqrt(((temp - 470.0) / 10.0) ** 2 + ((time - 12.0) / 6.0) ** 2)
        cols['near_470_12'] = np.exp(-cols['dist_to_470_12'] ** 2)

        # 460℃/12h、440℃/24h 也是有代表性的机制点
        cols['dist_to_460_12'] = np.sqrt(((temp - 460.0) / 10.0) ** 2 + ((time - 12.0) / 6.0) ** 2)
        cols['near_460_12'] = np.exp(-cols['dist_to_460_12'] ** 2)

        cols['dist_to_440_24'] = np.sqrt(((temp - 440.0) / 10.0) ** 2 + ((time - 24.0) / 8.0) ** 2)
        cols['near_440_24'] = np.exp(-cols['dist_to_440_24'] ** 2)

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names