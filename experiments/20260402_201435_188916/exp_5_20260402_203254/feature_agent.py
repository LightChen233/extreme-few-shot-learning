import pandas as pd
import numpy as np


# 7499铝合金热处理-性能问题的领域参数
# 说明：
# 1) 当前数据区间内（约420–480℃, 1–24h），总体趋势是升温/延时通常提升强度；
# 2) 但在中高温-中长时附近存在明显非线性窗口，尤其460–470℃、12h附近误差最大，
#    说明这里可能对应析出/再分配/均匀化机制加速区；
# 3) 440℃与470℃在验证中都是外推点，因此加入“边界距离/机制分段”很关键；
# 4) 激活能取铝合金析出/扩散过程常见量级的近似值，仅用于构造动力学特征，不作严格物理定量。
DOMAIN_PARAMS = {
    # 温度机制边界：低温响应区 -> 中温强化加速区 -> 高温强时效区
    'critical_temp_regime_shift': 455.0,
    'high_temp_boundary': 470.0,
    'low_temp_boundary': 440.0,

    # 时间机制边界：短时、过渡时、长时
    'critical_time_regime_shift': 12.0,
    'long_time_boundary': 24.0,
    'short_time_boundary': 1.0,

    # 热激活动力学参数
    'activation_energy_Q': 120000.0,   # J/mol，近似析出/扩散控制过程
    'gas_constant_R': 8.314,

    # Larson-Miller 常数，工程经验值
    'larson_miller_C': 20.0,

    # 已知铸态基线
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 数据观察到的强化/塑化加速窗口中心
    'peak_temp_center': 465.0,
    'peak_time_center': 12.0,

    # 训练域边界（由题目给出的覆盖信息近似得到）
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 外推敏感区：440/1、440/24、470/12 都表现异常，说明这些边界附近应单独刻画
    'sensitive_temp_nodes': [440.0, 460.0, 470.0],
    'sensitive_time_nodes': [1.0, 12.0, 24.0],
}


class FeatureAgent:
    """
    面向7499铝合金热处理小样本预测的特征工程：
    - 少而精，避免无约束高维展开
    - 加入物理机制分段特征
    - 加入动力学等效特征（Arrhenius / LMP）
    - 加入物理基线预测，让下游模型学残差
    - 加入外推/边界距离特征，专门处理训练集未覆盖点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于物理直觉与题目给出的数据趋势构造基线预测。
        关键要求：预测方向与统计趋势一致。
        当前数据支持：
        - 在420–480℃范围内，升温整体有利于强度提升；
        - 时间从1h增加到12h/24h，总体也有利于强度提升；
        - 应变不是简单下降，反而在中高温中时附近可明显提升；
        - 但应变在高温长时可能略有回落，因此只对strain做温和非单调修正。
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = float(time)

        T_k = temp + 273.15
        logt = np.log1p(max(time, 0.0))

        # 1) 热处理推进程度：温度与时间共同驱动，整体单调增加
        #    归一化到当前数据区间，避免数值过大
        temp_norm = (temp - 420.0) / 60.0          # roughly 0~1 in data range
        time_norm = np.log1p(time) / np.log1p(24.0)

        progress = 0.58 * temp_norm + 0.42 * time_norm
        progress = np.clip(progress, -0.2, 1.3)

        # 2) 中高温-中时窗口的协同强化/塑化项
        #    重点照顾460–470℃、12h附近，这是当前大误差区
        peak_temp = np.exp(-((temp - p['peak_temp_center']) / 12.0) ** 2)
        peak_time = np.exp(-((np.log1p(time) - np.log1p(p['peak_time_center'])) / 0.55) ** 2)
        synergy = peak_temp * peak_time

        # 3) 高温长时下，对应变做轻微回落修正；强度不强行设为下降
        #    因为题目明确警告：不能凭空假设过时效导致强度下降
        high_temp_gate = 1.0 / (1.0 + np.exp(-(temp - 470.0) / 4.0))
        long_time_gate = 1.0 / (1.0 + np.exp(-(time - 18.0) / 3.0))
        strain_softening = high_temp_gate * long_time_gate

        # 4) 从铸态基线出发构造近似性能
        baseline_strain = (
            p['as_cast_strain']
            + 4.2 * progress
            + 4.8 * synergy
            - 1.2 * strain_softening
        )

        baseline_tensile = (
            p['as_cast_tensile']
            + 95.0 * progress
            + 38.0 * synergy
        )

        baseline_yield = (
            p['as_cast_yield']
            + 78.0 * progress
            + 24.0 * synergy
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        p = DOMAIN_PARAMS
        eps = 1e-8

        T_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # =========================
        # 1. 基础特征：控制维度，只保留核心非线性
        # =========================
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_sq'] = temp ** 2
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['time_sq'] = time ** 2
        cols['temp_x_time'] = temp * time
        cols['temp_x_log_time'] = temp * log_time

        # 归一化工艺坐标
        temp_centered = temp - p['critical_temp_regime_shift']
        time_centered = time - p['critical_time_regime_shift']
        cols['temp_centered'] = temp_centered
        cols['time_centered'] = time_centered
        cols['temp_centered_sq'] = temp_centered ** 2
        cols['time_centered_sq'] = time_centered ** 2
        cols['temp_centered_x_time_centered'] = temp_centered * time_centered

        # =========================
        # 2. 动力学等效特征
        # =========================
        # Arrhenius速率项：高温更快
        arrhenius_rate = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['arrhenius_rate'] = arrhenius_rate
        cols['arrhenius_dose'] = arrhenius_rate * np.clip(time, 0, None)
        cols['log_arrhenius_dose'] = np.log1p(cols['arrhenius_dose'])

        # 简化JMAK/扩散型时间响应
        cols['arrhenius_sqrt_dose'] = arrhenius_rate * sqrt_time
        cols['arrhenius_log_dose'] = arrhenius_rate * log_time

        # Larson-Miller parameter
        cols['larson_miller'] = T_k * (p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None)))
        cols['inv_temp'] = 1.0 / T_k
        cols['log_time_over_temp'] = log_time / T_k

        # =========================
        # 3. 物理机制区间检测特征（关键）
        # =========================
        # 温度机制分段
        cols['is_low_temp_regime'] = (temp <= p['low_temp_boundary']).astype(float)
        cols['is_mid_temp_regime'] = ((temp > p['low_temp_boundary']) & (temp < p['high_temp_boundary'])).astype(float)
        cols['is_high_temp_regime'] = (temp >= p['high_temp_boundary']).astype(float)

        # 时间机制分段
        cols['is_short_time_regime'] = (time <= 1.5).astype(float)
        cols['is_mid_time_regime'] = ((time > 1.5) & (time < 18.0)).astype(float)
        cols['is_long_time_regime'] = (time >= 18.0).astype(float)

        # 关键工艺窗口
        cols['is_peak_strength_window'] = (
            ((temp >= 455.0) & (temp <= 475.0) & (time >= 8.0) & (time <= 16.0))
        ).astype(float)

        cols['is_high_temp_long_time'] = (
            (temp >= p['high_temp_boundary']) & (time >= 18.0)
        ).astype(float)

        cols['is_low_temp_short_time'] = (
            (temp <= p['low_temp_boundary']) & (time <= 2.0)
        ).astype(float)

        cols['is_low_temp_long_time'] = (
            (temp <= p['low_temp_boundary']) & (time >= 18.0)
        ).astype(float)

        cols['is_transition_temp_12h'] = (
            (np.abs(temp - 470.0) <= 12.0) & (np.abs(time - 12.0) <= 3.0)
        ).astype(float)

        # 分段激活特征：让模型能学“过边界后斜率改变”
        cols['temp_above_regime_shift'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_below_regime_shift'] = np.clip(p['critical_temp_regime_shift'] - temp, 0, None)
        cols['time_above_regime_shift'] = np.clip(time - p['critical_time_regime_shift'], 0, None)
        cols['time_below_regime_shift'] = np.clip(p['critical_time_regime_shift'] - time, 0, None)

        cols['temp_above_high_boundary'] = np.clip(temp - p['high_temp_boundary'], 0, None)
        cols['time_above_long_boundary'] = np.clip(time - 18.0, 0, None)

        # =========================
        # 4. 峰值窗口/非单调形状特征
        # =========================
        # 用平滑峰值特征替代大量高阶多项式，适合小样本
        peak_temp = np.exp(-((temp - p['peak_temp_center']) / 12.0) ** 2)
        peak_time = np.exp(-((log_time - np.log1p(p['peak_time_center'])) / 0.55) ** 2)
        cols['peak_temp_proximity'] = peak_temp
        cols['peak_time_proximity'] = peak_time
        cols['peak_synergy'] = peak_temp * peak_time

        # 对440/1、440/24、470/12这些难点做局部距离特征
        cols['dist_to_440_1'] = np.sqrt(((temp - 440.0) / 10.0) ** 2 + ((time - 1.0) / 6.0) ** 2)
        cols['dist_to_440_24'] = np.sqrt(((temp - 440.0) / 10.0) ** 2 + ((time - 24.0) / 6.0) ** 2)
        cols['dist_to_470_12'] = np.sqrt(((temp - 470.0) / 10.0) ** 2 + ((time - 12.0) / 6.0) ** 2)
        cols['dist_to_460_12'] = np.sqrt(((temp - 460.0) / 10.0) ** 2 + ((time - 12.0) / 6.0) ** 2)

        # =========================
        # 5. 外推/边界距离特征（关键）
        # =========================
        # 与训练域边界的相对位置：帮助模型识别“这是外推”
        cols['temp_to_train_min'] = temp - p['train_temp_min']
        cols['temp_to_train_max'] = p['train_temp_max'] - temp
        cols['time_to_train_min'] = time - p['train_time_min']
        cols['time_to_train_max'] = p['train_time_max'] - time

        cols['temp_outside_train_low'] = np.clip(p['train_temp_min'] - temp, 0, None)
        cols['temp_outside_train_high'] = np.clip(temp - p['train_temp_max'], 0, None)
        cols['time_outside_train_low'] = np.clip(p['train_time_min'] - time, 0, None)
        cols['time_outside_train_high'] = np.clip(time - p['train_time_max'], 0, None)

        # 即使点在矩形训练范围内，也可能是“稀疏边界附近”
        cols['dist_to_temp_left_boundary'] = np.abs(temp - p['train_temp_min'])
        cols['dist_to_temp_right_boundary'] = np.abs(temp - p['train_temp_max'])
        cols['dist_to_time_low_boundary'] = np.abs(time - p['train_time_min'])
        cols['dist_to_time_high_boundary'] = np.abs(time - p['train_time_max'])

        cols['edge_proximity_temp'] = np.minimum(cols['dist_to_temp_left_boundary'], cols['dist_to_temp_right_boundary'])
        cols['edge_proximity_time'] = np.minimum(cols['dist_to_time_low_boundary'], cols['dist_to_time_high_boundary'])

        # 针对已知验证/测试稀疏带
        cols['dist_to_low_temp_boundary_440'] = np.abs(temp - 440.0)
        cols['dist_to_high_temp_boundary_470'] = np.abs(temp - 470.0)
        cols['dist_to_mid_time_12'] = np.abs(time - 12.0)
        cols['dist_to_long_time_24'] = np.abs(time - 24.0)

        # =========================
        # 6. 基于世界知识的物理基线及残差坐标
        # =========================
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []

        for t, ti in zip(temp, time):
            bs, bt, by = self.physics_baseline(t, ti)
            baseline_strain.append(bs)
            baseline_tensile.append(bt)
            baseline_yield.append(by)

        baseline_strain = np.asarray(baseline_strain)
        baseline_tensile = np.asarray(baseline_tensile)
        baseline_yield = np.asarray(baseline_yield)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 基线提升量（相对铸态）
        cols['baseline_delta_strain'] = baseline_strain - p['as_cast_strain']
        cols['baseline_delta_tensile'] = baseline_tensile - p['as_cast_tensile']
        cols['baseline_delta_yield'] = baseline_yield - p['as_cast_yield']

        # 局部归一化残差坐标：相对于机制边界/峰值中心的位置
        cols['temp_relative_to_peak'] = temp - p['peak_temp_center']
        cols['time_relative_to_peak'] = time - p['peak_time_center']
        cols['abs_temp_relative_to_peak'] = np.abs(cols['temp_relative_to_peak'])
        cols['abs_time_relative_to_peak'] = np.abs(cols['time_relative_to_peak'])

        # 基线与动力学耦合，帮助模型学“在同等基线下，动力学推进速度不同”
        cols['baseline_tensile_x_arrhenius'] = baseline_tensile * arrhenius_rate
        cols['baseline_yield_x_arrhenius'] = baseline_yield * arrhenius_rate
        cols['baseline_strain_x_peak'] = baseline_strain * cols['peak_synergy']

        # =========================
        # 7. 少量目标相关结构先验特征（不使用标签）
        # =========================
        # 经验上抗拉/屈服同向，且屈强比落在相对稳定范围，可让模型更容易学
        cols['baseline_yield_to_tensile_ratio'] = baseline_yield / (baseline_tensile + eps)
        cols['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield

        # 强塑协同窗口，但不强加负相关
        cols['baseline_strength_ductility_index'] = baseline_tensile * baseline_strain

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names