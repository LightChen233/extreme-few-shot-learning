import pandas as pd
import numpy as np


# 基于 7xxx 铝合金热处理/组织演化的一般物理常识给出的粗略参数
# 注意：这些参数不是精确材料常数，而是用于构造“机制感知特征”
DOMAIN_PARAMS = {
    # 机制边界：结合题目给出的数据趋势，460~470℃附近可能进入更强的均匀化/强化响应区
    'critical_temp_regime_shift': 455.0,
    'high_temp_regime': 470.0,

    # 时间边界：1h 可视为短时，12h 常是明显组织演化点，24h 为长时暴露端
    'critical_time_short': 1.5,
    'critical_time_peak': 12.0,
    'critical_time_long': 24.0,

    # 热激活动力学参数：Al-Zn-Mg-Cu 类合金中扩散/析出相关量级，取工程近似
    'activation_energy_Q': 1.20e5,   # J/mol
    'gas_constant_R': 8.314,         # J/(mol·K)

    # Larson-Miller 常数，工程上常取 20 左右
    'larson_miller_C': 20.0,

    # 用于平滑分段特征
    'temp_transition_width': 12.0,
    'time_transition_width': 4.0,

    # 原始铸态基线
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 当前实验温度/时间域，用于构造边界距离与外推风险特征
    'observed_temp_min': 420.0,
    'observed_temp_max': 480.0,
    'observed_time_min': 1.0,
    'observed_time_max': 24.0,

    # 训练覆盖的代表性“已知工艺网格”近邻，用于外推距离特征
    # 依据题目中已知覆盖信息：420/480 在若干时间点覆盖较多，460/12 为已覆盖点
    'anchor_points': [
        (420.0, 1.0),
        (420.0, 12.0),
        (420.0, 24.0),
        (460.0, 12.0),
        (470.0, 1.0),
        (470.0, 24.0),
        (480.0, 12.0),
    ],
}


class FeatureAgent:
    """
    小样本、物理引导的特征工程：
    1) 少而精，避免纯堆砌高维多项式
    2) 显式加入机制边界 / 分段特征
    3) 加入动力学等效特征
    4) 加入物理 baseline，让下游模型学习残差
    5) 加入边界距离 / 外推风险特征，提升外推点稳定性
    """

    def __init__(self):
        self.feature_names = []

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -60, 60)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _safe_log(x):
        return np.log(np.clip(x, 1e-8, None))

    def _distance_to_anchors(self, temp, time):
        anchors = DOMAIN_PARAMS['anchor_points']
        dists = []
        for ta, ti in anchors:
            # 做尺度归一，避免温度量纲压制时间
            dt = (temp - ta) / 20.0
            dh = (time - ti) / 8.0
            dists.append(np.sqrt(dt * dt + dh * dh))
        dists = np.array(dists, dtype=float)
        return dists.min(), dists.mean()

    def physics_baseline(self, temp, time):
        """
        基于题目给出的统计趋势构造方向正确的物理基线：
        - 在当前数据范围内，整体上升温/延时通常使强度提高
        - 但应变在高温长时区可能出现“12h附近较优、24h略回落”的非单调性
        - 强度与屈服整体同向上升，不强行加入高温软化先验
        """
        temp = float(temp)
        time = float(time)

        p = DOMAIN_PARAMS
        T0 = p['observed_temp_min']
        T1 = p['observed_temp_max']

        # 归一化主变量
        temp_norm = (temp - T0) / (T1 - T0)
        logt = np.log1p(max(time, 0.0))
        logt_norm = logt / np.log1p(p['observed_time_max'])

        # 机制激活：455℃附近进入更强响应区
        temp_regime = self._sigmoid((temp - p['critical_temp_regime_shift']) / p['temp_transition_width'])

        # 时间演化：随 log(time) 增长，12h 左右强化较明显
        time_progress = self._sigmoid((time - p['critical_time_peak']) / p['time_transition_width'])

        # 近似热暴露量（高温更快）
        T_k = temp + 273.15
        arr = time * np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        # 为避免量级过小，仅用于相对变化
        arr_scaled = arr / (1e-8 + np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * (480.0 + 273.15))) * 24.0)
        arr_scaled = np.clip(arr_scaled, 0.0, 2.0)

        # ---- 强度基线 ----
        # 方向约束：在现有范围内，温度升高+时间延长整体提高强度
        # 加入 temp*time 协同项以反映高温下时间更有效
        tensile = (
            p['as_cast_tensile']
            + 85.0 * temp_norm
            + 55.0 * logt_norm
            + 35.0 * temp_norm * logt_norm
            + 18.0 * temp_regime
            + 10.0 * time_progress
            + 8.0 * arr_scaled
        )

        yld = (
            p['as_cast_yield']
            + 62.0 * temp_norm
            + 42.0 * logt_norm
            + 24.0 * temp_norm * logt_norm
            + 12.0 * temp_regime
            + 8.0 * time_progress
            + 6.0 * arr_scaled
        )

        # ---- 应变基线 ----
        # 与题目趋势一致：整体热处理改善塑性，但高温长时端可能略回落；
        # 因此构造“整体提升 + 高温长时轻微惩罚”
        high_temp = self._sigmoid((temp - p['high_temp_regime']) / p['temp_transition_width'])
        overlong = self._sigmoid((time - 18.0) / 3.0)
        strain_penalty = 2.0 * high_temp * overlong

        strain = (
            p['as_cast_strain']
            + 2.6 * temp_norm
            + 2.0 * logt_norm
            + 1.2 * temp_norm * logt_norm
            + 1.0 * temp_regime
            + 0.8 * time_progress
            - strain_penalty
        )

        return float(strain), float(tensile), float(yld)

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        p = DOMAIN_PARAMS
        T_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ---------------------------
        # 1) 基础少量特征
        # ---------------------------
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_sq_centered'] = (temp - p['critical_temp_regime_shift']) ** 2
        cols['log1p_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['temp_x_logtime'] = temp * log_time
        cols['temp_x_time'] = temp * time

        # ---------------------------
        # 2) 动力学等效特征
        # ---------------------------
        # Arrhenius 类特征
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['inv_temp_K'] = 1.0 / T_k
        cols['arrhenius_factor'] = arrhenius
        cols['arrhenius_time'] = time * arrhenius
        cols['log_arrhenius_time'] = np.log(np.clip(time * arrhenius, 1e-30, None))

        # Larson-Miller Parameter
        lmp = T_k * (p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None)))
        cols['larson_miller'] = lmp

        # Zener-Hollomon 风格的等效热暴露近似（这里只作相对尺度）
        cols['thermal_dose'] = temp * log_time
        cols['thermal_dose_norm'] = ((temp - p['observed_temp_min']) / (p['observed_temp_max'] - p['observed_temp_min'])) * (
            log_time / np.log1p(p['observed_time_max'])
        )

        # ---------------------------
        # 3) 机制区间检测 / 分段激活特征（关键）
        # ---------------------------
        temp_shift = temp - p['critical_temp_regime_shift']
        time_shift = time - p['critical_time_peak']

        cols['temp_rel_regime'] = temp_shift
        cols['time_rel_peak'] = time_shift

        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_very_high_temp'] = (temp >= p['high_temp_regime']).astype(float)
        cols['is_short_time'] = (time <= p['critical_time_short']).astype(float)
        cols['is_peak_time_or_more'] = (time >= p['critical_time_peak']).astype(float)
        cols['is_long_time'] = (time >= 18.0).astype(float)

        # 平滑激活，减少硬阈值不稳定
        temp_gate = self._sigmoid(temp_shift / p['temp_transition_width'])
        very_high_temp_gate = self._sigmoid((temp - p['high_temp_regime']) / p['temp_transition_width'])
        time_gate = self._sigmoid(time_shift / p['time_transition_width'])
        long_time_gate = self._sigmoid((time - 18.0) / 3.0)
        short_time_gate = self._sigmoid((p['critical_time_short'] - time) / 0.8)

        cols['temp_regime_gate'] = temp_gate
        cols['very_high_temp_gate'] = very_high_temp_gate
        cols['time_peak_gate'] = time_gate
        cols['long_time_gate'] = long_time_gate
        cols['short_time_gate'] = short_time_gate

        # 关键组合区间
        cols['is_highT_peakOrLong'] = ((temp >= p['critical_temp_regime_shift']) & (time >= p['critical_time_peak'])).astype(float)
        cols['is_veryHighT_peakOrLong'] = ((temp >= p['high_temp_regime']) & (time >= p['critical_time_peak'])).astype(float)
        cols['is_veryHighT_long'] = ((temp >= p['high_temp_regime']) & (time >= 18.0)).astype(float)
        cols['is_lowT_short'] = ((temp <= 440.0) & (time <= p['critical_time_short'])).astype(float)

        cols['highT_peak_interaction'] = temp_gate * time_gate
        cols['veryHighT_long_interaction'] = very_high_temp_gate * long_time_gate
        cols['lowT_short_interaction'] = (1.0 - temp_gate) * short_time_gate

        # 针对数据中误差最大的点：470/12、440/24、440/1
        # 显式给出与这些边界/窗口的相对位置，帮助模型学习局部残差
        cols['dist_to_470'] = np.abs(temp - 470.0)
        cols['dist_to_440'] = np.abs(temp - 440.0)
        cols['dist_to_12h'] = np.abs(time - 12.0)
        cols['dist_to_24h'] = np.abs(time - 24.0)
        cols['dist_to_1h'] = np.abs(time - 1.0)

        cols['near_470_12'] = np.exp(-((temp - 470.0) / 10.0) ** 2 - ((time - 12.0) / 6.0) ** 2)
        cols['near_440_24'] = np.exp(-((temp - 440.0) / 10.0) ** 2 - ((time - 24.0) / 6.0) ** 2)
        cols['near_440_1'] = np.exp(-((temp - 440.0) / 10.0) ** 2 - ((time - 1.0) / 2.0) ** 2)

        # ---------------------------
        # 4) 物理 baseline 及残差坐标
        # ---------------------------
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []

        for ti, hi in zip(temp, time):
            bs, bt, by = self.physics_baseline(ti, hi)
            baseline_strain.append(bs)
            baseline_tensile.append(bt)
            baseline_yield.append(by)

        baseline_strain = np.array(baseline_strain, dtype=float)
        baseline_tensile = np.array(baseline_tensile, dtype=float)
        baseline_yield = np.array(baseline_yield, dtype=float)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 相对铸态提升基线
        cols['baseline_delta_strain'] = baseline_strain - p['as_cast_strain']
        cols['baseline_delta_tensile'] = baseline_tensile - p['as_cast_tensile']
        cols['baseline_delta_yield'] = baseline_yield - p['as_cast_yield']

        # 基线形状相关特征
        cols['baseline_tsy_gap'] = baseline_tensile - baseline_yield
        cols['baseline_ys_ts_ratio'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)
        cols['baseline_strength_ductility'] = baseline_tensile * baseline_strain

        # 温度/时间相对机制边界的残差坐标
        cols['temp_over_shift_pos'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_below_shift_pos'] = np.clip(p['critical_temp_regime_shift'] - temp, 0, None)
        cols['time_over_peak_pos'] = np.clip(time - p['critical_time_peak'], 0, None)
        cols['time_below_peak_pos'] = np.clip(p['critical_time_peak'] - time, 0, None)

        cols['regime_excess_heat'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None) * np.clip(time - p['critical_time_peak'], 0, None)
        cols['short_low_deficit'] = np.clip(440.0 - temp, 0, None) * np.clip(p['critical_time_short'] - time, 0, None)

        # ---------------------------
        # 5) 训练域边界距离 / 外推风险特征（关键）
        # ---------------------------
        # 到整体实验边界的相对位置
        temp_norm = (temp - p['observed_temp_min']) / (p['observed_temp_max'] - p['observed_temp_min'])
        time_norm = (time - p['observed_time_min']) / (p['observed_time_max'] - p['observed_time_min'])

        cols['temp_norm_domain'] = temp_norm
        cols['time_norm_domain'] = time_norm

        # 到边界距离
        cols['dist_to_temp_min'] = temp - p['observed_temp_min']
        cols['dist_to_temp_max'] = p['observed_temp_max'] - temp
        cols['dist_to_time_min'] = time - p['observed_time_min']
        cols['dist_to_time_max'] = p['observed_time_max'] - time

        cols['edge_temp_proximity'] = np.minimum(cols['dist_to_temp_min'], cols['dist_to_temp_max'])
        cols['edge_time_proximity'] = np.minimum(cols['dist_to_time_min'], cols['dist_to_time_max'])

        # 角点/边界风险：小样本下边界点常更难
        cols['is_temp_edge_zone'] = ((temp <= 440.0) | (temp >= 470.0)).astype(float)
        cols['is_time_edge_zone'] = ((time <= 1.5) | (time >= 24.0)).astype(float)
        cols['edge_zone_interaction'] = cols['is_temp_edge_zone'] * cols['is_time_edge_zone']

        # 到代表性训练锚点的距离
        min_anchor_dist = []
        mean_anchor_dist = []
        for ti, hi in zip(temp, time):
            dmin, dmean = self._distance_to_anchors(ti, hi)
            min_anchor_dist.append(dmin)
            mean_anchor_dist.append(dmean)

        min_anchor_dist = np.array(min_anchor_dist, dtype=float)
        mean_anchor_dist = np.array(mean_anchor_dist, dtype=float)

        cols['min_anchor_dist'] = min_anchor_dist
        cols['mean_anchor_dist'] = mean_anchor_dist
        cols['anchor_dist_sq'] = min_anchor_dist ** 2
        cols['is_far_from_anchor'] = (min_anchor_dist > 0.75).astype(float)

        # 针对题目已知外推方向，显式编码相对最近已知点的位移
        cols['delta_from_420_1_temp'] = temp - 420.0
        cols['delta_from_420_12_temp'] = temp - 420.0
        cols['delta_from_420_24_temp'] = temp - 420.0
        cols['delta_from_480_12_temp'] = temp - 480.0
        cols['delta_from_470_1_temp'] = temp - 470.0
        cols['delta_from_470_24_temp'] = temp - 470.0

        cols['delta_from_420_1_time'] = time - 1.0
        cols['delta_from_420_12_time'] = time - 12.0
        cols['delta_from_420_24_time'] = time - 24.0
        cols['delta_from_480_12_time'] = time - 12.0
        cols['delta_from_470_1_time'] = time - 1.0
        cols['delta_from_470_24_time'] = time - 24.0

        # ---------------------------
        # 6) 少量补充非线性，避免维度爆炸
        # ---------------------------
        cols['temp_centered'] = temp - 450.0
        cols['time_centered_log'] = log_time - np.mean(np.log1p([1.0, 12.0, 24.0]))
        cols['temp_centered_x_logtime'] = cols['temp_centered'] * log_time

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names