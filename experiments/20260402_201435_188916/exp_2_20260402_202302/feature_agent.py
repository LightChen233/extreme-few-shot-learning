import pandas as pd
import numpy as np


# 7499铝合金热处理的粗略物理先验参数
# 说明：
# 1) 这里不是精确材料常数，而是用于小样本建模的“机制感知”阈值
# 2) 阈值选择必须与题目给出的数据趋势一致：
#    - 420→480℃整体强度提升明显
#    - 高温区存在时间效应分化，尤其应变在高温中时附近可能见峰
#    - 外推点 470,12 / 440,24 / 440,1 误差大，说明需要显式刻画边界/机制切换
DOMAIN_PARAMS = {
    # 机制边界：约在中高温区进入更强烈的溶质扩散/析出演化加速阶段
    'critical_temp_regime_shift': 450.0,      # ℃，低温/高温机制分界
    'high_temp_threshold': 470.0,             # ℃，高温强化/组织快速演化区
    'critical_time_regime_shift': 12.0,       # h，短时/中长时边界
    'long_time_threshold': 24.0,              # h，长时暴露边界

    # 高温区应变可能在中等保温时间附近达到较优，之后略回落
    'ductility_peak_temp': 475.0,             # ℃
    'ductility_peak_time': 12.0,              # h

    # 动力学参数：铝合金中扩散/析出控制过程常见量级，取工程近似值
    'activation_energy_Q': 120000.0,          # J/mol
    'gas_constant_R': 8.314,                  # J/mol/K
    'larson_miller_C': 20.0,                  # LMP常数

    # 训练域边界（由题目信息可知）
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # 训练点网格（用于“距训练域/训练点距离”）
    # 从题目覆盖信息可知：训练中存在 420/460/480 × 1/12/24 这些关键点
    'known_train_temps': [420.0, 460.0, 480.0],
    'known_train_times': [1.0, 12.0, 24.0],

    # 原始铸态基准
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,
}


class FeatureAgent:
    """
    面向小样本热处理-性能预测的特征工程：
    - 少而精，避免纯堆砌高维多项式
    - 引入机制区间、动力学等效量、训练域边界距离
    - 加入 physics baseline 作为残差学习支点
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于材料世界知识、且与题目给出的数据统计一致的粗略基线：
        - 在当前温度范围内：升温总体提升强度
        - 时间延长总体提升强度，但在高温区对应变可能出现中时较优
        - 不强加“高温长时一定软化”的错误先验
        """
        p = DOMAIN_PARAMS

        temp = float(temp)
        time = float(time)

        T_k = temp + 273.15
        logt = np.log1p(max(time, 0.0))

        # 1) 热激活进程：高温显著加速
        arr = time * np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        # 尺度压缩，避免数值极小
        arr_s = np.log1p(arr * 1e12)

        # 2) 温度主效应：在420~480范围内总体向上
        temp_norm = (temp - 420.0) / 60.0  # roughly [0,1] in known range
        temp_norm = np.clip(temp_norm, -0.5, 1.5)

        # 3) 时间主效应：早期增长快，后期趋缓
        time_norm = np.log1p(time) / np.log1p(24.0)
        time_norm = np.clip(time_norm, 0.0, 1.5)

        # 4) 高温-中时对应变优化窗口
        duct_peak = np.exp(
            -((temp - p['ductility_peak_temp']) / 18.0) ** 2
            -((np.log1p(time) - np.log1p(p['ductility_peak_time'])) / 0.55) ** 2
        )

        # 5) 长时高温下，应变可能略回落，但幅度不能压倒整体强化趋势
        hi_temp = max(0.0, temp - p['high_temp_threshold']) / 20.0
        long_time = max(0.0, np.log1p(time) - np.log1p(p['critical_time_regime_shift']))
        overexposure = hi_temp * long_time

        # ---- baseline: 与数据趋势一致的近似 ----
        # 强度：总体随温度上升而升高；时间延长总体增益，带轻微平台化
        baseline_tensile = (
            p['as_cast_tensile']
            + 85.0 * temp_norm
            + 38.0 * time_norm
            + 10.0 * arr_s
            + 8.0 * temp_norm * time_norm
        )

        baseline_yield = (
            p['as_cast_yield']
            + 70.0 * temp_norm
            + 34.0 * time_norm
            + 8.0 * arr_s
            + 7.0 * temp_norm * time_norm
        )

        # 应变：整体较铸态提升；高温中时有更优窗口；高温长时略有回落
        baseline_strain = (
            p['as_cast_strain']
            + 2.0 * temp_norm
            + 3.2 * time_norm
            + 1.8 * duct_peak
            - 0.9 * overexposure
            + 0.5 * arr_s
        )

        return baseline_strain, baseline_tensile, baseline_yield

    def _distance_to_range(self, x, low, high):
        if x < low:
            return low - x
        if x > high:
            return x - high
        return 0.0

    def _nearest_distance_to_grid(self, temp, time):
        temps = DOMAIN_PARAMS['known_train_temps']
        times = DOMAIN_PARAMS['known_train_times']

        best = np.inf
        for tt in temps:
            for hh in times:
                # 归一化欧氏距离，避免温度和时间量纲不一致
                d = np.sqrt(((temp - tt) / 20.0) ** 2 + ((time - hh) / 11.0) ** 2)
                if d < best:
                    best = d
        return best

    def engineer_features(self, df):
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        p = DOMAIN_PARAMS
        T_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ---------------------------
        # 1) 基础低维特征
        # ---------------------------
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_centered'] = temp - p['critical_temp_regime_shift']
        cols['time_centered'] = time - p['critical_time_regime_shift']
        cols['temp_norm_from_420'] = (temp - 420.0) / 60.0
        cols['log1p_time'] = log_time
        cols['sqrt_time'] = sqrt_time

        # 有限的非线性项，避免过多高维
        cols['temp_sq_scaled'] = ((temp - 450.0) / 30.0) ** 2
        cols['log_time_sq'] = log_time ** 2
        cols['temp_x_logtime'] = temp * log_time
        cols['temp_x_time_scaled'] = (temp / 100.0) * np.sqrt(time)

        # ---------------------------
        # 2) 动力学等效特征
        # ---------------------------
        arr = time * np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['arrhenius_progress'] = arr
        cols['log_arrhenius_progress'] = np.log1p(arr * 1e12)

        # Larson-Miller Parameter（只作等效热暴露表征）
        lmp = T_k * (p['larson_miller_C'] + np.log10(np.clip(time, 1e-6, None)))
        cols['larson_miller'] = lmp / 1000.0

        # Zener-Hollomon风格的逆温度-时间组合（非严格定义，作表征用）
        cols['logtime_over_T'] = log_time / T_k
        cols['inv_T'] = 1.0 / T_k

        # ---------------------------
        # 3) 机制区间检测特征（关键）
        # ---------------------------
        is_high_temp = (temp >= p['critical_temp_regime_shift']).astype(float)
        is_very_high_temp = (temp >= p['high_temp_threshold']).astype(float)
        is_mid_long_time = (time >= p['critical_time_regime_shift']).astype(float)
        is_long_time = (time >= p['long_time_threshold']).astype(float)

        cols['is_high_temp_regime'] = is_high_temp
        cols['is_very_high_temp'] = is_very_high_temp
        cols['is_mid_long_time'] = is_mid_long_time
        cols['is_long_time'] = is_long_time

        # 温度/时间分段激活：帮助模型学习“过阈值后斜率变化”
        cols['temp_above_regime'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_above_high'] = np.clip(temp - p['high_temp_threshold'], 0, None)
        cols['time_above_regime'] = np.clip(time - p['critical_time_regime_shift'], 0, None)
        cols['time_above_long'] = np.clip(time - p['long_time_threshold'], 0, None)

        # 高温+长时联合作用：用于捕捉机制切换/粗化风险，但不强加负方向
        cols['highT_longt_gate'] = is_high_temp * is_mid_long_time
        cols['veryhighT_longt_gate'] = is_very_high_temp * is_mid_long_time
        cols['highT_x_logtime'] = is_high_temp * log_time
        cols['temp_above_x_time_above'] = cols['temp_above_regime'] * cols['time_above_regime']

        # 应变峰区探测：高温中时可能塑性最优
        ductility_peak = np.exp(
            -((temp - p['ductility_peak_temp']) / 18.0) ** 2
            -((log_time - np.log1p(p['ductility_peak_time'])) / 0.55) ** 2
        )
        cols['ductility_peak_window'] = ductility_peak

        # ---------------------------
        # 4) 训练域边界 / 外推距离特征（关键）
        # ---------------------------
        dist_temp_out = np.array([
            self._distance_to_range(v, p['train_temp_min'], p['train_temp_max']) for v in temp
        ], dtype=float)
        dist_time_out = np.array([
            self._distance_to_range(v, p['train_time_min'], p['train_time_max']) for v in time
        ], dtype=float)

        cols['dist_outside_temp_range'] = dist_temp_out
        cols['dist_outside_time_range'] = dist_time_out
        cols['is_outside_train_box'] = ((dist_temp_out > 0) | (dist_time_out > 0)).astype(float)

        # 即使在训练盒内，也可能是“未见工艺点”，用最近训练网格距离刻画
        nearest_grid_dist = np.array([
            self._nearest_distance_to_grid(tt, hh) for tt, hh in zip(temp, time)
        ], dtype=float)
        cols['nearest_train_grid_dist'] = nearest_grid_dist

        # 离机制边界距离：外推点常发生在边界附近或跨边界
        cols['dist_to_temp_regime_boundary'] = np.abs(temp - p['critical_temp_regime_shift'])
        cols['dist_to_high_temp_boundary'] = np.abs(temp - p['high_temp_threshold'])
        cols['dist_to_time_regime_boundary'] = np.abs(time - p['critical_time_regime_shift'])

        # 与验证/测试常见外推方向一致的局部相对量
        cols['temp_relative_to_440'] = temp - 440.0
        cols['temp_relative_to_470'] = temp - 470.0
        cols['time_relative_to_12'] = time - 12.0
        cols['time_relative_to_24'] = time - 24.0

        # ---------------------------
        # 5) 物理基线及残差支点特征
        # ---------------------------
        b_strain = []
        b_ts = []
        b_ys = []
        for tt, hh in zip(temp, time):
            s0, ts0, ys0 = self.physics_baseline(tt, hh)
            b_strain.append(s0)
            b_ts.append(ts0)
            b_ys.append(ys0)

        b_strain = np.array(b_strain, dtype=float)
        b_ts = np.array(b_ts, dtype=float)
        b_ys = np.array(b_ys, dtype=float)

        cols['baseline_strain'] = b_strain
        cols['baseline_tensile'] = b_ts
        cols['baseline_yield'] = b_ys

        # 相对铸态提升量基线
        cols['baseline_delta_strain'] = b_strain - p['as_cast_strain']
        cols['baseline_delta_tensile'] = b_ts - p['as_cast_tensile']
        cols['baseline_delta_yield'] = b_ys - p['as_cast_yield']

        # 基线与机制/外推交互：让模型学“哪些区域 baseline 更不可信”
        cols['baseline_ts_x_grid_dist'] = b_ts * nearest_grid_dist
        cols['baseline_ys_x_grid_dist'] = b_ys * nearest_grid_dist
        cols['baseline_strain_x_peak'] = b_strain * ductility_peak
        cols['baseline_ts_x_highTlongt'] = b_ts * cols['highT_longt_gate']
        cols['baseline_ys_x_highTlongt'] = b_ys * cols['highT_longt_gate']

        # ---------------------------
        # 6) 少量与性能机理相关的组合特征
        # ---------------------------
        # 强化-塑性协同窗口的代理量
        cols['process_intensity'] = cols['temp_norm_from_420'] * log_time
        cols['high_temp_process_intensity'] = is_high_temp * cols['process_intensity']

        # 对440/470等关键外推边界更敏感的分段项
        cols['temp_above_440'] = np.clip(temp - 440.0, 0, None)
        cols['temp_below_440'] = np.clip(440.0 - temp, 0, None)
        cols['temp_below_470'] = np.clip(470.0 - temp, 0, None)
        cols['temp_above_470'] = np.clip(temp - 470.0, 0, None)

        # 24h外推边界及1h短时边界
        cols['near_1h_short_process'] = 1.0 / np.clip(time, 1.0, None)
        cols['time_to_24_ratio'] = time / 24.0

        # ---------------------------
        # 输出
        # ---------------------------
        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names