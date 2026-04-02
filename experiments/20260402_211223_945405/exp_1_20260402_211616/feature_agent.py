import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # 7499 铝合金热处理/时效过程中的经验性机制分界
    # 这里不追求严格材料常数，而是给出“小样本可泛化”的物理先验阈值
    'critical_temp_regime_shift': 455.0,   # ℃，低温/高温时效机制分界近似点
    'high_temp_overdrive': 468.0,          # ℃，更强烈组织演化/粗化风险区
    'critical_time_short': 3.0,            # h，短时时效区
    'critical_time_long': 18.0,            # h，长时时效区/组织充分演化区

    # 激活能：铝合金析出/扩散控制过程量级，取经验近似
    'activation_energy_Q': 115000.0,       # J/mol
    'gas_constant_R': 8.314,               # J/(mol·K)

    # Larson-Miller / 时间温度等效参数
    'lmp_C': 20.0,

    # 原始样品（凝固态）平均强度，作为物理基线锚点
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # 在当前给定窗口内，根据误差分布与常见热处理规律，
    # 假设温度升高/时间延长总体上促进强化（至少在本数据窗口内如此）
    # 同时在高温长时端加入轻微“饱和/转折”而不是强行假设显著软化
    't_ref_celsius': 440.0,
    'time_ref': 1.0,
}


class FeatureAgent:
    """
    小样本（29条）下的物理启发式特征工程：
    1. 少而精，避免无约束高阶堆砌
    2. 引入热激活动力学、机制分区、边界距离
    3. 用 physics_baseline 作为先验，让模型学习残差
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于世界知识的近似基线：
        - 在当前数据窗口内，默认更高温度/更长时间 => 更高强度
        - 延性（strain）与强化通常存在竞争，给出温和下降趋势
        - 在高温长时端只做“饱和/轻微转折”，避免无依据地强行设定明显过时效软化
        """
        p = DOMAIN_PARAMS
        temp = float(temp)
        time = float(time)

        # Kelvin for Arrhenius-type kinetics
        T_k = temp + 273.15
        t_eff = max(time, 1e-6)

        # 相对参考条件的动力学推进程度
        T_ref_k = p['t_ref_celsius'] + 273.15
        arr_ratio = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k)) / \
                    np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_ref_k))

        # 用 log(time) + Arrhenius 组合表征时效推进
        progress = arr_ratio * (1.0 + np.log1p(t_eff))
        progress_ref = 1.0 * (1.0 + np.log1p(p['time_ref']))
        x = progress / progress_ref

        # 强化主趋势：随 x 增长而上升，并逐渐饱和
        harden = 1.0 - np.exp(-0.55 * max(x - 0.2, 0.0))

        # 高温长时端增加很弱的“过驱动/粗化”惩罚，避免方向过激
        over_temp = max(temp - p['high_temp_overdrive'], 0.0)
        over_time = max(np.log1p(time) - np.log1p(p['critical_time_long']), 0.0)
        overdrive = over_temp / 12.0 + over_time / 1.2
        penalty = 1.0 - np.exp(-0.35 * max(overdrive, 0.0))

        # 凝固态锚点 + 时效强化增量
        baseline_tensile = p['as_cast_tensile'] + 95.0 * harden - 18.0 * penalty
        baseline_yield = p['as_cast_yield'] + 78.0 * harden - 15.0 * penalty

        # 延性与强化竞争：先温和下降；若高温长时有一定粗化，则略回升一点
        baseline_strain = p['as_cast_strain'] - 1.4 * harden + 0.8 * penalty

        return baseline_strain, baseline_tensile, baseline_yield

    def engineer_features(self, df):
        p = DOMAIN_PARAMS
        cols = {}

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values

        T_k = temp + 273.15
        log_time = np.log1p(np.clip(time, 0, None))
        sqrt_time = np.sqrt(np.clip(time, 0, None))

        # ========= 1) 基础物理输入 =========
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_K'] = T_k
        cols['log_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['inv_temp_K'] = 1.0 / np.clip(T_k, 1e-6, None)

        # ========= 2) 动力学等效特征 =========
        # Arrhenius 因子（可与时间合并表示组织演化程度）
        arrhenius = np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_k))
        cols['arrhenius'] = arrhenius
        cols['arrhenius_x_time'] = arrhenius * np.clip(time, 0, None)
        cols['arrhenius_x_log_time'] = arrhenius * log_time

        # Larson-Miller Parameter（虽然常用于蠕变，但可作为时间-温度合并尺度）
        cols['larson_miller'] = T_k * (p['lmp_C'] + np.log10(np.clip(time, 1e-6, None)))

        # Hollomon/Jaffe 型简化时温参数
        cols['temp_x_log_time'] = temp * log_time
        cols['temp_x_sqrt_time'] = temp * sqrt_time

        # ========= 3) 机制区间检测特征（关键） =========
        Tc = p['critical_temp_regime_shift']
        Th = p['high_temp_overdrive']
        ts = p['critical_time_short']
        tl = p['critical_time_long']

        is_high_temp = (temp >= Tc).astype(float)
        is_very_high_temp = (temp >= Th).astype(float)
        is_short_time = (time <= ts).astype(float)
        is_long_time = (time >= tl).astype(float)
        is_high_temp_long_time = ((temp >= Tc) & (time >= tl)).astype(float)
        is_very_high_temp_long_time = ((temp >= Th) & (time >= 12.0)).astype(float)

        cols['is_high_temp'] = is_high_temp
        cols['is_very_high_temp'] = is_very_high_temp
        cols['is_short_time'] = is_short_time
        cols['is_long_time'] = is_long_time
        cols['is_high_temp_long_time'] = is_high_temp_long_time
        cols['is_very_high_temp_long_time'] = is_very_high_temp_long_time

        # 分段激活：帮助模型学习边界附近的斜率变化
        temp_above_tc = np.clip(temp - Tc, 0, None)
        temp_above_th = np.clip(temp - Th, 0, None)
        time_above_tl = np.clip(time - tl, 0, None)
        time_below_ts = np.clip(ts - time, 0, None)

        cols['temp_above_tc'] = temp_above_tc
        cols['temp_above_th'] = temp_above_th
        cols['time_above_tl'] = time_above_tl
        cols['time_below_ts'] = time_below_ts
        cols['temp_above_tc_x_log_time'] = temp_above_tc * log_time
        cols['temp_above_th_x_log_time'] = temp_above_th * log_time
        cols['temp_above_tc_x_time_above_tl'] = temp_above_tc * time_above_tl

        # ========= 4) 边界距离 / 外推风险特征（关键） =========
        # 根据已知训练覆盖：
        # temp 训练近似覆盖 420, 470, 480；时间覆盖 1,12,24（从提示推断）
        train_temps = np.array([420.0, 460.0, 470.0, 480.0])
        train_times = np.array([1.0, 12.0, 24.0])

        min_temp_dist = np.min(np.abs(temp[:, None] - train_temps[None, :]), axis=1)
        min_time_dist = np.min(np.abs(time[:, None] - train_times[None, :]), axis=1)

        cols['min_temp_dist_to_train_grid'] = min_temp_dist
        cols['min_time_dist_to_train_grid'] = min_time_dist
        cols['joint_dist_to_train_grid'] = np.sqrt((min_temp_dist / 10.0) ** 2 + (min_time_dist / 6.0) ** 2)

        # 对训练“主体区间”边界的距离：区间外为正，区间内为 0
        temp_low_extrap = np.clip(420.0 - temp, 0, None)
        temp_high_extrap = np.clip(temp - 480.0, 0, None)
        time_low_extrap = np.clip(1.0 - time, 0, None)
        time_high_extrap = np.clip(time - 24.0, 0, None)

        cols['temp_low_extrap'] = temp_low_extrap
        cols['temp_high_extrap'] = temp_high_extrap
        cols['time_low_extrap'] = time_low_extrap
        cols['time_high_extrap'] = time_high_extrap

        # 对“机制边界”的相对位置
        cols['temp_relative_to_tc'] = temp - Tc
        cols['temp_relative_to_th'] = temp - Th
        cols['log_time_relative_to_long'] = log_time - np.log1p(tl)
        cols['log_time_relative_to_short'] = log_time - np.log1p(ts)

        # ========= 5) 物理基线特征（让模型学残差） =========
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []
        for t, ti in zip(temp, time):
            bs, bt, by = self.physics_baseline(t, ti)
            baseline_strain.append(bs)
            baseline_tensile.append(bt)
            baseline_yield.append(by)

        baseline_strain = np.array(baseline_strain)
        baseline_tensile = np.array(baseline_tensile)
        baseline_yield = np.array(baseline_yield)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 基线组合特征
        cols['baseline_strength_sum'] = baseline_tensile + baseline_yield
        cols['baseline_strength_diff'] = baseline_tensile - baseline_yield
        cols['baseline_ductility_strength_ratio'] = baseline_strain / np.clip(baseline_tensile, 1e-6, None)

        # 与机制边界的相对基线耦合
        cols['baseline_tensile_x_high_temp'] = baseline_tensile * is_high_temp
        cols['baseline_yield_x_high_temp'] = baseline_yield * is_high_temp
        cols['baseline_strain_x_long_time'] = baseline_strain * is_long_time

        # ========= 6) 少量非线性项（控制维度） =========
        cols['temp_sq_scaled'] = (temp / 100.0) ** 2
        cols['log_time_sq'] = log_time ** 2
        cols['temp_x_time_scaled'] = (temp / 100.0) * log_time

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names