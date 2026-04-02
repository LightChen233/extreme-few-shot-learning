import pandas as pd
import numpy as np


DOMAIN_PARAMS = {
    # ---- 基本物理常数 ----
    'gas_constant_R': 8.314,                  # J/mol/K
    'kelvin_offset': 273.15,

    # ---- 机制边界 / 区间阈值 ----
    # 结合题目给出的数据趋势：420℃整体较低，460–480℃整体显著更强，
    # 440℃与460℃之间很可能是“强化机制明显加速”的区间。
    'critical_temp_regime_shift': 450.0,      # 低温/高温机制切换的经验边界
    'high_temp_threshold': 465.0,             # 更高温区，接近470–480℃
    'critical_time_regime_shift': 10.0,       # 1h vs 12/24h 的短时/中长时边界
    'long_time_threshold': 18.0,              # 区分12h和24h，表征长时暴露

    # ---- 动力学参数 ----
    # 铝合金析出/扩散控制过程常见量级，取中等估计用于构造热激活特征
    'activation_energy_Q': 125000.0,          # J/mol
    'activation_energy_Q_low': 100000.0,      # J/mol
    'activation_energy_Q_high': 150000.0,     # J/mol

    # ---- 外推边界（由当前已知工艺范围推断）----
    'train_temp_min': 420.0,
    'train_temp_max': 480.0,
    'train_time_min': 1.0,
    'train_time_max': 24.0,

    # ---- 基线标尺 ----
    'as_cast_strain': 6.94,
    'as_cast_tensile': 145.83,
    'as_cast_yield': 96.60,

    # ---- 经验最优窗口/转折中心 ----
    # 用于柔性刻画“470℃,12h”附近可能存在特殊组织状态，
    # 该点在验证集中误差最大，说明需要显式建模这一局部机制区。
    'transition_temp_center': 470.0,
    'transition_time_center': 12.0,
    'transition_temp_width': 12.0,
    'transition_time_width': 6.0,
}


class FeatureAgent:
    """
    面向 7499 铝合金热处理小样本回归的物理特征工程版本。

    设计原则：
    1. 小样本（29条）下，控制特征数量，优先保留少而精的物理特征；
    2. 显式加入机制边界/区间特征，帮助模型处理外推点；
    3. 用 physics_baseline 作为“先验”，让模型学习残差；
    4. 引入热激活/等效热暴露特征，表达温度-时间耦合。
    """

    def __init__(self):
        self.feature_names = []

    def physics_baseline(self, temp, time):
        """
        基于领域知识的近似基线预测，不依赖训练标签。

        与题目给出的统计趋势保持一致：
        - 在当前温度范围（420–480℃）内，温度升高总体使强度上升；
        - 时间延长总体有利于强化，但应变在高温中等时间附近可能更优，
          长时可能出现一定回落；
        - 不强行假设整体软化，只对高温长时应变做轻微非单调修正。
        """
        p = DOMAIN_PARAMS
        T = float(temp)
        t = max(float(time), 1e-6)

        # 归一化工艺尺度
        temp_norm = (T - p['train_temp_min']) / (p['train_temp_max'] - p['train_temp_min'])  # roughly 0~1
        time_log = np.log1p(t) / np.log1p(p['train_time_max'])                                 # roughly 0~1

        # 热激活进程指标：高温和长时都提高 process_index
        Tk = T + p['kelvin_offset']
        arr = t * np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * Tk))

        # 数值稳定的归一化：映射到较平滑的 0~1 区间
        process_index = (
            0.55 * np.clip(temp_norm, 0.0, 1.2) +
            0.45 * np.clip(time_log, 0.0, 1.2)
        )

        # ---- 强度基线：总体随温度/时间升高而升高 ----
        # 抗拉强度增量
        tensile_gain = (
            70.0 * temp_norm +
            55.0 * time_log +
            35.0 * temp_norm * time_log
        )

        # 中高温区强化加速
        if T >= p['critical_temp_regime_shift']:
            tensile_gain += 12.0 * (T - p['critical_temp_regime_shift']) / 30.0

        if t >= p['critical_time_regime_shift']:
            tensile_gain += 8.0 * (np.log1p(t) - np.log1p(p['critical_time_regime_shift']))

        baseline_tensile = p['as_cast_tensile'] + tensile_gain

        # 屈服强度与抗拉强度同向，但灵敏度略高
        yield_gain = (
            75.0 * temp_norm +
            48.0 * time_log +
            40.0 * temp_norm * time_log
        )
        if T >= p['critical_temp_regime_shift']:
            yield_gain += 14.0 * (T - p['critical_temp_regime_shift']) / 30.0
        if t >= p['critical_time_regime_shift']:
            yield_gain += 7.0 * (np.log1p(t) - np.log1p(p['critical_time_regime_shift']))

        baseline_yield = p['as_cast_yield'] + yield_gain

        # ---- 应变基线：总体改善，但在高温长时不强行上升 ----
        strain_gain = (
            1.8 * temp_norm +
            2.4 * time_log +
            1.2 * temp_norm * time_log
        )

        # 470℃, 12h 附近可能存在塑性改善窗口
        peak_temp = np.exp(-((T - 470.0) / 18.0) ** 2)
        peak_time = np.exp(-((np.log1p(t) - np.log1p(12.0)) / 0.7) ** 2)
        strain_gain += 2.2 * peak_temp * peak_time

        # 高温长时对应变做轻微回落修正（只做温和非单调，不对强度施加整体软化）
        if (T >= p['high_temp_threshold']) and (t >= p['long_time_threshold']):
            strain_gain -= 0.8

        baseline_strain = p['as_cast_strain'] + strain_gain

        return baseline_strain, baseline_tensile, baseline_yield

    def _nearest_boundary_distance(self, x, low, high):
        """
        到已知训练边界区间的有符号距离：
        区间内为 0，区间外为正距离。
        """
        if x < low:
            return low - x
        if x > high:
            return x - high
        return 0.0

    def engineer_features(self, df):
        p = DOMAIN_PARAMS

        temp = df['temp'].astype(float).values
        time = df['time'].astype(float).values
        time_safe = np.clip(time, 1e-8, None)

        T_kelvin = temp + p['kelvin_offset']
        log_time = np.log1p(time_safe)
        sqrt_time = np.sqrt(time_safe)

        cols = {}

        # ------------------------------------------------------------------
        # 1) 原始基础特征
        # ------------------------------------------------------------------
        cols['temp'] = temp
        cols['time'] = time
        cols['temp_sq'] = temp ** 2
        cols['log1p_time'] = log_time
        cols['sqrt_time'] = sqrt_time
        cols['inv_temp_K'] = 1.0 / T_kelvin
        cols['inv_sqrt_time'] = 1.0 / np.sqrt(np.clip(time_safe, 1e-6, None))

        # 控制数量，只保留最关键交互
        cols['temp_x_logtime'] = temp * log_time
        cols['temp_x_time'] = temp * time
        cols['temp_over_time'] = temp / np.clip(time_safe, 1e-6, None)

        # ------------------------------------------------------------------
        # 2) 动力学等效特征
        # ------------------------------------------------------------------
        # Larson-Miller 风格参数（小时制）
        C = 20.0
        cols['larson_miller'] = T_kelvin * (C + np.log10(np.clip(time_safe, 1e-6, None)))

        # Arrhenius 型热暴露
        arr_mid = time_safe * np.exp(-p['activation_energy_Q'] / (p['gas_constant_R'] * T_kelvin))
        arr_low = time_safe * np.exp(-p['activation_energy_Q_low'] / (p['gas_constant_R'] * T_kelvin))
        arr_high = time_safe * np.exp(-p['activation_energy_Q_high'] / (p['gas_constant_R'] * T_kelvin))

        cols['arrhenius_exposure_mid'] = arr_mid
        cols['log_arrhenius_exposure_mid'] = np.log1p(arr_mid)
        cols['arrhenius_exposure_lowQ'] = arr_low
        cols['arrhenius_exposure_highQ'] = arr_high

        # 经验热处理强度
        cols['thermal_dose_log'] = temp * log_time
        cols['thermal_dose_sqrt'] = temp * sqrt_time

        # ------------------------------------------------------------------
        # 3) 机制区间检测特征（关键）
        # ------------------------------------------------------------------
        cols['is_high_temp_regime'] = (temp >= p['critical_temp_regime_shift']).astype(float)
        cols['is_very_high_temp'] = (temp >= p['high_temp_threshold']).astype(float)
        cols['is_long_time_regime'] = (time >= p['critical_time_regime_shift']).astype(float)
        cols['is_very_long_time'] = (time >= p['long_time_threshold']).astype(float)

        cols['highT_and_longt'] = (
            (temp >= p['critical_temp_regime_shift']) &
            (time >= p['critical_time_regime_shift'])
        ).astype(float)

        cols['very_highT_and_very_longt'] = (
            (temp >= p['high_temp_threshold']) &
            (time >= p['long_time_threshold'])
        ).astype(float)

        # 分段激活：距边界的“超过程度”
        cols['temp_above_regime'] = np.clip(temp - p['critical_temp_regime_shift'], 0, None)
        cols['temp_above_high'] = np.clip(temp - p['high_temp_threshold'], 0, None)
        cols['time_above_regime'] = np.clip(time - p['critical_time_regime_shift'], 0, None)
        cols['time_above_long'] = np.clip(time - p['long_time_threshold'], 0, None)

        cols['temp_below_regime'] = np.clip(p['critical_temp_regime_shift'] - temp, 0, None)
        cols['time_below_regime'] = np.clip(p['critical_time_regime_shift'] - time, 0, None)

        # 高温长时协同激活：帮助捕捉局部非线性
        cols['over_regime_interaction'] = (
            np.clip(temp - p['critical_temp_regime_shift'], 0, None) *
            np.clip(np.log1p(time) - np.log1p(p['critical_time_regime_shift']), 0, None)
        )

        # 围绕 470℃,12h 的局部窗口激活
        local_window = np.exp(
            -((temp - p['transition_temp_center']) / p['transition_temp_width']) ** 2
            -((time - p['transition_time_center']) / p['transition_time_width']) ** 2
        )
        cols['local_transition_window'] = local_window

        # ------------------------------------------------------------------
        # 4) 外推/边界距离特征（关键）
        # ------------------------------------------------------------------
        temp_dist_out = np.array([
            self._nearest_boundary_distance(x, p['train_temp_min'], p['train_temp_max']) for x in temp
        ], dtype=float)
        time_dist_out = np.array([
            self._nearest_boundary_distance(x, p['train_time_min'], p['train_time_max']) for x in time
        ], dtype=float)

        cols['temp_outside_train_range'] = temp_dist_out
        cols['time_outside_train_range'] = time_dist_out
        cols['joint_outside_distance'] = np.sqrt(temp_dist_out ** 2 + time_dist_out ** 2)

        # 相对关键机制边界的距离
        cols['dist_to_temp_regime_boundary'] = temp - p['critical_temp_regime_shift']
        cols['dist_to_time_regime_boundary'] = time - p['critical_time_regime_shift']
        cols['abs_dist_to_temp_regime_boundary'] = np.abs(temp - p['critical_temp_regime_shift'])
        cols['abs_dist_to_time_regime_boundary'] = np.abs(time - p['critical_time_regime_shift'])

        # 到重点误差区域中心（470,12）距离
        cols['dist_to_transition_center'] = np.sqrt(
            ((temp - p['transition_temp_center']) / p['transition_temp_width']) ** 2 +
            ((time - p['transition_time_center']) / p['transition_time_width']) ** 2
        )

        # 到几个关键工艺时长的距离
        cols['dist_to_1h'] = np.abs(time - 1.0)
        cols['dist_to_12h'] = np.abs(time - 12.0)
        cols['dist_to_24h'] = np.abs(time - 24.0)

        # ------------------------------------------------------------------
        # 5) 物理基线预测特征（让模型学残差）
        # ------------------------------------------------------------------
        baseline_strain = []
        baseline_tensile = []
        baseline_yield = []

        for T, t in zip(temp, time):
            s0, ts0, ys0 = self.physics_baseline(T, t)
            baseline_strain.append(s0)
            baseline_tensile.append(ts0)
            baseline_yield.append(ys0)

        baseline_strain = np.array(baseline_strain, dtype=float)
        baseline_tensile = np.array(baseline_tensile, dtype=float)
        baseline_yield = np.array(baseline_yield, dtype=float)

        cols['baseline_strain'] = baseline_strain
        cols['baseline_tensile'] = baseline_tensile
        cols['baseline_yield'] = baseline_yield

        # 相对铸态提升量的先验
        cols['baseline_delta_strain'] = baseline_strain - p['as_cast_strain']
        cols['baseline_delta_tensile'] = baseline_tensile - p['as_cast_tensile']
        cols['baseline_delta_yield'] = baseline_yield - p['as_cast_yield']

        # 基线派生结构特征
        cols['baseline_yield_tensile_ratio'] = baseline_yield / np.clip(baseline_tensile, 1e-6, None)
        cols['baseline_tensile_minus_yield'] = baseline_tensile - baseline_yield
        cols['baseline_strength_ductility'] = baseline_tensile * baseline_strain

        # 基线与机制边界相互作用，帮助模型学习“在哪些边界附近基线会失效”
        cols['baseline_tensile_x_highT'] = baseline_tensile * cols['is_high_temp_regime']
        cols['baseline_yield_x_longt'] = baseline_yield * cols['is_long_time_regime']
        cols['baseline_strain_x_window'] = baseline_strain * local_window

        # ------------------------------------------------------------------
        # 6) 少量归一化/无量纲特征
        # ------------------------------------------------------------------
        cols['temp_norm'] = (temp - p['train_temp_min']) / (p['train_temp_max'] - p['train_temp_min'])
        cols['time_log_norm'] = log_time / np.log1p(p['train_time_max'])
        cols['combined_process_index'] = 0.6 * cols['temp_norm'] + 0.4 * cols['time_log_norm']

        X = pd.DataFrame(cols, index=df.index)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.tolist()
        return X.values

    def get_feature_names(self):
        return self.feature_names