#!/usr/bin/env python3
"""
对齐 MOSFET 功率循环中的电压与温度数据。

数据约定（基于当前仓库 data 目录）：
1) 按器件目录组织：
   data/
   └── 12号/
       ├── PC/
       └── 温度/
2) PC 文件命名：
   YYYYMMDD_PC_<group_id>_<device_id>_chX.mat
   其中 ch1/ch3/ch5 为同步采样通道，本脚本以 ch1 对齐温度后，同步切片 ch3/ch5。
3) 温度文件命名：
   <date>_<group_or_range>_<device_id>.mat
   例如：251108_0_12.mat、251108_1-10_12.mat、2511016_11-20_12.mat

核心处理逻辑：
1) 在每组 ch1 中，基于高低电平阈值找到“高 -> 低”的边沿。
2) 以第一和第十个边沿为锚点，使用可调 pre/post 点数截取该组有效段，去除电路切换段。
3) 在温度序列中寻找对应 10 周期峰值窗口，以同样 pre 点数回退作为起点，对齐温度切片。
4) 将同一组的 ch1/ch3/ch5 采用相同索引切片，并与温度做等长裁剪。

新增兼容逻辑（用于异常/失效工况）：
1) 主流程会跳过温度文件中的 _0_ 独立单循环文件，仅处理常规区间（如 1-10、11-20）。
2) 最后一组允许不足 10 循环（可配置），避免因器件提前失效导致整组被丢弃。
3) _0_ 单循环使用专用对齐：ch1 的高->低边沿 对齐 温度最高点，再按电压长度裁切温度。

维护与改参说明（建议先看）：
1) 你最常改的参数：
   - --high-threshold：电压高电平阈值（默认 2.0V）
   - --pre-points / --post-points：以“高->低边沿”为锚点的窗口前后长度
   - --peak-smooth-window / --peak-prom-factor：温度峰值识别灵敏度
2) 数据质量检查：
   - 优先看 output/plots/group_XX_check.png
   - 再看 alignment_summary.csv 里的 falling_edges_all/falling_edges_used/temp_peak_idx_*
3) 常见问题定位：
   - “边沿数量不足”：通常是阈值不合适，或电压文件通道选错
   - “温度数据耗尽”：前面组切片过长，导致后续组无可用温度数据
   - “温度峰值检测失败”：先调大平滑窗口，再调低峰值显著性系数
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# =========================
# 快速改参区（后期最常改）
# =========================

DEFAULT_INPUT_ROOT = r'H:\2026.2.10数据\老化数据' #"data"
DEFAULT_OUTPUT_ROOT = r'G:\深度学习资料\双脉冲实验平台\2026.3.1整理数据输出\aligned_output'  #"data/aligned_output"
# 可手动填入多个器件编号，如 ["12", "13", "21"]
DEFAULT_DEVICE_IDS = ['17']
# 部分循环最小循环数（可改为 1，但误检风险更高）
DEFAULT_PARTIAL_MIN_CYCLES = 3


PC_FILE_RE = re.compile(r"^(\d{8})_PC_(\d+)_(\d+)_ch(\d+)\.mat$", re.IGNORECASE)
TEMP_FILE_RE = re.compile(r"^(\d{6,8})_([0-9]+(?:-[0-9]+)?)_(\d+)\.mat$", re.IGNORECASE)


@dataclass
class AlignConfig:
    """
    处理参数集合。

    字段含义（按调参优先级）：
    - sample_rate_hz：采样率（Hz），本项目默认 10Hz
    - high_level_threshold：电压高电平判定阈值（V）
    - cycles_per_group：每组内的功率循环个数（默认 10）
    - pre_fall_points：相对“高->低边沿”的前向保留点数
    - post_fall_points：相对“最后一个高->低边沿”的后向保留点数
    - edge_min_gap_seconds：相邻边沿最小间隔（秒），用于抑制抖动误检
    - peak_smooth_window：温度峰值检测前的平滑窗口（点）
    - peak_min_prominence_factor：峰值显著性阈值系数（相对标准差）
    - temp_mode / temp_signal_index / temp_signal_name：温度通道选择方式
    - max_groups：最多处理组数；0 表示处理全部
    """
    sample_rate_hz: float = 10.0
    high_level_threshold: float = 2.0
    cycles_per_group: int = 10
    pre_fall_points: int = 155
    post_fall_points: int = 245
    edge_min_gap_seconds: float = 20.0
    peak_smooth_window: int = 7
    peak_min_prominence_factor: float = 0.10
    peak_min_gap_seconds: float = 20.0
    temp_mode: str = "index"
    temp_signal_index: int = 9
    temp_signal_name: str = "tempture"
    max_groups: int = 0
    strict_mode: bool = False
    allow_partial_last_group: bool = True
    allow_partial_any_group: bool = False
    partial_min_cycles: int = DEFAULT_PARTIAL_MIN_CYCLES
    allow_single_cycle_temp_peak_fallback: bool = True

    @property
    def expected_cycle_points(self) -> int:
        # 15s 高电平 + 25s 低电平，默认 10Hz 采样
        return int(round((15.0 + 25.0) * self.sample_rate_hz))

    @property
    def edge_min_gap_points(self) -> int:
        return max(1, int(round(self.edge_min_gap_seconds * self.sample_rate_hz)))

    @property
    def peak_min_gap_points(self) -> int:
        return max(1, int(round(self.peak_min_gap_seconds * self.sample_rate_hz)))


@dataclass
class GroupAlignedResult:
    """
    单组对齐结果，用于输出文件和可视化。

    说明：
    - voltage_* 字段记录电压切片区间
    - temp_* 字段记录温度切片区间/峰值位置
    - *_segment 是最终“组内等长”数据
    - voltage_raw_signal 仅用于绘图诊断，不会写入最终 mat/npz 的主结构
    """
    device_id: str
    group_id: int
    ch1_file: str
    ch3_file: str
    ch5_file: str
    ch1_raw_len: int
    ch3_raw_len: int
    ch5_raw_len: int
    voltage_start_idx: int
    voltage_end_idx: int
    voltage_trim_len: int
    falling_edge_count_all: int
    falling_edges_all: list[int]
    falling_edges_used: list[int]
    temp_peak_count_all: int
    temp_peaks_used_global: list[int]
    temp_peak_idx_global: int
    temp_peak_idx_in_segment: int
    temp_start_idx: int
    temp_end_idx: int
    aligned_len: int
    used_cycles: int
    is_partial_group: bool
    alignment_mode: str
    ch1_segment: np.ndarray
    ch3_segment: np.ndarray
    ch5_segment: np.ndarray
    temperature_segment: np.ndarray
    temperature_time_segment: np.ndarray
    voltage_raw_signal: np.ndarray


@dataclass
class GroupInput:
    """单组输入文件（同组内多通道同步采样）。"""

    group_id: int
    ch1_file: Path
    ch3_file: Path
    ch5_file: Path


def _to_1d_float_array(value: Any) -> np.ndarray | None:
    """将输入安全转换为一维 float 数组；失败时返回 None。"""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def _import_scipy_io() -> tuple[Any, Any]:
    """延迟导入 scipy，避免仅查看 --help 时因缺依赖而退出。"""
    try:
        from scipy.io import loadmat, savemat
    except ImportError as exc:
        raise RuntimeError("未找到 scipy，请先安装后再运行（pip install scipy）。") from exc
    return loadmat, savemat


def _import_scipy_find_peaks() -> Any:
    """延迟导入 scipy.signal.find_peaks。"""
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise RuntimeError("未找到 scipy.signal，请先安装 scipy。") from exc
    return find_peaks


def _import_matplotlib_pyplot() -> Any:
    """延迟导入 matplotlib，并固定 Agg 后端以支持无 GUI 环境生成 PNG。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("未找到 matplotlib，请先安装后再运行（pip install matplotlib）。") from exc
    return plt


def _import_h5py() -> Any:
    """延迟导入 h5py，用于无 matlab.engine 时解析 v7.3 温度文件。"""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("未找到 h5py，请先安装后再运行（pip install h5py）。") from exc
    return h5py


def _collect_numeric_candidates(obj: Any, prefix: str, out: list[tuple[str, np.ndarray]], depth: int = 0) -> None:
    """
    从任意 MATLAB 反序列化对象中递归收集“可能是信号”的数值数组。

    该函数用于电压文件的兜底读取：
    - 首选变量 data
    - 若 data 缺失或类型异常，则在结构体/对象中递归找最长数值数组
    """
    if depth > 6:
        return

    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            # 结构化数组：递归各字段
            for name in obj.dtype.names:
                _collect_numeric_candidates(obj[name], f"{prefix}.{name}", out, depth + 1)
            return

        if obj.dtype == object:
            # object 数组：递归每个元素
            for idx, item in np.ndenumerate(obj):
                _collect_numeric_candidates(item, f"{prefix}[{idx}]", out, depth + 1)
            return

        if np.issubdtype(obj.dtype, np.number):
            arr = obj.astype(float, copy=False).reshape(-1)
            if arr.size > 5:
                out.append((prefix, arr))
            return

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _collect_numeric_candidates(item, f"{prefix}[{i}]", out, depth + 1)
        return

    # MATLAB struct_as_record=False 场景可能是 mat_struct，走属性递归
    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            if key.startswith("_"):
                continue
            _collect_numeric_candidates(value, f"{prefix}.{key}", out, depth + 1)


def load_voltage_signal(voltage_file: Path) -> np.ndarray:
    """
    从 Tek v5 mat 文件中提取电压序列。
    优先读 data 变量；若失败，回退到“最长数值数组”策略。
    """
    loadmat, _ = _import_scipy_io()
    mat_data = loadmat(voltage_file, squeeze_me=True, struct_as_record=False)
    variables = {k: v for k, v in mat_data.items() if not k.startswith("__")}

    if "data" in variables:
        arr = _to_1d_float_array(variables["data"])
        if arr is not None:
            return arr

    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in variables.items():
        _collect_numeric_candidates(value, key, candidates)

    if not candidates:
        raise ValueError(f"无法从电压文件提取数值序列: {voltage_file}")

    # 优先包含 data 关键字，其次长度更长
    candidates.sort(key=lambda x: ("data" in x[0].lower(), x[1].size), reverse=True)
    return candidates[0][1]


def parse_device_ids(device_ids_text: str) -> list[str]:
    """将逗号分隔的器件编号字符串解析为列表。"""
    ids = [x.strip() for x in device_ids_text.split(",") if x.strip()]
    if not ids:
        raise ValueError("device_ids 为空，请至少提供一个器件编号。")
    return ids


def parse_group_range_token(token: str) -> tuple[int, int]:
    """
    将温度文件中的组号标识解析为区间。
    示例：
    - "0" -> (0, 0)
    - "1-10" -> (1, 10)
    """
    if "-" in token:
        a, b = token.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(token)
    if end < start:
        start, end = end, start
    return start, end


def discover_device_dir(input_root: Path, device_id: str) -> Path:
    """
    发现器件目录，优先匹配“12号”风格，次选纯编号目录。
    """
    candidates = [
        input_root / f"{device_id}号",
        input_root / device_id,
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"未找到器件目录: {candidates[0]} 或 {candidates[1]}")


def discover_pc_group_inputs(pc_dir: Path, device_id: str, required_channels: tuple[int, ...] = (1, 3, 5)) -> list[GroupInput]:
    """
    扫描 PC 目录，返回具备 ch1/ch3/ch5 的完整组列表。

    命名约定：
    YYYYMMDD_PC_<group_id>_<device_id>_chX.mat
    """
    groups: dict[int, dict[int, Path]] = {}
    for path in pc_dir.glob("*.mat"):
        m = PC_FILE_RE.match(path.name)
        if not m:
            continue
        _, group_text, dev_text, ch_text = m.groups()
        if str(int(dev_text)) != str(int(device_id)):
            continue
        group_id = int(group_text)
        ch = int(ch_text)
        if ch not in required_channels:
            continue
        groups.setdefault(group_id, {})[ch] = path

    outputs: list[GroupInput] = []
    for group_id in sorted(groups.keys()):
        row = groups[group_id]
        if all(ch in row for ch in required_channels):
            outputs.append(
                GroupInput(
                    group_id=group_id,
                    ch1_file=row[1],
                    ch3_file=row[3],
                    ch5_file=row[5],
                )
            )
    return outputs


def discover_temperature_files(temp_dir: Path, device_id: str) -> tuple[list[Path], list[Path]]:
    """
    扫描温度目录并按规则分桶：
    - regular_files：常规区间文件（如 1-10 / 11-20）
    - zero_group_files：_0_ 单循环文件（如 251108_0_12.mat）

    命名约定：
    <date>_<range_or_group>_<device_id>.mat
    示例：
    251108_0_12.mat
    251108_1-10_12.mat
    2511016_11-20_12.mat
    """
    regular_rows: list[tuple[int, int, str, Path]] = []
    zero_rows: list[tuple[int, int, str, Path]] = []
    for path in temp_dir.glob("*.mat"):
        m = TEMP_FILE_RE.match(path.name)
        if not m:
            continue
        date_text, range_text, dev_text = m.groups()
        if str(int(dev_text)) != str(int(device_id)):
            continue
        start, end = parse_group_range_token(range_text)
        row = (start, end, date_text, path)
        if start == 0 and end == 0:
            zero_rows.append(row)
        else:
            regular_rows.append(row)

    regular_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3].name))
    zero_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3].name))
    return [x[3] for x in regular_rows], [x[3] for x in zero_rows]


def load_temperature_series_from_files(temp_files: list[Path], cfg: AlignConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """
    将多个温度片段文件拼接为一条连续温度序列。

    时间轴策略：
    - 对每个片段按 sample_rate_hz 重建局部时间
    - 拼接时使用累计样本数构建全局时间
    """
    if not temp_files:
        raise FileNotFoundError("温度文件列表为空，无法构建温度序列。")

    all_temp: list[np.ndarray] = []
    temp_signal_name: str | None = None
    total_len = 0
    for p in temp_files:
        _, y, name = load_temperature_signal(p, cfg)
        all_temp.append(y.reshape(-1))
        if temp_signal_name is None:
            temp_signal_name = name
        total_len += int(y.size)

    temperature_data = np.concatenate(all_temp, axis=0)
    temperature_time = np.arange(total_len, dtype=float) / float(cfg.sample_rate_hz)
    return temperature_time, temperature_data, (temp_signal_name or "temperature")


def detect_falling_edges(voltage: np.ndarray, threshold: float, min_gap_points: int) -> np.ndarray:
    """
    识别“高电平 -> 低电平”边沿。

    算法：
    1) voltage > threshold 判定高电平布尔序列
    2) 从 True -> False 的跃迁点作为候选边沿
    3) 使用最小间隔约束过滤掉抖动引起的伪边沿
    """
    high = voltage > threshold
    raw_edges = np.flatnonzero(high[:-1] & (~high[1:])) + 1
    if raw_edges.size == 0:
        return raw_edges

    filtered = [int(raw_edges[0])]
    for idx in raw_edges[1:]:
        if int(idx) - filtered[-1] >= min_gap_points:
            filtered.append(int(idx))
    return np.asarray(filtered, dtype=int)


def select_group_edges(edges: np.ndarray, cycles_per_group: int, expected_cycle_points: int) -> np.ndarray:
    """
    从候选边沿中选出“最像一个完整测试组”的窗口（默认 10 个边沿）。

    当候选边沿数 > 目标边沿数时，滑窗评分规则：
    - 相邻边沿间距与 expected_cycle_points 的平均偏差越小越好
    - 间距标准差越小越好
    - 对明显异常间距加惩罚项
    """
    if edges.size < cycles_per_group:
        raise ValueError(
            f"检测到的高->低边沿数量不足：{edges.size}，期望至少 {cycles_per_group}。"
        )

    if edges.size == cycles_per_group:
        return edges

    # 优先策略：返回“最早出现的合法窗口”，避免跨组选择到后面的更优窗口
    min_gap = 0.85 * expected_cycle_points
    max_gap = 1.15 * expected_cycle_points
    for start in range(0, edges.size - cycles_per_group + 1):
        window = edges[start : start + cycles_per_group]
        gaps = np.diff(window)
        if gaps.size == 0:
            continue
        if np.all(gaps >= min_gap) and np.all(gaps <= max_gap):
            return window

    # 回退策略：若没有严格合法窗口，再用评分挑选最接近期望周期的窗口
    best_window: np.ndarray | None = None
    best_score: float | None = None

    for start in range(0, edges.size - cycles_per_group + 1):
        window = edges[start : start + cycles_per_group]
        gaps = np.diff(window)
        if gaps.size == 0:
            score = 0.0
        else:
            mean_abs_err = float(np.mean(np.abs(gaps - expected_cycle_points)))
            std_val = float(np.std(gaps))
            score = mean_abs_err + std_val
            if np.any(gaps < 0.5 * expected_cycle_points) or np.any(gaps > 1.8 * expected_cycle_points):
                score += float(expected_cycle_points)

        if best_score is None or score < best_score:
            best_score = score
            best_window = window

    if best_window is None:
        raise ValueError("无法从边沿序列中选出有效的 10 周期窗口。")
    return best_window


def smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均平滑，用于提升温度峰值识别稳定性。"""
    if window <= 1:
        return signal
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def find_first_peak_idx(signal: np.ndarray, cfg: AlignConfig) -> int:
    """
    在当前温度片段内寻找“第一个有效峰值”。

    处理步骤：
    1) 平滑（降低噪声）
    2) 搜索局部极大值
    3) 用前后 1 秒抬升/下降幅度做显著性筛选
    4) 若筛选失败，回退到首个局部峰，保证流程不中断
    """
    if signal.size < 3:
        raise ValueError("温度序列太短，无法搜索波峰。")

    smooth = smooth_signal(signal, cfg.peak_smooth_window)
    candidates = np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    if candidates.size == 0:
        raise ValueError("温度序列未检测到局部峰值。")

    # 使用前后 1 秒窗口估计峰值显著性，降低噪声峰误检概率
    flank = max(2, int(round(cfg.sample_rate_hz)))
    std_val = float(np.std(smooth))
    min_delta = max(1e-6, cfg.peak_min_prominence_factor * std_val)

    for idx in candidates:
        if idx - flank < 0 or idx + flank >= smooth.size:
            continue
        rise = float(smooth[idx] - smooth[idx - flank])
        fall = float(smooth[idx] - smooth[idx + flank])
        if rise > min_delta and fall > min_delta:
            return int(idx)

    # 回退：若显著性筛选失败，使用第一个局部峰
    return int(candidates[0])


def detect_temperature_peaks(signal: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    """
    识别温度序列中的有效峰值，并过滤噪声峰。

    过滤策略：
    1) 先做平滑
    2) 用 scipy.signal.find_peaks 按 prominence + 最小间距检峰
    3) 若峰数过少，自动放宽为“仅最小间距”模式，保证流程鲁棒
    """
    if signal.size < 3:
        return np.asarray([], dtype=int)

    smooth = smooth_signal(signal, cfg.peak_smooth_window)
    find_peaks = _import_scipy_find_peaks()
    min_gap = cfg.peak_min_gap_points
    prominence = max(1e-6, cfg.peak_min_prominence_factor * float(np.std(smooth)))

    # 先用 prominence 约束，避免噪声峰
    peaks, _ = find_peaks(smooth, distance=min_gap, prominence=prominence)

    # 若过严导致峰数量过少，则放宽为仅 distance 约束
    if peaks.size < cfg.cycles_per_group:
        peaks, _ = find_peaks(smooth, distance=min_gap)

    # 兜底：仍无峰时，用最简单局部极大值
    if peaks.size == 0:
        peaks = np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1

    return np.asarray(peaks, dtype=int)


def decide_cycles_for_group(detected_count: int, cfg: AlignConfig, is_last_group: bool) -> tuple[int, bool]:
    """
    决定当前组实际采用多少个循环边沿。

    规则：
    1) 常规情况：>= cycles_per_group 时按满组处理
    2) 不足满组：按配置决定是否允许部分循环
       - allow_partial_last_group=True：允许最后一组部分循环
       - allow_partial_any_group=True：允许任意组部分循环
    3) 部分组仍需满足 partial_min_cycles
    """
    target = int(cfg.cycles_per_group)
    if detected_count >= target:
        return target, False

    allow_partial = bool(cfg.allow_partial_any_group) or (
        bool(cfg.allow_partial_last_group) and bool(is_last_group)
    )
    if not allow_partial:
        raise ValueError(f"检测到边沿数量不足：{detected_count}，期望至少 {target}。")

    used = int(detected_count)
    if used < int(max(1, cfg.partial_min_cycles)):
        raise ValueError(
            f"边沿数仅 {used}，低于 partial_min_cycles={cfg.partial_min_cycles}，无法做部分组对齐。"
        )
    return used, True


def select_zero_group_edge(voltage: np.ndarray, edges: np.ndarray, cfg: AlignConfig) -> int:
    """
    为 _0_ 单循环组选择最可信的“高->低”边沿。

    评分思路：
    - pre_high_ratio：边沿前窗口中“高电平”占比（越高越好）
    - post_low_ratio：边沿后窗口中“低电平”占比（越高越好）
    - drop_norm：边沿瞬时跌落幅度（归一化后越大越好）
    """
    if edges.size == 0:
        raise ValueError("_0_ 单循环组没有可用边沿候选。")

    pre_pts = max(5, int(round(3.0 * cfg.sample_rate_hz)))
    post_pts = max(5, int(round(3.0 * cfg.sample_rate_hz)))
    low_level_threshold = min(cfg.high_level_threshold * 0.5, 1.0)
    amp_span = max(1e-6, float(np.max(voltage) - np.min(voltage)))

    best_edge = int(edges[0])
    best_score = -1e12

    for e in edges:
        edge = int(e)
        pre = voltage[max(0, edge - pre_pts) : edge]
        post = voltage[edge : min(voltage.size, edge + post_pts)]
        if pre.size < 3 or post.size < 3:
            continue

        pre_high_ratio = float(np.mean(pre > cfg.high_level_threshold))
        post_low_ratio = float(np.mean(post < low_level_threshold))

        pre_short = pre[-min(pre.size, 5) :]
        post_short = post[: min(post.size, 5)]
        drop = float(np.mean(pre_short) - np.mean(post_short))
        drop_norm = drop / amp_span

        score = 3.0 * pre_high_ratio + 4.0 * post_low_ratio + 2.0 * drop_norm
        if score > best_score:
            best_score = score
            best_edge = edge

    return best_edge


def _list_temperature_signals_matlab(temp_file: Path) -> list[str]:
    """用 matlab.engine 枚举 Simulink Dataset 中的信号名。"""
    try:
        import matlab.engine
    except ImportError as exc:
        raise RuntimeError("未找到 matlab.engine，请在 MATLAB 环境中安装 Python Engine。") from exc

    eng = matlab.engine.start_matlab()
    try:
        eng.workspace["MATPATH"] = str(temp_file.resolve())
        eng.eval("S = load(MATPATH);", nargout=0)
        eng.eval("N = S.data.numElements;", nargout=0)
        n = int(float(eng.workspace["N"]))
        names: list[str] = []
        for i in range(1, n + 1):
            eng.workspace["I"] = float(i)
            eng.eval("nm = S.data{I}.Name;", nargout=0)
            names.append(str(eng.workspace["nm"]))
        return names
    finally:
        eng.quit()


def _decode_uint16_string(arr: np.ndarray) -> str:
    """将 MATLAB UTF-16 编码字符数组转换为 Python 字符串。"""
    flat = np.asarray(arr).reshape(-1)
    chars = [chr(int(x)) for x in flat if int(x) != 0]
    return "".join(chars)


def _load_sigstream_channels_h5(temp_file: Path) -> tuple[list[str], dict[int, np.ndarray]]:
    """
    在无 MATLAB 环境时，直接解析 v7.3(HDF5) 文件中的 Sigstream 通道。

    返回：
    - names：通道名称列表（1-based 对应）
    - channels：{1: arr1, 2: arr2, ...}
    """
    h5py = _import_h5py()

    with h5py.File(temp_file, "r") as f:
        if "/#refs#/c/Elements" not in f or "/#sigstream#" not in f:
            raise RuntimeError("温度文件不是预期的 Simulink Dataset v7.3 结构。")

        element_refs = f["/#refs#/c/Elements"][:]
        names: list[str] = []
        for i in range(element_refs.shape[0]):
            obj = f[element_refs[i, 0]]
            name = _decode_uint16_string(np.asarray(obj["Name"][:]))
            names.append(name)

        sigstream = f["/#sigstream#"]
        n_records = int(np.asarray(sigstream["#nRecords#"]).reshape(-1)[0])
        channels: dict[int, np.ndarray] = {}
        for i in range(n_records):
            g = sigstream[str(i)]
            sample_size = int(np.asarray(g["#sampleSize#"]).reshape(-1)[0])
            length = int(np.asarray(g["#length#"]).reshape(-1)[0])
            raw = np.asarray(g["#data#"][:], dtype=np.uint8)

            if sample_size == 8:
                arr = raw.view("<f8")
            elif sample_size == 4:
                arr = raw.view("<f4").astype(float)
            elif sample_size == 2:
                arr = raw.view("<f2").astype(float)
            else:
                # 未知采样宽度时回退为字节值（尽量不中断流程）
                arr = raw.astype(float)

            channels[i + 1] = np.asarray(arr[:length], dtype=float).reshape(-1)

    return names, channels


def list_temperature_signals_h5(temp_file: Path) -> list[str]:
    """通过 h5py 枚举温度文件信号。"""
    names, _ = _load_sigstream_channels_h5(temp_file)
    return names


def list_temperature_signals(temp_file: Path) -> list[str]:
    """优先 matlab.engine；失败则自动回退 h5py。"""
    try:
        return _list_temperature_signals_matlab(temp_file)
    except Exception as exc:
        print(f"[WARN] matlab.engine 不可用或读取失败，改用 h5py 解析温度文件: {exc}")
        return list_temperature_signals_h5(temp_file)


def _load_temperature_signal_matlab(temp_file: Path, cfg: AlignConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """用 matlab.engine 读取温度通道（含 Time/Data）。"""
    try:
        import matlab.engine
    except ImportError as exc:
        raise RuntimeError("未找到 matlab.engine，请在 MATLAB 环境中安装 Python Engine。") from exc

    eng = matlab.engine.start_matlab()
    try:
        eng.eval("warning('off','all');", nargout=0)
        eng.workspace["MATPATH"] = str(temp_file.resolve())
        eng.eval("S = load(MATPATH);", nargout=0)

        mode = cfg.temp_mode.lower()
        if mode == "index":
            eng.workspace["IDX"] = float(cfg.temp_signal_index)
            eng.eval("sig = S.data{IDX};", nargout=0)
        elif mode == "name":
            eng.workspace["TARGET"] = cfg.temp_signal_name
            eng.eval(
                """
                n = S.data.numElements;
                hit = 0;
                for k = 1:n
                    if strcmp(string(S.data{k}.Name), string(TARGET))
                        sig = S.data{k};
                        hit = k;
                        break;
                    end
                end
                if hit == 0
                    error("Signal name not found: %s", string(TARGET));
                end
                """,
                nargout=0,
            )
        else:
            raise ValueError("temp_mode 仅支持 index 或 name。")

        eng.eval("ts = sig.Values;", nargout=0)
        eng.eval("t = ts.Time;", nargout=0)
        eng.eval("y = ts.Data;", nargout=0)
        eng.eval("nm = sig.Name;", nargout=0)

        t = np.asarray(eng.workspace["t"], dtype=float).reshape(-1)
        y = np.asarray(eng.workspace["y"], dtype=float).reshape(-1)
        name = str(eng.workspace["nm"])
        return t, y, name
    finally:
        eng.quit()


def _load_temperature_signal_h5(temp_file: Path, cfg: AlignConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """
    用 h5py 读取温度通道。

    注意：
    - Sigstream 内没有独立 Time 变量，因此按 sample_rate_hz 人工构造时间轴。
    """
    names, channels = _load_sigstream_channels_h5(temp_file)
    if not channels:
        raise RuntimeError("未从温度文件中解析到任何信号通道。")

    mode = cfg.temp_mode.lower()
    if mode == "index":
        idx = cfg.temp_signal_index
        if idx not in channels:
            raise ValueError(f"温度信号索引 {idx} 超出范围，可用范围 1~{len(channels)}。")
        signal = channels[idx]
        name = names[idx - 1] if 1 <= idx <= len(names) else f"signal_{idx}"
    elif mode == "name":
        target = cfg.temp_signal_name.strip().lower()
        match_idx = None
        for i, name in enumerate(names, start=1):
            if name.strip().lower() == target:
                match_idx = i
                break
        if match_idx is None:
            raise ValueError(f"未在温度文件中找到名为 '{cfg.temp_signal_name}' 的信号。")
        signal = channels[match_idx]
        name = names[match_idx - 1]
    else:
        raise ValueError("temp_mode 仅支持 index 或 name。")

    # Sigstream 内无显式时间轴，用采样率构造时间向量
    time = np.arange(signal.size, dtype=float) / float(cfg.sample_rate_hz)
    return time, signal, name


def load_temperature_signal(temp_file: Path, cfg: AlignConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """统一温度读取入口：MATLAB 优先，h5py 回退。"""
    try:
        return _load_temperature_signal_matlab(temp_file, cfg)
    except Exception as exc:
        print(f"[WARN] matlab.engine 不可用或读取失败，改用 h5py 解析温度文件: {exc}")
        return _load_temperature_signal_h5(temp_file, cfg)


def align_groups(
    device_id: str,
    group_inputs: list[GroupInput],
    temperature_time: np.ndarray,
    temperature_data: np.ndarray,
    cfg: AlignConfig,
) -> list[GroupAlignedResult]:
    """
    核心对齐流程（逐组处理）。

    关键设计：
    - 电压组与温度组“顺序消费”，每处理一组就推进 temp_cursor
    - 每组最后做等长裁剪，确保温度/电压样本点数一致
    """
    results: list[GroupAlignedResult] = []
    temp_cursor = 0

    total_groups = len(group_inputs)
    for idx, item in enumerate(group_inputs):
        group_id = item.group_id
        try:
            # A) 三通道同步读取；对齐逻辑以 ch1（原电压通道）为基准
            ch1_raw = load_voltage_signal(item.ch1_file)
            ch3_raw = load_voltage_signal(item.ch3_file)
            ch5_raw = load_voltage_signal(item.ch5_file)

            # 理论上三通道等长；若异常则截到最短长度以保证同步索引有效
            raw_min_len = int(min(ch1_raw.size, ch3_raw.size, ch5_raw.size))
            ch1_raw = ch1_raw[:raw_min_len]
            ch3_raw = ch3_raw[:raw_min_len]
            ch5_raw = ch5_raw[:raw_min_len]

            edges_all = detect_falling_edges(
                voltage=ch1_raw,
                threshold=cfg.high_level_threshold,
                min_gap_points=cfg.edge_min_gap_points,
            )
            used_cycles, is_partial = decide_cycles_for_group(
                detected_count=int(edges_all.size),
                cfg=cfg,
                is_last_group=(idx == total_groups - 1),
            )
            edges_used = select_group_edges(
                edges=edges_all,
                cycles_per_group=used_cycles,
                expected_cycle_points=cfg.expected_cycle_points,
            )

            voltage_start = max(0, int(edges_used[0]) - cfg.pre_fall_points)
            voltage_end = min(ch1_raw.size, int(edges_used[-1]) + cfg.post_fall_points)
            if voltage_end <= voltage_start:
                raise ValueError(f"组 {group_id} 的电压切片区间无效：[{voltage_start}, {voltage_end})")

            ch1_trim = ch1_raw[voltage_start:voltage_end]
            ch3_trim = ch3_raw[voltage_start:voltage_end]
            ch5_trim = ch5_raw[voltage_start:voltage_end]

            # B) 温度：在“尚未消费”的剩余序列中找 10 峰窗口，按首峰对齐截取
            if temp_cursor >= temperature_data.size:
                raise ValueError("温度数据已经耗尽，无法继续对齐后续电压组。")

            temp_remaining = temperature_data[temp_cursor:]
            temp_peaks_rel = detect_temperature_peaks(temp_remaining, cfg)
            if (
                temp_peaks_rel.size == 0
                and used_cycles == 1
                and is_partial
                and cfg.allow_single_cycle_temp_peak_fallback
            ):
                # 对“单循环部分组”做兜底：若无局部峰，使用全局最大值作为锚点。
                temp_peaks_rel = np.asarray([int(np.argmax(temp_remaining))], dtype=int)
            temp_peaks_used_rel = select_group_edges(
                edges=temp_peaks_rel,
                cycles_per_group=used_cycles,
                expected_cycle_points=cfg.expected_cycle_points,
            )
            temp_peak_rel = int(temp_peaks_used_rel[0])
            temp_peak_global = temp_cursor + temp_peak_rel
            temp_start = max(temp_cursor, temp_peak_global - cfg.pre_fall_points)
            temp_end = temp_start + ch1_trim.size

            if temp_end > temperature_data.size:
                temp_end = temperature_data.size

            temp_segment = temperature_data[temp_start:temp_end]
            time_segment = temperature_time[temp_start:temp_end]

            # C) 组内等长裁剪（温度 + ch1/ch3/ch5 全部一致）
            common_len = int(min(ch1_trim.size, ch3_trim.size, ch5_trim.size, temp_segment.size))
            if common_len <= 0:
                raise ValueError(f"组 {group_id} 对齐后长度为 0，请检查阈值与峰值检测参数。")

            ch1_aligned = ch1_trim[:common_len]
            ch3_aligned = ch3_trim[:common_len]
            ch5_aligned = ch5_trim[:common_len]
            temp_aligned = temp_segment[:common_len]
            time_aligned = time_segment[:common_len]
            temp_peak_local = int(temp_peak_global - temp_start)

            results.append(
                GroupAlignedResult(
                    device_id=device_id,
                    group_id=group_id,
                    ch1_file=item.ch1_file.name,
                    ch3_file=item.ch3_file.name,
                    ch5_file=item.ch5_file.name,
                    ch1_raw_len=int(ch1_raw.size),
                    ch3_raw_len=int(ch3_raw.size),
                    ch5_raw_len=int(ch5_raw.size),
                    voltage_start_idx=voltage_start,
                    voltage_end_idx=voltage_start + common_len,
                    voltage_trim_len=int(ch1_trim.size),
                    falling_edge_count_all=int(edges_all.size),
                    falling_edges_all=[int(x) for x in edges_all.tolist()],
                    falling_edges_used=[int(x) for x in edges_used.tolist()],
                    temp_peak_count_all=int(temp_peaks_rel.size),
                    temp_peaks_used_global=[int(temp_cursor + x) for x in temp_peaks_used_rel.tolist()],
                    temp_peak_idx_global=int(temp_peak_global),
                    temp_peak_idx_in_segment=temp_peak_local,
                    temp_start_idx=int(temp_start),
                    temp_end_idx=int(temp_start + common_len),
                    aligned_len=common_len,
                    used_cycles=used_cycles,
                    is_partial_group=is_partial,
                    alignment_mode=(
                        "regular_partial_last"
                        if (is_partial and idx == total_groups - 1)
                        else ("regular_partial_any" if is_partial else "regular_full")
                    ),
                    ch1_segment=ch1_aligned,
                    ch3_segment=ch3_aligned,
                    ch5_segment=ch5_aligned,
                    temperature_segment=temp_aligned,
                    temperature_time_segment=time_aligned,
                    voltage_raw_signal=ch1_raw,
                )
            )

            # 删除已对齐区段，进入下一组
            temp_cursor = int(temp_start + common_len)
        except Exception as exc:
            if cfg.strict_mode:
                raise
            print(f"[WARN] 器件 {device_id} group={group_id:02d} 跳过，原因: {exc}")

    return results


def align_zero_group(
    device_id: str,
    zero_group_input: GroupInput,
    zero_temp_file: Path,
    cfg: AlignConfig,
) -> tuple[GroupAlignedResult, str]:
    """
    _0_ 单循环专用对齐：
    - 电压锚点：ch1 第一个有效“高->低边沿”
    - 温度锚点：温度序列最高点
    - 按电压序列长度裁切温度，必要时两端同步裁剪以保证锚点对应
    """
    ch1_raw = load_voltage_signal(zero_group_input.ch1_file)
    ch3_raw = load_voltage_signal(zero_group_input.ch3_file)
    ch5_raw = load_voltage_signal(zero_group_input.ch5_file)
    raw_min_len = int(min(ch1_raw.size, ch3_raw.size, ch5_raw.size))
    ch1_raw = ch1_raw[:raw_min_len]
    ch3_raw = ch3_raw[:raw_min_len]
    ch5_raw = ch5_raw[:raw_min_len]

    # _0_ 单循环组使用更小的 min_gap，先收集更多候选，再通过评分选最可信边沿。
    zero_min_gap = max(1, int(round(cfg.sample_rate_hz)))
    edges_all = detect_falling_edges(
        voltage=ch1_raw,
        threshold=cfg.high_level_threshold,
        min_gap_points=zero_min_gap,
    )
    if edges_all.size == 0:
        raise ValueError("单循环组未检测到高->低边沿。")
    edge_idx = select_zero_group_edge(ch1_raw, edges_all, cfg)

    temp_time, temp_data, temp_signal_name = load_temperature_signal(zero_temp_file, cfg)
    if temp_data.size <= 0:
        raise ValueError("单循环温度文件为空。")

    # 使用平滑后峰值索引，提高抗噪能力；锚点仍对应“最高点”语义。
    temp_smooth = smooth_signal(temp_data, cfg.peak_smooth_window)
    temp_peak_idx = int(np.argmax(temp_smooth))

    voltage_start = 0
    voltage_end = int(raw_min_len)
    target_len = voltage_end - voltage_start
    temp_start = int(temp_peak_idx - edge_idx)
    temp_end = int(temp_start + target_len)

    # 若温度前段不足，等量裁掉电压头部，维持锚点在对齐后的同一相对位置。
    if temp_start < 0:
        head_short = -temp_start
        voltage_start += head_short
        temp_start = 0
        temp_end = temp_start + (voltage_end - voltage_start)

    # 若温度尾段不足，等量裁掉电压尾部，维持等长。
    if temp_end > int(temp_data.size):
        tail_over = temp_end - int(temp_data.size)
        voltage_end -= tail_over
        temp_end = int(temp_data.size)

    if voltage_end <= voltage_start or temp_end <= temp_start:
        raise ValueError("单循环对齐后区间无效，请检查阈值/温度峰值。")

    ch1_seg = ch1_raw[voltage_start:voltage_end]
    ch3_seg = ch3_raw[voltage_start:voltage_end]
    ch5_seg = ch5_raw[voltage_start:voltage_end]
    temp_seg = temp_data[temp_start:temp_end]
    time_seg = temp_time[temp_start:temp_end]

    common_len = int(min(ch1_seg.size, ch3_seg.size, ch5_seg.size, temp_seg.size))
    if common_len <= 0:
        raise ValueError("单循环对齐后长度为 0。")

    ch1_seg = ch1_seg[:common_len]
    ch3_seg = ch3_seg[:common_len]
    ch5_seg = ch5_seg[:common_len]
    temp_seg = temp_seg[:common_len]
    time_seg = time_seg[:common_len]

    temp_peak_local = int(temp_peak_idx - temp_start)
    result = GroupAlignedResult(
        device_id=device_id,
        group_id=zero_group_input.group_id,
        ch1_file=zero_group_input.ch1_file.name,
        ch3_file=zero_group_input.ch3_file.name,
        ch5_file=zero_group_input.ch5_file.name,
        ch1_raw_len=int(ch1_raw.size),
        ch3_raw_len=int(ch3_raw.size),
        ch5_raw_len=int(ch5_raw.size),
        voltage_start_idx=int(voltage_start),
        voltage_end_idx=int(voltage_start + common_len),
        voltage_trim_len=int(voltage_end - voltage_start),
        falling_edge_count_all=int(edges_all.size),
        falling_edges_all=[int(x) for x in edges_all.tolist()],
        falling_edges_used=[int(edge_idx)],
        temp_peak_count_all=1,
        temp_peaks_used_global=[int(temp_peak_idx)],
        temp_peak_idx_global=int(temp_peak_idx),
        temp_peak_idx_in_segment=temp_peak_local,
        temp_start_idx=int(temp_start),
        temp_end_idx=int(temp_start + common_len),
        aligned_len=common_len,
        used_cycles=1,
        is_partial_group=False,
        alignment_mode="special_zero_group",
        ch1_segment=ch1_seg,
        ch3_segment=ch3_seg,
        ch5_segment=ch5_seg,
        temperature_segment=temp_seg,
        temperature_time_segment=time_seg,
        voltage_raw_signal=ch1_raw,
    )
    return result, temp_signal_name


def _device_prefix(device_id: str) -> str:
    """生成文件名前缀，便于多器件输出区分。"""
    return f"device_{device_id}"


def save_outputs(
    output_dir: Path,
    results: list[GroupAlignedResult],
    cfg: AlignConfig,
    temp_signal_name: str,
    device_group_count_found: int,
) -> dict[str, Path]:
    """
    写出结构化结果文件（CSV / NPZ / MAT / JSON）。

    文件用途：
    - alignment_summary.csv：调参与排错首选
    - group_XX_aligned.csv：单组明细，便于 Excel/Origin 快速核验
    - aligned_power_cycle.npz：Python 训练/分析方便
    - aligned_power_cycle.mat：MATLAB 侧继续处理方便
    - alignment_meta.json：记录本次参数，便于复现实验
    """
    _, savemat = _import_scipy_io()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_id = results[0].device_id
    prefix = _device_prefix(device_id)
    processed_count = len(results)

    # 1) summary.csv：便于人工核查索引和长度
    summary_file = output_dir / f"{prefix}_alignment_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "device_id",
                "sample_rate_hz",
                "device_group_count_found",
                "device_group_count_processed",
                "group_id",
                "ch1_file",
                "ch3_file",
                "ch5_file",
                "ch1_raw_len",
                "ch3_raw_len",
                "ch5_raw_len",
                "voltage_start_idx",
                "voltage_end_idx",
                "voltage_trim_len",
                "falling_edge_count_all",
                "falling_edges_all",
                "falling_edges_used",
                "temp_peak_count_all",
                "temp_peaks_used_global",
                "temp_peak_idx_global",
                "temp_peak_idx_in_segment",
                "temp_start_idx",
                "temp_end_idx",
                "aligned_len",
                "used_cycles",
                "is_partial_group",
                "alignment_mode",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "device_id": item.device_id,
                    "sample_rate_hz": cfg.sample_rate_hz,
                    "device_group_count_found": device_group_count_found,
                    "device_group_count_processed": processed_count,
                    "group_id": item.group_id,
                    "ch1_file": item.ch1_file,
                    "ch3_file": item.ch3_file,
                    "ch5_file": item.ch5_file,
                    "ch1_raw_len": item.ch1_raw_len,
                    "ch3_raw_len": item.ch3_raw_len,
                    "ch5_raw_len": item.ch5_raw_len,
                    "voltage_start_idx": item.voltage_start_idx,
                    "voltage_end_idx": item.voltage_end_idx,
                    "voltage_trim_len": item.voltage_trim_len,
                    "falling_edge_count_all": item.falling_edge_count_all,
                    "falling_edges_all": json.dumps(item.falling_edges_all, ensure_ascii=False),
                    "falling_edges_used": json.dumps(item.falling_edges_used, ensure_ascii=False),
                    "temp_peak_count_all": item.temp_peak_count_all,
                    "temp_peaks_used_global": json.dumps(item.temp_peaks_used_global, ensure_ascii=False),
                    "temp_peak_idx_global": item.temp_peak_idx_global,
                    "temp_peak_idx_in_segment": item.temp_peak_idx_in_segment,
                    "temp_start_idx": item.temp_start_idx,
                    "temp_end_idx": item.temp_end_idx,
                    "aligned_len": item.aligned_len,
                    "used_cycles": item.used_cycles,
                    "is_partial_group": int(item.is_partial_group),
                    "alignment_mode": item.alignment_mode,
                }
            )

    # 2) 每组输出一个 CSV（time, temperature, ch1, ch3, ch5）
    for item in results:
        group_csv = output_dir / f"{prefix}_group_{item.group_id:02d}_aligned.csv"
        stacked = np.column_stack(
            [
                item.temperature_time_segment.reshape(-1),
                item.temperature_segment.reshape(-1),
                item.ch1_segment.reshape(-1),
                item.ch3_segment.reshape(-1),
                item.ch5_segment.reshape(-1),
            ]
        )
        np.savetxt(group_csv, stacked, delimiter=",", header="time,temperature,ch1,ch3,ch5", comments="")

    # 3) NPZ 汇总（Python 侧读取更方便）
    npz_file = output_dir / f"{prefix}_aligned_power_cycle.npz"
    np.savez(
        npz_file,
        device_id=np.asarray([results[0].device_id if results else ""], dtype=object),
        group_ids=np.asarray([x.group_id for x in results], dtype=np.int32),
        ch1_segments=np.asarray([x.ch1_segment for x in results], dtype=object),
        ch3_segments=np.asarray([x.ch3_segment for x in results], dtype=object),
        ch5_segments=np.asarray([x.ch5_segment for x in results], dtype=object),
        temperature_segments=np.asarray([x.temperature_segment for x in results], dtype=object),
        temperature_time_segments=np.asarray([x.temperature_time_segment for x in results], dtype=object),
        aligned_lengths=np.asarray([x.aligned_len for x in results], dtype=np.int32),
        used_cycles=np.asarray([x.used_cycles for x in results], dtype=np.int32),
        is_partial_group=np.asarray([x.is_partial_group for x in results], dtype=bool),
        alignment_mode=np.asarray([x.alignment_mode for x in results], dtype=object),
    )

    # 4) MAT 汇总（MATLAB 侧读取）
    mat_file = output_dir / f"{prefix}_aligned_power_cycle.mat"
    n = len(results)
    ch1_cells = np.empty((1, n), dtype=object)
    ch3_cells = np.empty((1, n), dtype=object)
    ch5_cells = np.empty((1, n), dtype=object)
    temperature_cells = np.empty((1, n), dtype=object)
    time_cells = np.empty((1, n), dtype=object)
    for i, item in enumerate(results):
        ch1_cells[0, i] = item.ch1_segment.reshape(-1, 1)
        ch3_cells[0, i] = item.ch3_segment.reshape(-1, 1)
        ch5_cells[0, i] = item.ch5_segment.reshape(-1, 1)
        temperature_cells[0, i] = item.temperature_segment.reshape(-1, 1)
        time_cells[0, i] = item.temperature_time_segment.reshape(-1, 1)

    savemat(
        mat_file,
        {
            "device_id": np.array([results[0].device_id if results else ""], dtype=object),
            "group_ids": np.asarray([x.group_id for x in results], dtype=np.int32).reshape(1, -1),
            "aligned_lengths": np.asarray([x.aligned_len for x in results], dtype=np.int32).reshape(1, -1),
            "used_cycles": np.asarray([x.used_cycles for x in results], dtype=np.int32).reshape(1, -1),
            "is_partial_group": np.asarray([x.is_partial_group for x in results], dtype=np.uint8).reshape(1, -1),
            "alignment_mode": np.asarray([x.alignment_mode for x in results], dtype=object).reshape(1, -1),
            "ch1_segments": ch1_cells,
            "ch3_segments": ch3_cells,
            "ch5_segments": ch5_cells,
            "temperature_segments": temperature_cells,
            "temperature_time_segments": time_cells,
            "high_level_threshold": np.asarray([[cfg.high_level_threshold]], dtype=float),
            "sample_rate_hz": np.asarray([[cfg.sample_rate_hz]], dtype=float),
        },
    )

    # 5) JSON 元信息
    meta_file = output_dir / f"{prefix}_alignment_meta.json"
    meta = {
        "config": asdict(cfg),
        "temp_signal_name": temp_signal_name,
        "device_group_count_found": device_group_count_found,
        "device_group_count_processed": processed_count,
        "group_count": len(results),
        "output_dir": str(output_dir.resolve()),
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {
        "summary_csv": summary_file,
        "npz": npz_file,
        "mat": mat_file,
        "meta_json": meta_file,
    }


def save_visualizations(output_dir: Path, results: list[GroupAlignedResult], cfg: AlignConfig) -> tuple[Path, Path]:
    """
    生成可视化检查图。

    每组输出一张双子图：
    - 上图：原始电压 + 边沿检测结果 + 实际切片边界
    - 下图：对齐后的温度/电压叠加（双坐标）
    """
    plt = _import_matplotlib_pyplot()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = _device_prefix(results[0].device_id)

    for item in results:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

        # 图1：原始电压 + 边沿检测 + 截取边界
        ax_raw = axes[0]
        x_raw = np.arange(item.voltage_raw_signal.size, dtype=int)
        ax_raw.plot(x_raw, item.voltage_raw_signal, color="#3b3b3b", linewidth=1.0, label="Voltage (raw)")
        ax_raw.axhline(
            cfg.high_level_threshold,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.0,
            label=f"Threshold={cfg.high_level_threshold:.3f}V",
        )
        if item.falling_edges_all:
            idx_all = np.asarray(item.falling_edges_all, dtype=int)
            ax_raw.scatter(
                idx_all,
                item.voltage_raw_signal[idx_all],
                s=12,
                color="#ffb347",
                alpha=0.75,
                label="Falling edges (all)",
                zorder=3,
            )
        if item.falling_edges_used:
            idx_used = np.asarray(item.falling_edges_used, dtype=int)
            ax_raw.scatter(
                idx_used,
                item.voltage_raw_signal[idx_used],
                s=20,
                color="#d62728",
                alpha=0.95,
                label="Falling edges (used)",
                zorder=4,
            )
        ax_raw.axvline(item.voltage_start_idx, color="#2ca02c", linewidth=1.0, linestyle="--", label="Voltage start")
        ax_raw.axvline(item.voltage_end_idx, color="#9467bd", linewidth=1.0, linestyle="--", label="Voltage end")
        ax_raw.set_title(f"Group {item.group_id:02d}: Raw Voltage / Edge Detection")
        ax_raw.set_xlabel("Sample index")
        ax_raw.set_ylabel("Voltage (V)")
        ax_raw.grid(alpha=0.25, linewidth=0.5)
        ax_raw.legend(loc="best", fontsize=8)

        # 图2：对齐后的电压与温度（双坐标）
        ax_align_v = axes[1]
        t_rel = np.arange(item.aligned_len, dtype=float) / float(cfg.sample_rate_hz)
        ax_align_v.plot(t_rel, item.ch1_segment, color="#1f77b4", linewidth=1.2, label="CH1 (aligned)")
        ax_align_v.set_xlabel("Time (s)")
        ax_align_v.set_ylabel("CH1 (V)", color="#1f77b4")
        ax_align_v.tick_params(axis="y", labelcolor="#1f77b4")
        ax_align_v.grid(alpha=0.25, linewidth=0.5)

        ax_align_t = ax_align_v.twinx()
        ax_align_t.plot(t_rel, item.temperature_segment, color="#d62728", linewidth=1.2, label="Temperature (aligned)")
        ax_align_t.set_ylabel("Temperature", color="#d62728")
        ax_align_t.tick_params(axis="y", labelcolor="#d62728")

        if 0 <= item.temp_peak_idx_in_segment < item.aligned_len:
            p = item.temp_peak_idx_in_segment
            ax_align_t.scatter(
                [t_rel[p]],
                [item.temperature_segment[p]],
                color="#d62728",
                s=30,
                zorder=5,
                label="First temp peak",
            )

        # 合并两个坐标轴图例
        handles_v, labels_v = ax_align_v.get_legend_handles_labels()
        handles_t, labels_t = ax_align_t.get_legend_handles_labels()
        ax_align_v.legend(handles_v + handles_t, labels_v + labels_t, loc="best", fontsize=8)
        ax_align_v.set_title(
            f"Group {item.group_id:02d}: Aligned Signals (len={item.aligned_len}, fs={cfg.sample_rate_hz:.1f}Hz)"
        )

        fig.savefig(plot_dir / f"{prefix}_group_{item.group_id:02d}_check.png", dpi=170)
        plt.close(fig)

    # 组长度总览图
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    group_ids = [x.group_id for x in results]
    lengths = [x.aligned_len for x in results]
    ax.bar(group_ids, lengths, color="#4c78a8", alpha=0.9)
    ax.set_title("Aligned Length per Group")
    ax.set_xlabel("Group ID")
    ax.set_ylabel("Aligned length (samples)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    length_plot = plot_dir / f"{prefix}_aligned_length_summary.png"
    fig.savefig(length_plot, dpi=170)
    plt.close(fig)

    return plot_dir, length_plot


def parse_args() -> argparse.Namespace:
    """命令行参数定义。"""
    parser = argparse.ArgumentParser(description="功率循环温度-电压数据对齐脚本")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="输入根目录（包含“12号/PC、12号/温度”）")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="输出根目录（按器件分别写入）")
    parser.add_argument(
        "--device-ids",
        default=",".join(DEFAULT_DEVICE_IDS),
        help='待处理器件编号，逗号分隔，如 "12,13,21"',
    )

    # 电压边沿相关参数（你后续最常改）
    parser.add_argument("--high-threshold", type=float, default=2.0, help="高电平阈值（V），默认 2.0")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="采样率（Hz），默认 10")
    parser.add_argument("--cycles-per-group", type=int, default=10, help="每组包含周期数，默认 10")
    parser.add_argument(
        "--allow-partial-last-group",
        dest="allow_partial_last_group",
        action="store_true",
        help="允许最后一组不足 cycles-per-group 时按部分循环对齐（默认开启）",
    )
    parser.add_argument(
        "--no-allow-partial-last-group",
        dest="allow_partial_last_group",
        action="store_false",
        help="禁用最后一组部分循环对齐",
    )
    parser.set_defaults(allow_partial_last_group=True)
    parser.add_argument(
        "--allow-partial-any-group",
        action="store_true",
        help="允许任意组（不只最后一组）在边沿不足时按部分循环对齐",
    )
    parser.add_argument(
        "--partial-min-cycles",
        type=int,
        default=DEFAULT_PARTIAL_MIN_CYCLES,
        help="启用部分循环时允许的最少循环数（默认 3，可设为 1）",
    )
    parser.add_argument("--pre-points", type=int, default=155, help="边沿前保留点数，默认 155")
    parser.add_argument("--post-points", type=int, default=245, help="边沿后保留点数，默认 245")
    parser.add_argument(
        "--edge-min-gap-sec",
        type=float,
        default=20.0,
        help="边沿最小间隔（秒），用于抑制抖动误检，默认 20s",
    )

    # 温度信号读取参数（参考 tempture.py）
    parser.add_argument("--temp-mode", choices=["index", "name"], default="name", help="按索引或名称读取温度信号")
    parser.add_argument("--temp-index", type=int, default=9, help="temp-mode=index 时生效（MATLAB 1-based）")
    parser.add_argument("--temp-name", default="tempture", help="temp-mode=name 时生效")
    parser.add_argument("--list-temp-signals", action="store_true", help="仅列出温度文件中的信号并退出")

    # 温度峰值搜索参数
    parser.add_argument("--peak-smooth-window", type=int, default=7, help="峰值搜索前平滑窗口（点）")
    parser.add_argument(
        "--peak-prom-factor",
        type=float,
        default=0.10,
        help="峰值显著性系数（乘以温度序列标准差）",
    )
    parser.add_argument(
        "--peak-min-gap-sec",
        type=float,
        default=20.0,
        help="温度峰值最小间隔（秒），用于剔除噪声峰",
    )
    parser.add_argument(
        "--allow-single-cycle-temp-peak-fallback",
        dest="allow_single_cycle_temp_peak_fallback",
        action="store_true",
        help="单循环部分组在温度无局部峰时，允许回退到温度最大值对齐（默认开启）",
    )
    parser.add_argument(
        "--no-allow-single-cycle-temp-peak-fallback",
        dest="allow_single_cycle_temp_peak_fallback",
        action="store_false",
        help="禁用单循环部分组温度峰值回退",
    )
    parser.set_defaults(allow_single_cycle_temp_peak_fallback=True)

    parser.add_argument("--save-plots", dest="save_plots", action="store_true", help="保存可视化检查图（默认开启）")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", help="不保存可视化检查图")
    parser.set_defaults(save_plots=True)
    parser.add_argument(
        "--process-zero-group",
        dest="process_zero_group",
        action="store_true",
        help="处理 _0_ 温度文件对应的单循环独立组（默认开启）",
    )
    parser.add_argument(
        "--skip-zero-group",
        dest="process_zero_group",
        action="store_false",
        help="跳过 _0_ 单循环独立组处理",
    )
    parser.set_defaults(process_zero_group=True)

    parser.add_argument("--strict", action="store_true", help="严格模式：任一组异常即终止（默认跳过异常组）")
    parser.add_argument("--max-groups", type=int, default=0, help="最多处理多少组（0 表示全部）")
    return parser.parse_args()


def main() -> None:
    """
    脚本主入口。

    执行顺序：
    1) 解析参数与文件列表
    2) 读取温度总序列
    3) 逐组对齐
    4) 输出结果与检查图
    """
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    device_ids = parse_device_ids(args.device_ids)

    cfg = AlignConfig(
        sample_rate_hz=args.sample_rate,
        high_level_threshold=args.high_threshold,
        cycles_per_group=args.cycles_per_group,
        pre_fall_points=args.pre_points,
        post_fall_points=args.post_points,
        edge_min_gap_seconds=args.edge_min_gap_sec,
        peak_smooth_window=args.peak_smooth_window,
        peak_min_prominence_factor=args.peak_prom_factor,
        peak_min_gap_seconds=args.peak_min_gap_sec,
        temp_mode=args.temp_mode,
        temp_signal_index=args.temp_index,
        temp_signal_name=args.temp_name,
        max_groups=args.max_groups,
        strict_mode=args.strict,
        allow_partial_last_group=args.allow_partial_last_group,
        allow_partial_any_group=args.allow_partial_any_group,
        partial_min_cycles=args.partial_min_cycles,
        allow_single_cycle_temp_peak_fallback=args.allow_single_cycle_temp_peak_fallback,
    )
    print(
        "[CFG] "
        f"partial_min_cycles={cfg.partial_min_cycles}, "
        f"allow_partial_last_group={cfg.allow_partial_last_group}, "
        f"allow_partial_any_group={cfg.allow_partial_any_group}"
    )

    if args.list_temp_signals:
        for device_id in device_ids:
            device_dir = discover_device_dir(input_root, device_id)
            temp_dir = device_dir / "温度"
            regular_temp_files, zero_temp_files = discover_temperature_files(temp_dir, device_id)
            temp_files = regular_temp_files + zero_temp_files
            if not temp_files:
                print(f"[WARN] 器件 {device_id}: 温度目录无可用文件 ({temp_dir})")
                continue
            names = list_temperature_signals(temp_files[0])
            print(f"器件 {device_id} 温度信号列表（文件: {temp_files[0].name}）:")
            for i, name in enumerate(names, start=1):
                print(f"  {i}: {name}")
        return

    for device_id in device_ids:
        print(f"\n===== 器件 {device_id} =====")
        device_dir = discover_device_dir(input_root, device_id)
        pc_dir = device_dir / "PC"
        temp_dir = device_dir / "温度"

        if not pc_dir.is_dir():
            raise FileNotFoundError(f"器件 {device_id} 的 PC 目录不存在: {pc_dir}")
        if not temp_dir.is_dir():
            raise FileNotFoundError(f"器件 {device_id} 的温度目录不存在: {temp_dir}")

        all_group_inputs = discover_pc_group_inputs(pc_dir, device_id)
        zero_group_input = next((x for x in all_group_inputs if x.group_id == 0), None)
        regular_group_inputs = [x for x in all_group_inputs if x.group_id != 0]

        if cfg.max_groups > 0:
            regular_group_inputs = regular_group_inputs[: cfg.max_groups]

        if not all_group_inputs:
            raise FileNotFoundError(f"器件 {device_id} 未找到完整组数据（需同时存在 ch1/ch3/ch5）。")

        print(f"[1/4] 发现完整电压组(含group0): {len(all_group_inputs)}")
        print(f"  常规组数(排除group0): {len(regular_group_inputs)}")
        print(f"  是否存在group0单循环组: {'是' if zero_group_input is not None else '否'}")
        for row in regular_group_inputs:
            print(
                f"  group={row.group_id:02d} "
                f"ch1={row.ch1_file.name} ch3={row.ch3_file.name} ch5={row.ch5_file.name}"
            )
        if zero_group_input is not None:
            print(
                f"  group=00 ch1={zero_group_input.ch1_file.name} "
                f"ch3={zero_group_input.ch3_file.name} ch5={zero_group_input.ch5_file.name}"
            )

        regular_temp_files, zero_temp_files = discover_temperature_files(temp_dir, device_id)
        if not regular_temp_files and not zero_temp_files:
            raise FileNotFoundError(f"器件 {device_id} 温度目录下未找到匹配文件: {temp_dir}")

        print("[2/4] 读取温度文件列表 ...")
        if regular_temp_files:
            print("  常规温度文件: " + ", ".join(p.name for p in regular_temp_files))
        else:
            print("  常规温度文件: 无")
        if zero_temp_files:
            print("  _0_单循环温度文件: " + ", ".join(p.name for p in zero_temp_files))
        else:
            print("  _0_单循环温度文件: 无")

        print("[3/4] 开始对齐 ...")
        results: list[GroupAlignedResult] = []
        used_temp_signal_names: list[str] = []

        # 3.1 常规组：只用 regular_temp_files，确保主流程跳过 _0_ 文件
        if regular_group_inputs:
            if not regular_temp_files:
                raise FileNotFoundError(
                    f"器件 {device_id} 存在常规电压组，但未找到常规温度区间文件（如 *_1-10_*）。"
                )
            temp_time, temp_data, temp_signal_name_regular = load_temperature_series_from_files(regular_temp_files, cfg)
            print(f"  常规温度信号: {temp_signal_name_regular}，长度: {temp_data.size}")
            regular_results = align_groups(
                device_id=device_id,
                group_inputs=regular_group_inputs,
                temperature_time=temp_time,
                temperature_data=temp_data,
                cfg=cfg,
            )
            results.extend(regular_results)
            used_temp_signal_names.append(temp_signal_name_regular)

        # 3.2 group0 单循环独立处理：锚点对齐（电压边沿 vs 温度最高点）
        if args.process_zero_group:
            if zero_group_input is not None and zero_temp_files:
                if len(zero_temp_files) > 1:
                    print(
                        "[WARN] 检测到多个 _0_ 温度文件，当前仅使用第一个: "
                        f"{zero_temp_files[0].name}"
                    )
                zero_result, temp_signal_name_zero = align_zero_group(
                    device_id=device_id,
                    zero_group_input=zero_group_input,
                    zero_temp_file=zero_temp_files[0],
                    cfg=cfg,
                )
                results.append(zero_result)
                used_temp_signal_names.append(temp_signal_name_zero)
            elif zero_group_input is not None and not zero_temp_files:
                print("[WARN] 存在 group0 电压组，但未找到对应 _0_ 温度文件，已跳过 group0。")
            elif zero_group_input is None and zero_temp_files:
                print("[WARN] 找到 _0_ 温度文件，但未找到 group0 电压组，已跳过 _0_ 文件。")
        else:
            if zero_group_input is not None or zero_temp_files:
                print("[INFO] 已按参数 --skip-zero-group 跳过 _0_ 单循环组处理。")

        if not results:
            raise RuntimeError(f"器件 {device_id} 没有可用对齐结果（全部组被跳过）。")

        # 统一按组号排序，确保输出与可视化按组递增
        results = sorted(results, key=lambda x: int(x.group_id))
        temp_signal_name = " / ".join(sorted(set(used_temp_signal_names))) if used_temp_signal_names else "temperature"
        device_group_count_found = len(regular_group_inputs) + (1 if zero_group_input is not None else 0)

        output_dir = (output_root / f"{device_id}号").resolve()
        print("[4/4] 写出结果文件 ...")
        output_files = save_outputs(
            output_dir=output_dir,
            results=results,
            cfg=cfg,
            temp_signal_name=temp_signal_name,
            device_group_count_found=device_group_count_found,
        )
        if args.save_plots:
            plot_dir, length_plot = save_visualizations(output_dir=output_dir, results=results, cfg=cfg)
            print(f"  可视化检查图目录: {plot_dir}")
            print(f"  组长度统计图: {length_plot.name}")
        print(f"  Summary: {output_files['summary_csv'].name}")
        print(f"  MAT: {output_files['mat'].name}")
        print(f"  NPZ: {output_files['npz'].name}")
        print(f"  Meta: {output_files['meta_json'].name}")

        print(f"- 器件 {device_id} 输出目录: {output_dir}")
        print(f"- 已对齐组数: {len(results)}")
        for item in results:
            print(
                f"  group={item.group_id:02d}, aligned_len={item.aligned_len}, "
                f"mode={item.alignment_mode}, used_cycles={item.used_cycles}, "
                f"voltage_idx=[{item.voltage_start_idx},{item.voltage_end_idx}), "
                f"temp_idx=[{item.temp_start_idx},{item.temp_end_idx})"
            )


if __name__ == "__main__":
    main()
