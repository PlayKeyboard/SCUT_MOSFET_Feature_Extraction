#!/usr/bin/env python3
"""
对齐 MOSFET 功率循环中的电压与温度数据。

数据约定（基于当前仓库 data 目录）：
1) 10 个电压文件（每个文件是一组测试）：
   例如：20251115_PC_1_12_ch1.mat ... 20251115_PC_10_12_ch1.mat
2) 1 个温度总文件（连续记录 1~10 组）：
   例如：251108_1-10_12.mat（Simulink.SimulationData）

核心处理逻辑：
1) 在每组电压中，基于高低电平阈值找到“高 -> 低”的边沿。
2) 以第一和第十个边沿为锚点，使用可调 pre/post 点数截取该组有效段，去除电路切换段。
3) 在剩余温度序列里寻找第一个峰值，以同样 pre 点数回退作为起点，截取与电压同长度的数据段。
4) 对每组做等长裁剪，保证一组内的电压与温度长度一致。

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


VOLTAGE_FILE_RE = re.compile(r"^\d{8}_PC_(\d+)_\d+_ch1\.mat$", re.IGNORECASE)


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
    group_id: int
    voltage_file: str
    voltage_raw_len: int
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
    voltage_segment: np.ndarray
    temperature_segment: np.ndarray
    temperature_time_segment: np.ndarray
    voltage_raw_signal: np.ndarray


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


def discover_voltage_files(data_dir: Path) -> list[tuple[int, Path]]:
    """扫描 data 目录并按组号排序电压文件。返回 [(group_id, path), ...]。"""
    files: list[tuple[int, Path]] = []
    for path in data_dir.glob("*.mat"):
        match = VOLTAGE_FILE_RE.match(path.name)
        if match:
            group_id = int(match.group(1))
            files.append((group_id, path))
    files.sort(key=lambda x: x[0])
    return files


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
    voltage_files: list[tuple[int, Path]],
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

    for group_id, voltage_file in voltage_files:
        # A) 电压：读原始波形 -> 找边沿 -> 取出当前组有效窗口
        voltage_raw = load_voltage_signal(voltage_file)
        edges_all = detect_falling_edges(
            voltage=voltage_raw,
            threshold=cfg.high_level_threshold,
            min_gap_points=cfg.edge_min_gap_points,
        )
        edges_used = select_group_edges(
            edges=edges_all,
            cycles_per_group=cfg.cycles_per_group,
            expected_cycle_points=cfg.expected_cycle_points,
        )

        voltage_start = max(0, int(edges_used[0]) - cfg.pre_fall_points)
        voltage_end = min(voltage_raw.size, int(edges_used[-1]) + cfg.post_fall_points)
        if voltage_end <= voltage_start:
            raise ValueError(f"组 {group_id} 的电压切片区间无效：[{voltage_start}, {voltage_end})")

        voltage_trim = voltage_raw[voltage_start:voltage_end]

        # B) 温度：在“尚未消费”的剩余序列中找首峰，并按同逻辑对齐截取
        if temp_cursor >= temperature_data.size:
            raise ValueError("温度数据已经耗尽，无法继续对齐后续电压组。")

        temp_remaining = temperature_data[temp_cursor:]
        temp_peaks_rel = detect_temperature_peaks(temp_remaining, cfg)
        temp_peaks_used_rel = select_group_edges(
            edges=temp_peaks_rel,
            cycles_per_group=cfg.cycles_per_group,
            expected_cycle_points=cfg.expected_cycle_points,
        )
        temp_peak_rel = int(temp_peaks_used_rel[0])
        temp_peak_global = temp_cursor + temp_peak_rel
        temp_start = max(temp_cursor, temp_peak_global - cfg.pre_fall_points)
        temp_end = temp_start + voltage_trim.size

        if temp_end > temperature_data.size:
            temp_end = temperature_data.size

        temp_segment = temperature_data[temp_start:temp_end]
        time_segment = temperature_time[temp_start:temp_end]

        # C) 组内等长裁剪（满足后续建模/存储需要）
        common_len = int(min(voltage_trim.size, temp_segment.size))
        if common_len <= 0:
            raise ValueError(f"组 {group_id} 对齐后长度为 0，请检查阈值与峰值检测参数。")

        voltage_aligned = voltage_trim[:common_len]
        temp_aligned = temp_segment[:common_len]
        time_aligned = time_segment[:common_len]
        temp_peak_local = int(temp_peak_global - temp_start)

        results.append(
            GroupAlignedResult(
                group_id=group_id,
                voltage_file=voltage_file.name,
                voltage_raw_len=int(voltage_raw.size),
                voltage_start_idx=voltage_start,
                voltage_end_idx=voltage_start + common_len,
                voltage_trim_len=int(voltage_trim.size),
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
                voltage_segment=voltage_aligned,
                temperature_segment=temp_aligned,
                temperature_time_segment=time_aligned,
                voltage_raw_signal=voltage_raw,
            )
        )

        # 删除已对齐区段，进入下一组
        temp_cursor = int(temp_start + common_len)

    return results


def save_outputs(output_dir: Path, results: list[GroupAlignedResult], cfg: AlignConfig, temp_signal_name: str) -> None:
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

    # 1) summary.csv：便于人工核查索引和长度
    summary_file = output_dir / "alignment_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "voltage_file",
                "voltage_raw_len",
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
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "group_id": item.group_id,
                    "voltage_file": item.voltage_file,
                    "voltage_raw_len": item.voltage_raw_len,
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
                }
            )

    # 2) 每组输出一个 CSV（time, temperature, voltage）
    for item in results:
        group_csv = output_dir / f"group_{item.group_id:02d}_aligned.csv"
        stacked = np.column_stack(
            [
                item.temperature_time_segment.reshape(-1),
                item.temperature_segment.reshape(-1),
                item.voltage_segment.reshape(-1),
            ]
        )
        np.savetxt(group_csv, stacked, delimiter=",", header="time,temperature,voltage", comments="")

    # 3) NPZ 汇总（Python 侧读取更方便）
    np.savez(
        output_dir / "aligned_power_cycle.npz",
        group_ids=np.asarray([x.group_id for x in results], dtype=np.int32),
        voltage_segments=np.asarray([x.voltage_segment for x in results], dtype=object),
        temperature_segments=np.asarray([x.temperature_segment for x in results], dtype=object),
        temperature_time_segments=np.asarray([x.temperature_time_segment for x in results], dtype=object),
        aligned_lengths=np.asarray([x.aligned_len for x in results], dtype=np.int32),
    )

    # 4) MAT 汇总（MATLAB 侧读取）
    n = len(results)
    voltage_cells = np.empty((1, n), dtype=object)
    temperature_cells = np.empty((1, n), dtype=object)
    time_cells = np.empty((1, n), dtype=object)
    for i, item in enumerate(results):
        voltage_cells[0, i] = item.voltage_segment.reshape(-1, 1)
        temperature_cells[0, i] = item.temperature_segment.reshape(-1, 1)
        time_cells[0, i] = item.temperature_time_segment.reshape(-1, 1)

    savemat(
        output_dir / "aligned_power_cycle.mat",
        {
            "group_ids": np.asarray([x.group_id for x in results], dtype=np.int32).reshape(1, -1),
            "aligned_lengths": np.asarray([x.aligned_len for x in results], dtype=np.int32).reshape(1, -1),
            "voltage_segments": voltage_cells,
            "temperature_segments": temperature_cells,
            "temperature_time_segments": time_cells,
            "high_level_threshold": np.asarray([[cfg.high_level_threshold]], dtype=float),
            "sample_rate_hz": np.asarray([[cfg.sample_rate_hz]], dtype=float),
        },
    )

    # 5) JSON 元信息
    meta = {
        "config": asdict(cfg),
        "temp_signal_name": temp_signal_name,
        "group_count": len(results),
        "output_dir": str(output_dir.resolve()),
    }
    with (output_dir / "alignment_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_visualizations(output_dir: Path, results: list[GroupAlignedResult], cfg: AlignConfig) -> Path:
    """
    生成可视化检查图。

    每组输出一张双子图：
    - 上图：原始电压 + 边沿检测结果 + 实际切片边界
    - 下图：对齐后的温度/电压叠加（双坐标）
    """
    plt = _import_matplotlib_pyplot()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

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
        ax_align_v.plot(t_rel, item.voltage_segment, color="#1f77b4", linewidth=1.2, label="Voltage (aligned)")
        ax_align_v.set_xlabel("Time (s)")
        ax_align_v.set_ylabel("Voltage (V)", color="#1f77b4")
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

        fig.savefig(plot_dir / f"group_{item.group_id:02d}_check.png", dpi=170)
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
    fig.savefig(plot_dir / "aligned_length_summary.png", dpi=170)
    plt.close(fig)

    return plot_dir


def parse_args() -> argparse.Namespace:
    """命令行参数定义。"""
    parser = argparse.ArgumentParser(description="功率循环温度-电压数据对齐脚本")
    parser.add_argument("--data-dir", default="data", help="包含 .mat 数据文件的目录")
    parser.add_argument(
        "--temp-file",
        default="251108_1-10_12.mat",
        help="温度总文件路径；相对路径时相对于 --data-dir 解析",
    )
    parser.add_argument("--output-dir", default="data/aligned_output", help="输出目录")

    # 电压边沿相关参数（你后续最常改）
    parser.add_argument("--high-threshold", type=float, default=2.0, help="高电平阈值（V），默认 2.0")
    parser.add_argument("--sample-rate", type=float, default=10.0, help="采样率（Hz），默认 10")
    parser.add_argument("--cycles-per-group", type=int, default=10, help="每组包含周期数，默认 10")
    parser.add_argument("--pre-points", type=int, default=155, help="边沿前保留点数，默认 155")
    parser.add_argument("--post-points", type=int, default=245, help="边沿后保留点数，默认 245")
    parser.add_argument(
        "--edge-min-gap-sec",
        type=float,
        default=20.0,
        help="边沿最小间隔（秒），用于抑制抖动误检，默认 20s",
    )

    # 温度信号读取参数（参考 tempture.py）
    parser.add_argument("--temp-mode", choices=["index", "name"], default="index", help="按索引或名称读取温度信号")
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

    parser.add_argument("--save-plots", dest="save_plots", action="store_true", help="保存可视化检查图（默认开启）")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", help="不保存可视化检查图")
    parser.set_defaults(save_plots=True)

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

    data_dir = Path(args.data_dir).resolve()
    temp_file = Path(args.temp_file)
    if not temp_file.is_absolute():
        temp_file = (data_dir / temp_file).resolve()

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
    )

    if args.list_temp_signals:
        names = list_temperature_signals(temp_file)
        print(f"温度文件信号列表 ({temp_file.name}):")
        for i, name in enumerate(names, start=1):
            print(f"  {i}: {name}")
        return

    voltage_files = discover_voltage_files(data_dir)
    if not voltage_files:
        raise FileNotFoundError(f"在目录 {data_dir} 下没有找到电压文件（*_PC_*_ch1.mat）。")
    if cfg.max_groups > 0:
        voltage_files = voltage_files[: cfg.max_groups]

    print(f"[1/4] 发现电压组文件: {len(voltage_files)}")
    for gid, path in voltage_files:
        print(f"  group={gid:02d} file={path.name}")

    print("[2/4] 读取温度总文件 ...")
    temp_time, temp_data, temp_signal_name = load_temperature_signal(temp_file, cfg)
    print(f"  温度信号: {temp_signal_name}")
    print(f"  温度长度: {temp_data.size}")

    print("[3/4] 开始逐组切除 + 对齐 ...")
    results = align_groups(
        voltage_files=voltage_files,
        temperature_time=temp_time,
        temperature_data=temp_data,
        cfg=cfg,
    )

    output_dir = Path(args.output_dir).resolve()
    print("[4/4] 写出结果文件 ...")
    save_outputs(output_dir=output_dir, results=results, cfg=cfg, temp_signal_name=temp_signal_name)
    if args.save_plots:
        plot_dir = save_visualizations(output_dir=output_dir, results=results, cfg=cfg)
        print(f"  可视化检查图目录: {plot_dir}")

    print("\n处理完成。")
    print(f"- 输出目录: {output_dir}")
    print(f"- 已对齐组数: {len(results)}")
    for item in results:
        print(
            f"  group={item.group_id:02d}, aligned_len={item.aligned_len}, "
            f"voltage_idx=[{item.voltage_start_idx},{item.voltage_end_idx}), "
            f"temp_idx=[{item.temp_start_idx},{item.temp_end_idx})"
        )


if __name__ == "__main__":
    main()
