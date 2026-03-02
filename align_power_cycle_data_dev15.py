#!/usr/bin/env python3
"""
15号器件专用温度-电压对齐脚本（独立实现）。

核心约束：
1) 不修改原始对齐脚本 align_power_cycle_data.py。
2) 温度读取只按 15号 文件结构索引读取，不依赖温度信号名称：
   - 数据路径：/#refs#/y
   - 可选名称：/#refs#/c（若不存在也不影响处理）
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
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks

try:
    import h5py
except Exception as _exc:  # pragma: no cover
    raise RuntimeError("需要 h5py 才能解析 15号温度文件，请先安装 h5py。") from _exc


def import_matplotlib_pyplot() -> Any:
    """延迟导入 matplotlib，并固定 Agg 后端用于无GUI环境输出图片。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("未找到 matplotlib，请先安装后再运行（pip install matplotlib）。") from exc
    return plt


DEFAULT_INPUT_ROOT = "data"
DEFAULT_OUTPUT_ROOT = "data/aligned_output_15"
DEFAULT_DEVICE_IDS = ["15"]

PC_FILE_RE = re.compile(r"^(\d{8})_PC_(\d+)_(\d+)_ch(\d+)\.mat$", re.IGNORECASE)
TEMP_FILE_RE = re.compile(r"^(\d{6,8})_([0-9]+(?:-[0-9]+)?)_(\d+)\.mat$", re.IGNORECASE)


@dataclass
class AlignConfig:
    sample_rate_hz: float = 10.0
    high_threshold: float = 2.0
    cycles_per_group: int = 10
    pre_points: int = 155
    post_points: int = 245
    edge_min_gap_sec: float = 20.0
    peak_smooth_window: int = 7
    peak_prom_factor: float = 0.10
    peak_min_gap_sec: float = 20.0
    allow_partial_last_group: bool = True
    partial_min_cycles: int = 3
    strict_mode: bool = False
    max_groups: int = 0
    process_zero_group: bool = True
    temp_index: int = 1

    @property
    def expected_cycle_points(self) -> int:
        return int(round((15.0 + 25.0) * self.sample_rate_hz))

    @property
    def edge_min_gap_points(self) -> int:
        return max(1, int(round(self.edge_min_gap_sec * self.sample_rate_hz)))

    @property
    def peak_min_gap_points(self) -> int:
        return max(1, int(round(self.peak_min_gap_sec * self.sample_rate_hz)))


@dataclass
class GroupInput:
    group_id: int
    ch1_file: Path
    ch3_file: Path
    ch5_file: Path


@dataclass
class GroupAlignedResult:
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


def decode_utf16(arr: np.ndarray) -> str:
    flat = np.asarray(arr).reshape(-1)
    return "".join(chr(int(x)) for x in flat if int(x) != 0)


def parse_group_token(token: str) -> tuple[int, int]:
    if "-" in token:
        a, b = token.split("-", 1)
        left, right = int(a), int(b)
    else:
        left = right = int(token)
    if left > right:
        left, right = right, left
    return left, right


def parse_device_ids(text: str) -> list[str]:
    ids = [x.strip() for x in text.split(",") if x.strip()]
    if not ids:
        raise ValueError("device_ids 为空，请至少指定一个器件编号。")
    return ids


def discover_device_dir(input_root: Path, device_id: str) -> Path:
    cands = [input_root / f"{device_id}号", input_root / device_id]
    for p in cands:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"未找到器件目录: {cands[0]} 或 {cands[1]}")


def discover_pc_groups(pc_dir: Path, device_id: str) -> list[GroupInput]:
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
        if ch not in (1, 3, 5):
            continue
        groups.setdefault(group_id, {})[ch] = path

    out: list[GroupInput] = []
    for gid in sorted(groups.keys()):
        row = groups[gid]
        if all(c in row for c in (1, 3, 5)):
            out.append(GroupInput(gid, row[1], row[3], row[5]))
    return out


def discover_temp_files(temp_dir: Path, device_id: str) -> tuple[list[Path], list[Path]]:
    regular: list[tuple[int, int, str, Path]] = []
    zero_only: list[tuple[int, int, str, Path]] = []
    for path in temp_dir.glob("*.mat"):
        m = TEMP_FILE_RE.match(path.name)
        if not m:
            continue
        date_text, range_text, dev_text = m.groups()
        if str(int(dev_text)) != str(int(device_id)):
            continue
        start, end = parse_group_token(range_text)
        row = (start, end, date_text, path)
        if start == 0 and end == 0:
            zero_only.append(row)
        else:
            regular.append(row)
    regular.sort(key=lambda x: (x[0], x[1], x[2], x[3].name))
    zero_only.sort(key=lambda x: (x[0], x[1], x[2], x[3].name))
    return [x[3] for x in regular], [x[3] for x in zero_only]


def _to_1d_float(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _collect_numeric(obj: Any, out: list[np.ndarray], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for name in obj.dtype.names:
                _collect_numeric(obj[name], out, depth + 1)
            return
        if obj.dtype == object:
            for _, item in np.ndenumerate(obj):
                _collect_numeric(item, out, depth + 1)
            return
        if np.issubdtype(obj.dtype, np.number):
            arr = obj.astype(float, copy=False).reshape(-1)
            if arr.size > 5:
                out.append(arr)
            return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_numeric(item, out, depth + 1)
        return
    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            if not key.startswith("_"):
                _collect_numeric(value, out, depth + 1)


def load_voltage_signal(path: Path) -> np.ndarray:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    vars_ = {k: v for k, v in data.items() if not k.startswith("__")}
    if "data" in vars_:
        arr = _to_1d_float(vars_["data"])
        if arr is not None:
            return arr
    cands: list[np.ndarray] = []
    for v in vars_.values():
        _collect_numeric(v, cands)
    if not cands:
        raise ValueError(f"无法从电压文件读取数值序列: {path}")
    cands.sort(key=lambda x: x.size, reverse=True)
    return cands[0]


def list_temp_signals_for_15(temp_file: Path) -> list[str]:
    with h5py.File(temp_file, "r") as f:
        if "/#refs#/c" in f:
            name = decode_utf16(np.asarray(f["/#refs#/c"][()]))
            if name:
                return [name]
    return ["index_1"]


def load_temp_signal_for_15(temp_file: Path, cfg: AlignConfig) -> tuple[np.ndarray, np.ndarray, str]:
    if cfg.temp_index != 1:
        raise ValueError(f"15号温度结构仅支持 temp_index=1，当前为 {cfg.temp_index}")
    with h5py.File(temp_file, "r") as f:
        if "/#refs#/y" not in f:
            raise RuntimeError(f"温度文件缺少 /#refs#/y: {temp_file}")
        y = np.asarray(f["/#refs#/y"][()], dtype=float).reshape(-1)
        if y.size == 0:
            raise RuntimeError(f"温度文件温度序列为空: {temp_file}")
        name = "index_1"
        if "/#refs#/c" in f:
            n = decode_utf16(np.asarray(f["/#refs#/c"][()]))
            if n:
                name = n
    t = np.arange(y.size, dtype=float) / float(cfg.sample_rate_hz)
    return t, y, name


def load_temp_series(temp_files: list[Path], cfg: AlignConfig) -> tuple[np.ndarray, np.ndarray, str]:
    if not temp_files:
        raise FileNotFoundError("温度文件列表为空。")
    y_all: list[np.ndarray] = []
    name = "index_1"
    total = 0
    for p in temp_files:
        _, y, n = load_temp_signal_for_15(p, cfg)
        y_all.append(y)
        total += y.size
        name = n
    y_cat = np.concatenate(y_all, axis=0)
    t_cat = np.arange(total, dtype=float) / float(cfg.sample_rate_hz)
    return t_cat, y_cat, name


def smooth(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def detect_falling_edges(voltage: np.ndarray, threshold: float, min_gap_points: int) -> np.ndarray:
    high = voltage > threshold
    edges = np.flatnonzero(high[:-1] & (~high[1:])) + 1
    if edges.size == 0:
        return edges
    keep = [int(edges[0])]
    for x in edges[1:]:
        if int(x) - keep[-1] >= min_gap_points:
            keep.append(int(x))
    return np.asarray(keep, dtype=int)


def choose_edge_window(edges: np.ndarray, target_count: int, expected_gap: int) -> np.ndarray:
    if edges.size < target_count:
        raise ValueError(f"边沿数量不足: {edges.size} < {target_count}")
    if edges.size == target_count:
        return edges
    low = 0.85 * expected_gap
    high = 1.15 * expected_gap
    for i in range(0, edges.size - target_count + 1):
        win = edges[i : i + target_count]
        gaps = np.diff(win)
        if gaps.size and np.all(gaps >= low) and np.all(gaps <= high):
            return win
    best = None
    best_score = None
    for i in range(0, edges.size - target_count + 1):
        win = edges[i : i + target_count]
        gaps = np.diff(win)
        if gaps.size == 0:
            score = 0.0
        else:
            score = float(np.mean(np.abs(gaps - expected_gap)) + np.std(gaps))
            if np.any(gaps < 0.5 * expected_gap) or np.any(gaps > 1.8 * expected_gap):
                score += float(expected_gap)
        if best_score is None or score < best_score:
            best_score = score
            best = win
    if best is None:
        raise ValueError("无法从边沿序列中选出有效窗口。")
    return best


def choose_temp_peak_window(
    peaks: np.ndarray, target_count: int, expected_gap: int, preferred_first_idx: int
) -> np.ndarray:
    """
    选择温度峰值窗口。

    与电压边沿不同，温度峰值在组起点附近更容易出现微小伪峰。
    这里在间隔评分之外，额外约束“首峰位置接近 preferred_first_idx”。
    """
    if peaks.size < target_count:
        raise ValueError(f"温度峰值数量不足: {peaks.size} < {target_count}")
    if peaks.size == target_count:
        return peaks

    min_first = max(1, int(round(0.6 * preferred_first_idx)))
    best = None
    best_score = None
    for i in range(0, peaks.size - target_count + 1):
        win = peaks[i : i + target_count]
        first_idx = int(win[0])
        if first_idx < min_first:
            # 过滤明显位于片段开头的伪峰窗口
            continue

        gaps = np.diff(win)
        if gaps.size == 0:
            gap_score = 0.0
        else:
            gap_score = float(np.mean(np.abs(gaps - expected_gap)) + np.std(gaps))
            if np.any(gaps < 0.5 * expected_gap) or np.any(gaps > 1.8 * expected_gap):
                gap_score += float(expected_gap)

        first_score = float(abs(first_idx - preferred_first_idx))
        score = gap_score + 0.35 * first_score

        if best_score is None or score < best_score:
            best_score = score
            best = win

    if best is None:
        # 若所有窗口都被首峰位置规则过滤，回退到通用策略
        return choose_edge_window(peaks, target_count, expected_gap)
    return best


def decide_cycles(detected_count: int, cfg: AlignConfig, is_last_group: bool) -> tuple[int, bool]:
    if detected_count >= cfg.cycles_per_group:
        return cfg.cycles_per_group, False
    if not (cfg.allow_partial_last_group and is_last_group):
        raise ValueError(f"边沿数不足: {detected_count} < {cfg.cycles_per_group}")
    if detected_count < max(1, cfg.partial_min_cycles):
        raise ValueError(
            f"边沿数 {detected_count} 低于 partial_min_cycles={cfg.partial_min_cycles}"
        )
    return detected_count, True


def detect_temp_peaks(signal: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    if signal.size < 3:
        return np.asarray([], dtype=int)
    s = smooth(signal, cfg.peak_smooth_window)
    prom = max(1e-6, cfg.peak_prom_factor * float(np.std(s)))
    peaks, _ = find_peaks(s, distance=cfg.peak_min_gap_points, prominence=prom)
    if peaks.size < cfg.cycles_per_group:
        peaks, _ = find_peaks(s, distance=cfg.peak_min_gap_points)
    if peaks.size == 0:
        peaks = np.flatnonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])) + 1
    return np.asarray(peaks, dtype=int)


def choose_zero_group_edge(voltage: np.ndarray, edges: np.ndarray, cfg: AlignConfig) -> int:
    if edges.size == 0:
        raise ValueError("group0 未检测到边沿。")
    pre_pts = max(5, int(round(3.0 * cfg.sample_rate_hz)))
    post_pts = max(5, int(round(3.0 * cfg.sample_rate_hz)))
    low_th = min(cfg.high_threshold * 0.5, 1.0)
    span = max(1e-6, float(np.max(voltage) - np.min(voltage)))
    best_edge = int(edges[0])
    best_score = -1e12
    for e in edges:
        idx = int(e)
        pre = voltage[max(0, idx - pre_pts) : idx]
        post = voltage[idx : min(voltage.size, idx + post_pts)]
        if pre.size < 3 or post.size < 3:
            continue
        high_ratio = float(np.mean(pre > cfg.high_threshold))
        low_ratio = float(np.mean(post < low_th))
        drop = float(np.mean(pre[-min(5, pre.size) :]) - np.mean(post[: min(5, post.size)]))
        score = 3.0 * high_ratio + 4.0 * low_ratio + 2.0 * (drop / span)
        if score > best_score:
            best_score = score
            best_edge = idx
    return best_edge


def align_regular_groups(
    device_id: str,
    groups: list[GroupInput],
    temp_time: np.ndarray,
    temp_data: np.ndarray,
    cfg: AlignConfig,
) -> list[GroupAlignedResult]:
    results: list[GroupAlignedResult] = []
    cursor = 0
    total = len(groups)
    for i, g in enumerate(groups):
        try:
            ch1 = load_voltage_signal(g.ch1_file)
            ch3 = load_voltage_signal(g.ch3_file)
            ch5 = load_voltage_signal(g.ch5_file)
            raw_min = int(min(ch1.size, ch3.size, ch5.size))
            ch1 = ch1[:raw_min]
            ch3 = ch3[:raw_min]
            ch5 = ch5[:raw_min]

            edges_all = detect_falling_edges(ch1, cfg.high_threshold, cfg.edge_min_gap_points)
            used_cycles, is_partial = decide_cycles(edges_all.size, cfg, i == total - 1)
            edges_used = choose_edge_window(edges_all, used_cycles, cfg.expected_cycle_points)

            v_start = max(0, int(edges_used[0]) - cfg.pre_points)
            v_end = min(ch1.size, int(edges_used[-1]) + cfg.post_points)
            if v_end <= v_start:
                raise ValueError(f"无效电压区间 [{v_start}, {v_end})")
            ch1_trim = ch1[v_start:v_end]
            ch3_trim = ch3[v_start:v_end]
            ch5_trim = ch5[v_start:v_end]

            if cursor >= temp_data.size:
                raise ValueError("温度数据耗尽。")
            remain = temp_data[cursor:]
            peaks_rel = detect_temp_peaks(remain, cfg)
            if peaks_rel.size == 0:
                raise ValueError("未检测到温度峰值。")
            peaks_used_rel = choose_temp_peak_window(
                peaks_rel,
                used_cycles,
                cfg.expected_cycle_points,
                preferred_first_idx=cfg.pre_points,
            )
            peak_global = int(cursor + peaks_used_rel[0])
            t_start = max(cursor, peak_global - cfg.pre_points)
            t_end = min(temp_data.size, t_start + ch1_trim.size)

            temp_seg = temp_data[t_start:t_end]
            time_seg = temp_time[t_start:t_end]
            common = int(min(ch1_trim.size, ch3_trim.size, ch5_trim.size, temp_seg.size))
            if common <= 0:
                raise ValueError("对齐后长度为 0。")

            temp_peak_local = int(peak_global - t_start)
            mode = "regular_full"
            if is_partial:
                mode = "regular_partial_last"

            results.append(
                GroupAlignedResult(
                    device_id=device_id,
                    group_id=g.group_id,
                    ch1_file=g.ch1_file.name,
                    ch3_file=g.ch3_file.name,
                    ch5_file=g.ch5_file.name,
                    ch1_raw_len=int(ch1.size),
                    ch3_raw_len=int(ch3.size),
                    ch5_raw_len=int(ch5.size),
                    voltage_start_idx=v_start,
                    voltage_end_idx=v_start + common,
                    voltage_trim_len=int(ch1_trim.size),
                    falling_edge_count_all=int(edges_all.size),
                    falling_edges_all=[int(x) for x in edges_all.tolist()],
                    falling_edges_used=[int(x) for x in edges_used.tolist()],
                    temp_peak_count_all=int(peaks_rel.size),
                    temp_peaks_used_global=[int(cursor + x) for x in peaks_used_rel.tolist()],
                    temp_peak_idx_global=peak_global,
                    temp_peak_idx_in_segment=temp_peak_local,
                    temp_start_idx=int(t_start),
                    temp_end_idx=int(t_start + common),
                    aligned_len=common,
                    used_cycles=used_cycles,
                    is_partial_group=is_partial,
                    alignment_mode=mode,
                    ch1_segment=ch1_trim[:common],
                    ch3_segment=ch3_trim[:common],
                    ch5_segment=ch5_trim[:common],
                    temperature_segment=temp_seg[:common],
                    temperature_time_segment=time_seg[:common],
                    voltage_raw_signal=ch1,
                )
            )
            cursor = int(t_start + common)
        except Exception as exc:
            if cfg.strict_mode:
                raise
            print(f"[WARN] 器件 {device_id} group={g.group_id:02d} 跳过: {exc}")
    return results


def align_zero_group(
    device_id: str, group0: GroupInput, zero_temp_file: Path, cfg: AlignConfig
) -> tuple[GroupAlignedResult, str]:
    ch1 = load_voltage_signal(group0.ch1_file)
    ch3 = load_voltage_signal(group0.ch3_file)
    ch5 = load_voltage_signal(group0.ch5_file)
    raw_min = int(min(ch1.size, ch3.size, ch5.size))
    ch1 = ch1[:raw_min]
    ch3 = ch3[:raw_min]
    ch5 = ch5[:raw_min]

    edges_all = detect_falling_edges(ch1, cfg.high_threshold, max(1, int(round(cfg.sample_rate_hz))))
    edge_idx = choose_zero_group_edge(ch1, edges_all, cfg)

    temp_time, temp_data, temp_name = load_temp_signal_for_15(zero_temp_file, cfg)
    temp_s = smooth(temp_data, cfg.peak_smooth_window)
    temp_peak = int(np.argmax(temp_s))

    v_start = 0
    v_end = int(raw_min)
    target_len = v_end - v_start
    t_start = int(temp_peak - edge_idx)
    t_end = int(t_start + target_len)
    if t_start < 0:
        head = -t_start
        v_start += head
        t_start = 0
        t_end = t_start + (v_end - v_start)
    if t_end > temp_data.size:
        tail = t_end - temp_data.size
        v_end -= tail
        t_end = int(temp_data.size)
    if v_end <= v_start or t_end <= t_start:
        raise ValueError("group0 对齐后区间无效。")

    ch1_seg = ch1[v_start:v_end]
    ch3_seg = ch3[v_start:v_end]
    ch5_seg = ch5[v_start:v_end]
    temp_seg = temp_data[t_start:t_end]
    time_seg = temp_time[t_start:t_end]
    common = int(min(ch1_seg.size, ch3_seg.size, ch5_seg.size, temp_seg.size))
    if common <= 0:
        raise ValueError("group0 对齐后长度为 0。")

    result = GroupAlignedResult(
        device_id=device_id,
        group_id=group0.group_id,
        ch1_file=group0.ch1_file.name,
        ch3_file=group0.ch3_file.name,
        ch5_file=group0.ch5_file.name,
        ch1_raw_len=int(ch1.size),
        ch3_raw_len=int(ch3.size),
        ch5_raw_len=int(ch5.size),
        voltage_start_idx=int(v_start),
        voltage_end_idx=int(v_start + common),
        voltage_trim_len=int(v_end - v_start),
        falling_edge_count_all=int(edges_all.size),
        falling_edges_all=[int(x) for x in edges_all.tolist()],
        falling_edges_used=[int(edge_idx)],
        temp_peak_count_all=1,
        temp_peaks_used_global=[int(temp_peak)],
        temp_peak_idx_global=int(temp_peak),
        temp_peak_idx_in_segment=int(temp_peak - t_start),
        temp_start_idx=int(t_start),
        temp_end_idx=int(t_start + common),
        aligned_len=common,
        used_cycles=1,
        is_partial_group=False,
        alignment_mode="special_zero_group",
        ch1_segment=ch1_seg[:common],
        ch3_segment=ch3_seg[:common],
        ch5_segment=ch5_seg[:common],
        temperature_segment=temp_seg[:common],
        temperature_time_segment=time_seg[:common],
        voltage_raw_signal=ch1,
    )
    return result, temp_name


def save_outputs(
    output_dir: Path,
    results: list[GroupAlignedResult],
    cfg: AlignConfig,
    temp_signal_name: str,
    group_count_found: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_id = results[0].device_id
    prefix = f"device_{device_id}"

    summary = output_dir / f"{prefix}_alignment_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
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
        for r in results:
            writer.writerow(
                {
                    "device_id": r.device_id,
                    "sample_rate_hz": cfg.sample_rate_hz,
                    "device_group_count_found": group_count_found,
                    "device_group_count_processed": len(results),
                    "group_id": r.group_id,
                    "ch1_file": r.ch1_file,
                    "ch3_file": r.ch3_file,
                    "ch5_file": r.ch5_file,
                    "ch1_raw_len": r.ch1_raw_len,
                    "ch3_raw_len": r.ch3_raw_len,
                    "ch5_raw_len": r.ch5_raw_len,
                    "voltage_start_idx": r.voltage_start_idx,
                    "voltage_end_idx": r.voltage_end_idx,
                    "voltage_trim_len": r.voltage_trim_len,
                    "falling_edge_count_all": r.falling_edge_count_all,
                    "falling_edges_all": json.dumps(r.falling_edges_all, ensure_ascii=False),
                    "falling_edges_used": json.dumps(r.falling_edges_used, ensure_ascii=False),
                    "temp_peak_count_all": r.temp_peak_count_all,
                    "temp_peaks_used_global": json.dumps(r.temp_peaks_used_global, ensure_ascii=False),
                    "temp_peak_idx_global": r.temp_peak_idx_global,
                    "temp_peak_idx_in_segment": r.temp_peak_idx_in_segment,
                    "temp_start_idx": r.temp_start_idx,
                    "temp_end_idx": r.temp_end_idx,
                    "aligned_len": r.aligned_len,
                    "used_cycles": r.used_cycles,
                    "is_partial_group": int(r.is_partial_group),
                    "alignment_mode": r.alignment_mode,
                }
            )

    for r in results:
        p = output_dir / f"{prefix}_group_{r.group_id:02d}_aligned.csv"
        arr = np.column_stack(
            [
                r.temperature_time_segment.reshape(-1),
                r.temperature_segment.reshape(-1),
                r.ch1_segment.reshape(-1),
                r.ch3_segment.reshape(-1),
                r.ch5_segment.reshape(-1),
            ]
        )
        np.savetxt(p, arr, delimiter=",", header="time,temperature,ch1,ch3,ch5", comments="")

    npz_file = output_dir / f"{prefix}_aligned_power_cycle.npz"
    np.savez(
        npz_file,
        device_id=np.asarray([device_id], dtype=object),
        group_ids=np.asarray([x.group_id for x in results], dtype=np.int32),
        ch1_segments=np.asarray([x.ch1_segment for x in results], dtype=object),
        ch3_segments=np.asarray([x.ch3_segment for x in results], dtype=object),
        ch5_segments=np.asarray([x.ch5_segment for x in results], dtype=object),
        temperature_segments=np.asarray([x.temperature_segment for x in results], dtype=object),
        temperature_time_segments=np.asarray([x.temperature_time_segment for x in results], dtype=object),
        aligned_lengths=np.asarray([x.aligned_len for x in results], dtype=np.int32),
        used_cycles=np.asarray([x.used_cycles for x in results], dtype=np.int32),
        alignment_mode=np.asarray([x.alignment_mode for x in results], dtype=object),
    )

    mat_file = output_dir / f"{prefix}_aligned_power_cycle.mat"
    n = len(results)
    ch1_cells = np.empty((1, n), dtype=object)
    ch3_cells = np.empty((1, n), dtype=object)
    ch5_cells = np.empty((1, n), dtype=object)
    temp_cells = np.empty((1, n), dtype=object)
    time_cells = np.empty((1, n), dtype=object)
    for i, r in enumerate(results):
        ch1_cells[0, i] = r.ch1_segment.reshape(-1, 1)
        ch3_cells[0, i] = r.ch3_segment.reshape(-1, 1)
        ch5_cells[0, i] = r.ch5_segment.reshape(-1, 1)
        temp_cells[0, i] = r.temperature_segment.reshape(-1, 1)
        time_cells[0, i] = r.temperature_time_segment.reshape(-1, 1)
    savemat(
        mat_file,
        {
            "device_id": np.asarray([device_id], dtype=object),
            "group_ids": np.asarray([x.group_id for x in results], dtype=np.int32).reshape(1, -1),
            "aligned_lengths": np.asarray([x.aligned_len for x in results], dtype=np.int32).reshape(1, -1),
            "used_cycles": np.asarray([x.used_cycles for x in results], dtype=np.int32).reshape(1, -1),
            "alignment_mode": np.asarray([x.alignment_mode for x in results], dtype=object).reshape(1, -1),
            "ch1_segments": ch1_cells,
            "ch3_segments": ch3_cells,
            "ch5_segments": ch5_cells,
            "temperature_segments": temp_cells,
            "temperature_time_segments": time_cells,
            "sample_rate_hz": np.asarray([[cfg.sample_rate_hz]], dtype=float),
            "high_threshold": np.asarray([[cfg.high_threshold]], dtype=float),
        },
    )

    meta = output_dir / f"{prefix}_alignment_meta.json"
    with meta.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "temp_signal_name": temp_signal_name,
                "device_group_count_found": group_count_found,
                "device_group_count_processed": len(results),
                "output_dir": str(output_dir.resolve()),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return {"summary_csv": summary, "npz": npz_file, "mat": mat_file, "meta_json": meta}


def save_visualizations(output_dir: Path, results: list[GroupAlignedResult], cfg: AlignConfig) -> tuple[Path, Path]:
    """保存对齐可视化检查图。"""
    plt = import_matplotlib_pyplot()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"device_{results[0].device_id}"

    for r in results:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

        # 图1：原始电压 + 边沿 + 截取区间
        ax_raw = axes[0]
        x_raw = np.arange(r.voltage_raw_signal.size, dtype=int)
        ax_raw.plot(x_raw, r.voltage_raw_signal, color="#3b3b3b", linewidth=1.0, label="Voltage (raw)")
        ax_raw.axhline(
            cfg.high_threshold,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.0,
            label=f"Threshold={cfg.high_threshold:.3f}V",
        )

        if r.falling_edges_all:
            idx_all = np.asarray(r.falling_edges_all, dtype=int)
            idx_all = idx_all[(idx_all >= 0) & (idx_all < r.voltage_raw_signal.size)]
            if idx_all.size > 0:
                ax_raw.scatter(
                    idx_all,
                    r.voltage_raw_signal[idx_all],
                    s=12,
                    color="#ffb347",
                    alpha=0.75,
                    label="Falling edges (all)",
                    zorder=3,
                )

        if r.falling_edges_used:
            idx_used = np.asarray(r.falling_edges_used, dtype=int)
            idx_used = idx_used[(idx_used >= 0) & (idx_used < r.voltage_raw_signal.size)]
            if idx_used.size > 0:
                ax_raw.scatter(
                    idx_used,
                    r.voltage_raw_signal[idx_used],
                    s=20,
                    color="#d62728",
                    alpha=0.95,
                    label="Falling edges (used)",
                    zorder=4,
                )

        ax_raw.axvline(r.voltage_start_idx, color="#2ca02c", linewidth=1.0, linestyle="--", label="Voltage start")
        ax_raw.axvline(r.voltage_end_idx, color="#9467bd", linewidth=1.0, linestyle="--", label="Voltage end")
        ax_raw.set_title(f"Group {r.group_id:02d}: Raw Voltage / Edge Detection")
        ax_raw.set_xlabel("Sample index")
        ax_raw.set_ylabel("Voltage (V)")
        ax_raw.grid(alpha=0.25, linewidth=0.5)
        ax_raw.legend(loc="best", fontsize=8)

        # 图2：对齐后电压与温度
        ax_v = axes[1]
        t_rel = np.arange(r.aligned_len, dtype=float) / float(cfg.sample_rate_hz)
        ax_v.plot(t_rel, r.ch1_segment, color="#1f77b4", linewidth=1.2, label="CH1 (aligned)")
        ax_v.set_xlabel("Time (s)")
        ax_v.set_ylabel("CH1 (V)", color="#1f77b4")
        ax_v.tick_params(axis="y", labelcolor="#1f77b4")
        ax_v.grid(alpha=0.25, linewidth=0.5)

        ax_t = ax_v.twinx()
        ax_t.plot(t_rel, r.temperature_segment, color="#d62728", linewidth=1.2, label="Temperature (aligned)")
        ax_t.set_ylabel("Temperature", color="#d62728")
        ax_t.tick_params(axis="y", labelcolor="#d62728")

        if 0 <= r.temp_peak_idx_in_segment < r.aligned_len:
            p = int(r.temp_peak_idx_in_segment)
            ax_t.scatter([t_rel[p]], [r.temperature_segment[p]], color="#d62728", s=30, zorder=5, label="First temp peak")

        hv, lv = ax_v.get_legend_handles_labels()
        ht, lt = ax_t.get_legend_handles_labels()
        ax_v.legend(hv + ht, lv + lt, loc="best", fontsize=8)
        ax_v.set_title(f"Group {r.group_id:02d}: Aligned Signals (len={r.aligned_len}, fs={cfg.sample_rate_hz:.1f}Hz)")

        fig.savefig(plot_dir / f"{prefix}_group_{r.group_id:02d}_check.png", dpi=170)
        plt.close(fig)

    # 组长度统计图
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
    p = argparse.ArgumentParser(description="15号器件温度-电压对齐脚本（独立版）")
    p.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--device-ids", default=",".join(DEFAULT_DEVICE_IDS))
    p.add_argument("--sample-rate", type=float, default=10.0)
    p.add_argument("--high-threshold", type=float, default=2.0)
    p.add_argument("--cycles-per-group", type=int, default=10)
    p.add_argument("--pre-points", type=int, default=155)
    p.add_argument("--post-points", type=int, default=245)
    p.add_argument("--edge-min-gap-sec", type=float, default=20.0)
    p.add_argument("--peak-smooth-window", type=int, default=7)
    p.add_argument("--peak-prom-factor", type=float, default=0.10)
    p.add_argument("--peak-min-gap-sec", type=float, default=20.0)
    p.add_argument(
        "--allow-partial-last-group",
        dest="allow_partial_last_group",
        action="store_true",
        help="最后一组允许不足10循环（默认开启）",
    )
    p.add_argument(
        "--no-allow-partial-last-group",
        dest="allow_partial_last_group",
        action="store_false",
        help="最后一组不足10循环时直接报错",
    )
    p.set_defaults(allow_partial_last_group=True)
    p.add_argument("--partial-min-cycles", type=int, default=3)
    p.add_argument("--temp-index", type=int, default=1, help="15号温度结构仅支持 1")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--max-groups", type=int, default=0)
    p.add_argument("--save-plots", dest="save_plots", action="store_true", help="保存可视化检查图（默认开启）")
    p.add_argument("--no-save-plots", dest="save_plots", action="store_false", help="不保存可视化检查图")
    p.set_defaults(save_plots=True)
    p.add_argument(
        "--process-zero-group",
        dest="process_zero_group",
        action="store_true",
        help="处理 group0（默认开启）",
    )
    p.add_argument(
        "--skip-zero-group",
        dest="process_zero_group",
        action="store_false",
        help="跳过 group0",
    )
    p.set_defaults(process_zero_group=True)
    p.add_argument("--list-temp-signals", action="store_true", help="仅列出温度信号并退出")
    return p.parse_args()


def run_for_device(
    device_id: str, input_root: Path, output_root: Path, cfg: AlignConfig, save_plots: bool
) -> None:
    print(f"\n===== 器件 {device_id} =====")
    device_dir = discover_device_dir(input_root, device_id)
    pc_dir = device_dir / "PC"
    temp_dir = device_dir / "温度"
    if not pc_dir.is_dir():
        raise FileNotFoundError(f"PC 目录不存在: {pc_dir}")
    if not temp_dir.is_dir():
        raise FileNotFoundError(f"温度目录不存在: {temp_dir}")

    all_groups = discover_pc_groups(pc_dir, device_id)
    if not all_groups:
        raise FileNotFoundError("未找到完整电压组（需同时包含 ch1/ch3/ch5）。")
    group0 = next((x for x in all_groups if x.group_id == 0), None)
    regular_groups = [x for x in all_groups if x.group_id != 0]
    if cfg.max_groups > 0:
        regular_groups = regular_groups[: cfg.max_groups]

    reg_temps, zero_temps = discover_temp_files(temp_dir, device_id)
    if not reg_temps and not zero_temps:
        raise FileNotFoundError(f"未找到温度文件: {temp_dir}")

    print(f"[1/4] 电压组总数(含group0): {len(all_groups)}")
    print(f"  常规组数: {len(regular_groups)}")
    print(f"  group0存在: {'是' if group0 else '否'}")
    print("[2/4] 温度文件列表 ...")
    print("  常规温度文件: " + (", ".join(x.name for x in reg_temps) if reg_temps else "无"))
    print("  group0温度文件: " + (", ".join(x.name for x in zero_temps) if zero_temps else "无"))

    print("[3/4] 开始对齐 ...")
    results: list[GroupAlignedResult] = []
    temp_names: list[str] = []

    if regular_groups:
        if not reg_temps:
            raise FileNotFoundError("存在常规电压组，但没有常规温度区间文件。")
        t_all, y_all, temp_name = load_temp_series(reg_temps, cfg)
        temp_names.append(temp_name)
        print(f"  常规温度序列长度: {y_all.size}")
        aligned = align_regular_groups(device_id, regular_groups, t_all, y_all, cfg)
        results.extend(aligned)

    if cfg.process_zero_group:
        if group0 is not None and zero_temps:
            if len(zero_temps) > 1:
                print(f"[WARN] group0 温度文件超过1个，仅使用: {zero_temps[0].name}")
            r0, tname0 = align_zero_group(device_id, group0, zero_temps[0], cfg)
            results.append(r0)
            temp_names.append(tname0)
        elif group0 is not None and not zero_temps:
            print("[WARN] 存在 group0 电压组，但未找到 group0 温度文件。")
        elif group0 is None and zero_temps:
            print("[WARN] 找到 group0 温度文件，但未找到 group0 电压组。")

    if not results:
        raise RuntimeError("没有可用对齐结果。")
    results = sorted(results, key=lambda x: int(x.group_id))

    out_dir = output_root / f"{device_id}号"
    group_found = len(regular_groups) + (1 if group0 else 0)
    temp_name_text = " / ".join(sorted(set(temp_names))) if temp_names else "index_1"

    print("[4/4] 写出结果 ...")
    out = save_outputs(out_dir, results, cfg, temp_name_text, group_found)
    if save_plots:
        plot_dir, length_plot = save_visualizations(out_dir, results, cfg)
        print(f"  可视化图目录: {plot_dir}")
        print(f"  组长度统计图: {length_plot.name}")
    print(f"  Summary: {out['summary_csv'].name}")
    print(f"  MAT: {out['mat'].name}")
    print(f"  NPZ: {out['npz'].name}")
    print(f"  Meta: {out['meta_json'].name}")
    print(f"- 输出目录: {out_dir.resolve()}")
    print(f"- 对齐组数: {len(results)}")
    for r in results:
        print(
            f"  group={r.group_id:02d}, len={r.aligned_len}, mode={r.alignment_mode}, "
            f"used_cycles={r.used_cycles}, voltage=[{r.voltage_start_idx},{r.voltage_end_idx}), "
            f"temp=[{r.temp_start_idx},{r.temp_end_idx})"
        )


def main() -> None:
    args = parse_args()
    cfg = AlignConfig(
        sample_rate_hz=args.sample_rate,
        high_threshold=args.high_threshold,
        cycles_per_group=args.cycles_per_group,
        pre_points=args.pre_points,
        post_points=args.post_points,
        edge_min_gap_sec=args.edge_min_gap_sec,
        peak_smooth_window=args.peak_smooth_window,
        peak_prom_factor=args.peak_prom_factor,
        peak_min_gap_sec=args.peak_min_gap_sec,
        allow_partial_last_group=args.allow_partial_last_group,
        partial_min_cycles=args.partial_min_cycles,
        strict_mode=args.strict,
        max_groups=args.max_groups,
        process_zero_group=args.process_zero_group,
        temp_index=args.temp_index,
    )

    print(
        "[CFG] "
        f"temp_index={cfg.temp_index}, "
        f"partial_min_cycles={cfg.partial_min_cycles}, "
        f"allow_partial_last_group={cfg.allow_partial_last_group}"
    )

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    device_ids = parse_device_ids(args.device_ids)

    if args.list_temp_signals:
        for device_id in device_ids:
            dev_dir = discover_device_dir(input_root, device_id)
            temp_dir = dev_dir / "温度"
            reg, zero = discover_temp_files(temp_dir, device_id)
            files = reg + zero
            if not files:
                print(f"[WARN] 器件 {device_id} 无温度文件。")
                continue
            names = list_temp_signals_for_15(files[0])
            print(f"器件 {device_id} 温度信号列表（文件: {files[0].name}）:")
            for i, name in enumerate(names, start=1):
                print(f"  {i}: {name}")
        return

    for device_id in device_ids:
        run_for_device(device_id, input_root, output_root, cfg, args.save_plots)


if __name__ == "__main__":
    main()
