#!/usr/bin/env python3
"""
15号器件温度结构专用对齐脚本。

说明：
1) 复用 align_power_cycle_data.py 的整体流程与电压处理逻辑。
2) 仅替换温度读取入口，兼容 15号温度文件的 v7.3 对象结构：
   - 常见结构：/#refs#/y (温度序列) + /#refs#/c (信号名，通常为 tempture)
3) 若遇到常规 Sigstream 结构（/#sigstream#），仍回退到原脚本解析逻辑。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import align_power_cycle_data as base


DEFAULT_INPUT_ROOT = r"H:\2026.2.10数据\老化数据"
DEFAULT_OUTPUT_ROOT = r"G:\深度学习资料\双脉冲实验平台\2026.3.1整理数据输出\aligned_output"
DEFAULT_DEVICE_IDS = ["15"]


def _decode_uint16_string(arr: np.ndarray) -> str:
    """将 MATLAB UTF-16 编码字符数组转换为 Python 字符串。"""
    flat = np.asarray(arr).reshape(-1)
    chars = [chr(int(x)) for x in flat if int(x) != 0]
    return "".join(chars)


def _load_temperature_signal_h5_device15(
    temp_file: Path, cfg: base.AlignConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    读取 15号温度文件（无 /#sigstream# 的 v7.3 对象结构）。

    结构特征：
    - /#refs#/y: 一维温度序列（float64）
    - /#refs#/c: 信号名（UTF-16），通常为 "tempture"
    """
    h5py = base._import_h5py()
    with h5py.File(temp_file, "r") as f:
        # 常规结构直接复用原脚本逻辑，保持兼容。
        if "/#sigstream#" in f and "/#refs#/c/Elements" in f:
            return base._load_temperature_signal_h5(temp_file, cfg)

        if "/#refs#/y" not in f:
            raise RuntimeError("温度文件不包含 /#refs#/y，无法按15号结构解析。")

        signal = np.asarray(f["/#refs#/y"][()], dtype=float).reshape(-1)
        if signal.size == 0:
            raise RuntimeError("温度文件 /#refs#/y 为空。")

        signal_name = "temperature"
        if "/#refs#/c" in f:
            signal_name = _decode_uint16_string(np.asarray(f["/#refs#/c"][()])) or "temperature"

    mode = cfg.temp_mode.lower()
    if mode == "index":
        idx = int(cfg.temp_signal_index)
        if idx != 1:
            raise ValueError(f"该温度结构仅包含 1 个通道（index=1），当前 temp-index={idx}。")
    elif mode == "name":
        target = cfg.temp_signal_name.strip().lower()
        if target and signal_name.strip().lower() != target:
            raise ValueError(
                f"温度通道名不匹配：文件为 '{signal_name}'，参数 --temp-name 为 '{cfg.temp_signal_name}'。"
            )
    else:
        raise ValueError("temp_mode 仅支持 index 或 name。")

    time = np.arange(signal.size, dtype=float) / float(cfg.sample_rate_hz)
    return time, signal, signal_name


def load_temperature_signal(temp_file: Path, cfg: base.AlignConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """温度读取入口：MATLAB engine 优先，失败后按 h5 结构解析。"""
    try:
        return base._load_temperature_signal_matlab(temp_file, cfg)
    except Exception as exc:
        print(f"[WARN] matlab.engine 不可用或读取失败，改用 h5py 解析温度文件: {exc}")
        return _load_temperature_signal_h5_device15(temp_file, cfg)


def list_temperature_signals(temp_file: Path) -> list[str]:
    """列出温度信号名：优先 MATLAB engine，失败后按 h5 结构解析。"""
    try:
        return base._list_temperature_signals_matlab(temp_file)
    except Exception as exc:
        print(f"[WARN] matlab.engine 不可用或读取失败，改用 h5py 枚举温度信号: {exc}")

    h5py = base._import_h5py()
    with h5py.File(temp_file, "r") as f:
        if "/#sigstream#" in f and "/#refs#/c/Elements" in f:
            return base.list_temperature_signals_h5(temp_file)
        if "/#refs#/c" in f:
            name = _decode_uint16_string(np.asarray(f["/#refs#/c"][()]))
            return [name or "temperature"]
    return ["temperature"]


def main() -> None:
    """
    启动 15号专用流程：
    - 默认输入: data
    - 默认输出: data/aligned_output_15
    - 默认器件: 15
    """
    base.DEFAULT_INPUT_ROOT = DEFAULT_INPUT_ROOT
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.DEFAULT_DEVICE_IDS = DEFAULT_DEVICE_IDS

    # 通过 monkey patch 复用原主流程，仅替换温度读取相关函数。
    base.load_temperature_signal = load_temperature_signal
    base.list_temperature_signals = list_temperature_signals

    base.main()


if __name__ == "__main__":
    main()
