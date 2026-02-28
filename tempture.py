import os
import numpy as np
import matlab.engine


# =========================
# ✅ 你后期最常改的参数在这里
# =========================
MAT_PATH = r"H:\2026.2.10数据\T_series\12号\251108_1-10_12.mat"
# ↑ 改成你的 .mat 文件路径（Windows 用原始字符串 r"" 最稳）

MODE = "index"            # <-- "index" 或 "name"
SIGNAL_INDEX = 9          # <-- MODE="index" 生效：读取 data{SIGNAL_INDEX}（MATLAB 1-based）
SIGNAL_NAME = "tempture"  # <-- MODE="name" 生效：按 data{i}.Name 匹配（你的拼写是 tempture）

OUT_DIR = None            # <-- 输出目录；None 表示跟 mat 文件同目录
SAVE_CSV = True           # <-- 是否保存 CSV（time,value）
SAVE_NPY = False           # <-- 是否保存 NPY（dict：time/value/name/index）

FLATTEN_DATA = True       # <-- True: y 拉平成一维；False: 保留原 shape（可能 Nx1 或 NxM）


def _to_numpy(x) -> np.ndarray:
    return np.array(x, dtype=float)


def list_signals(mat_path: str):
    """列出 Dataset 内所有信号的 (索引, Name)"""
    eng = matlab.engine.start_matlab()

    # ✅ MATLAB workspace 变量名必须以字母开头
    eng.workspace["MATPATH"] = os.path.abspath(mat_path)
    eng.eval("S = load(MATPATH);", nargout=0)

    eng.eval("N = S.data.numElements;", nargout=0)
    n = int(eng.workspace["N"])

    names = []
    for i in range(1, n + 1):
        eng.workspace["I"] = float(i)
        eng.eval("nm = S.data{I}.Name;", nargout=0)
        names.append(str(eng.workspace["nm"]))

    eng.quit()
    return names


def read_timeseries(mat_path: str, mode: str, signal_index: int, signal_name: str):
    """读取指定 signal 的 timeseries：返回 (t, y, idx, name)"""
    eng = matlab.engine.start_matlab()
    eng.eval("warning('off','all');", nargout=0)

    eng.workspace["MATPATH"] = os.path.abspath(mat_path)
    eng.eval("S = load(MATPATH);", nargout=0)

    if mode.lower() == "index":
        eng.workspace["IDX"] = float(signal_index)
        eng.eval("sig = S.data{IDX};", nargout=0)
        idx = signal_index

    elif mode.lower() == "name":
        eng.workspace["TARGET"] = signal_name
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
        eng.eval("IDX = hit;", nargout=0)
        idx = int(float(eng.workspace["IDX"]))
    else:
        eng.quit()
        raise ValueError('MODE must be "index" or "name".')

    # timeseries: sig.Values -> Time/Data
    eng.eval("ts = sig.Values;", nargout=0)
    eng.eval("t = ts.Time;", nargout=0)
    eng.eval("y = ts.Data;", nargout=0)
    eng.eval("nm = sig.Name;", nargout=0)

    name = str(eng.workspace["nm"])

    t = _to_numpy(eng.workspace["t"]).reshape(-1)

    y_np = _to_numpy(eng.workspace["y"])
    if FLATTEN_DATA:
        y_np = y_np.reshape(-1)

    eng.quit()
    return t, y_np, idx, name


def save_outputs(mat_path: str, t: np.ndarray, y: np.ndarray, idx: int, name: str):
    out_dir = os.path.dirname(os.path.abspath(mat_path)) if OUT_DIR is None else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(mat_path))[0]
    tag = f"data{idx}_{name}"

    csv_path = os.path.join(out_dir, f"{base}_{tag}.csv") #数据保存名
    npy_path = os.path.join(out_dir, f"{base}_{tag}.npy")

    if SAVE_CSV:
        if y.ndim == 1:
            arr = np.column_stack([t, y])
            header = "time,value"
        else:
            arr = np.column_stack([t.reshape(-1, 1), y])
            header = "time," + ",".join([f"value_{i}" for i in range(y.shape[1])])
        np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
        print("Saved CSV:", csv_path)

    if SAVE_NPY:
        np.save(npy_path, {"time": t, "value": y, "index": idx, "name": name})
        print("Saved NPY:", npy_path)


if __name__ == "__main__":
    names = list_signals(MAT_PATH)
    print("Dataset signals (1-based index):")
    for i, nm in enumerate(names, start=1):
        print(f"  {i}: {nm}")

    t, y, idx, name = read_timeseries(
        mat_path=MAT_PATH,
        mode=MODE,                 # <-- 改这里：index / name
        signal_index=SIGNAL_INDEX, # <-- 改这里：要读 data{几}
        signal_name=SIGNAL_NAME,   # <-- 改这里：要按 Name 找哪个
    )

    print(f"\nLoaded: index={idx}, name={name}, time_len={len(t)}, y_shape={y.shape}")
    print("Head (time, value):")
    if y.ndim == 1:
        print(np.column_stack([t[:5], y[:5]]))
    else:
        print(np.column_stack([t[:5].reshape(-1, 1), y[:5]]))

    save_outputs(MAT_PATH, t, y, idx, name)

    #value = np.load("xxx.npy", allow_pickle=True).item()["value"] # 一键读取数据