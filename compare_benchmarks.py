"""
compare_benchmarks.py
Reads the three benchmark result JSON files and prints a formatted
three-way comparison table.

Expected files:
  benchmark_results.json              – Python (Ultralytics / PyTorch MPS)
  benchmark_results_cpp_cpu.json      – C++ ONNX Runtime, CPU provider
  benchmark_results_cpp_coreml.json   – C++ ONNX Runtime, CoreML provider
"""

import json
import sys
from pathlib import Path

RESULTS = [
    ("Python\n(PyTorch MPS)",    "benchmark_results.json"),
    ("C++ CPU\n(ONNX Runtime)",  "benchmark_results_cpp_cpu.json"),
    ("C++ CoreML\n(ONNX Runtime)", "benchmark_results_cpp_coreml.json"),
]


def load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt(val, unit="", dec=2) -> str:
    return f"{val:.{dec}f}{unit}"


def speedup(base_ms: float, other_ms: float) -> str:
    if other_ms <= 0:
        return "—"
    ratio = base_ms / other_ms
    return f"{ratio:.2f}x {'faster' if ratio >= 1 else 'slower'}"


def main():
    datasets = []
    for label, path in RESULTS:
        d = load(path)
        if d is None:
            print(f"WARNING: {path} not found — run that benchmark first.\n"
                  f"         Skipping '{label.split(chr(10))[0]}' column.\n")
        datasets.append((label, d))

    available = [(lbl, d) for lbl, d in datasets if d is not None]
    if len(available) < 2:
        print("Need at least two result files to compare. Run the benchmarks first.")
        sys.exit(1)

    # Column widths
    LW  = 26   # label column
    CW  = 24   # data column

    sep_parts = ["+" + "-" * (LW + 2)]
    for _ in available:
        sep_parts.append("-" * (CW + 2))
    sep = "+".join(sep_parts) + "+"

    def header_row():
        cells = [f" {'Metric':<{LW}} "]
        for lbl, _ in available:
            # Use only the first line of multi-line labels for the header
            short = lbl.split("\n")[0]
            cells.append(f" {short:^{CW}} ")
        return "|" + "|".join(cells) + "|"

    def sub_header_row():
        cells = [f" {'':<{LW}} "]
        for lbl, _ in available:
            parts = lbl.split("\n")
            sub = parts[1] if len(parts) > 1 else ""
            cells.append(f" {sub:^{CW}} ")
        return "|" + "|".join(cells) + "|"

    def data_row(label, values):
        cells = [f" {label:<{LW}} "]
        for v in values:
            cells.append(f" {v:>{CW}} ")
        return "|" + "|".join(cells) + "|"

    print()
    print("=" * (LW + (CW + 3) * len(available) + 1))
    print("  FOD Tool Tracker — Python vs C++ Inference Benchmark")
    print("=" * (LW + (CW + 3) * len(available) + 1))
    print()
    print(sep)
    print(header_row())
    print(sub_header_row())
    print(sep)

    # Backend
    print(data_row("Backend",
        [d.get("backend", "?").replace(" ONNX Runtime", "\nONNX Runtime")
                              .split("\n")[0]
         for _, d in available]))

    # Model file
    print(data_row("Model file",
        [d.get("model", "?") for _, d in available]))

    # Frames
    print(data_row("Frames benchmarked",
        [str(d.get("frames", "?")) for _, d in available]))

    # Input size
    print(data_row("Input size (px)",
        [f"{d.get('imgsz','?')}x{d.get('imgsz','?')}" for _, d in available]))

    print(sep)

    # Avg inference — baseline is the first column (Python)
    base_avg = available[0][1]["avg_inference_ms"]
    avg_vals = []
    for lbl, d in available:
        ms = d["avg_inference_ms"]
        sp = speedup(base_avg, ms) if d is not available[0][1] else "(baseline)"
        avg_vals.append(f"{ms:.2f} ms  [{sp}]")
    print(data_row("Avg inference time", avg_vals))

    # FPS
    base_fps = available[0][1]["fps"]
    fps_vals = []
    for lbl, d in available:
        fps = d["fps"]
        sp = speedup(fps, base_fps) if d is not available[0][1] else "(baseline)"
        fps_vals.append(f"{fps:.2f} fps  [{sp}]")
    print(data_row("FPS", fps_vals))

    # Min
    print(data_row("Min inference time",
        [fmt(d["min_inference_ms"], " ms") for _, d in available]))

    # Max
    print(data_row("Max inference time",
        [fmt(d["max_inference_ms"], " ms") for _, d in available]))

    print(sep)
    print()

    # Summary sentence
    if len(available) == 3:
        py_avg     = available[0][1]["avg_inference_ms"]
        cpu_avg    = available[1][1]["avg_inference_ms"]
        coreml_avg = available[2][1]["avg_inference_ms"]

        print(f"  C++ CPU    is {speedup(py_avg, cpu_avg)} vs Python MPS")
        print(f"  C++ CoreML is {speedup(py_avg, coreml_avg)} vs Python MPS")
        print(f"  C++ CoreML is {speedup(cpu_avg, coreml_avg)} vs C++ CPU")
    elif len(available) == 2:
        py_avg  = available[0][1]["avg_inference_ms"]
        cpp_avg = available[1][1]["avg_inference_ms"]
        print(f"  {available[1][0].split(chr(10))[0]} is {speedup(py_avg, cpp_avg)} vs Python MPS")

    print()


if __name__ == "__main__":
    main()
