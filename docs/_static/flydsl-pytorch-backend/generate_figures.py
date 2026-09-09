"""Rebuild the four published operator charts from the accompanying CSV files.

Run with Python, Matplotlib, and NumPy. No GPU or PyTorch installation is needed.
Source measurements and their limits are documented in README.md.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
ORANGE = "#E4492C"
BLUE = "#3576A8"
PURPLE = "#7861A8"
GRAY = "#8D99A6"
INK = "#243447"
GRID = "#E4E8ED"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "flydsl-pytorch-blog",
        "savefig.facecolor": "white",
    }
)


def rows(name):
    with (ROOT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def geomean(values):
    return float(np.exp(np.mean(np.log(values))))


def style(ax, maximum, ticks):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.axvline(1, color=INK, linestyle=(0, (4, 4)), linewidth=1.2)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_xlim(0, maximum)
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])


def title(fig, headline, subtitle):
    fig.text(0.04, 0.965, headline, fontsize=19, fontweight="bold", va="top")
    fig.text(0.04, 0.965 - 0.38 / fig.get_figheight(), subtitle, fontsize=12, va="top", color="#526477")


def save(fig, name):
    for extension in ("png", "svg"):
        metadata = {"Software": "FlyDSL blog figures"} if extension == "png" else {"Date": None}
        output = ROOT / f"{name}.{extension}"
        fig.savefig(output, dpi=180, metadata=metadata)
        if extension == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


def dense_gemm():
    data = rows("dense_gemm.csv")
    speed = np.array(
        [float(r["flydsl_tflops"]) / max(float(r["aten_tflops"]), float(r["triton_tflops"])) for r in data]
    )
    labels = [" × ".join(r[k] for k in ("m", "n", "k")) for r in data]
    positions = np.r_[np.arange(len(data)), len(data) + 0.8]
    values = np.r_[speed, geomean(speed)]
    colors = [ORANGE if s > 1.01 else GRAY for s in speed] + [BLUE]
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    fig.subplots_adjust(left=0.28, right=0.92, top=0.855, bottom=0.12)
    title(
        fig,
        "Dense GEMM: gains across BF16 matrix shapes",
        "NT layout · speedup over the faster ATen/Triton baseline at each shape",
    )
    ax.barh(positions, values, height=0.64, color=colors)
    ax.set_yticks(positions, labels + ["Geometric mean"])
    ax.invert_yaxis()
    style(ax, 1.52, [0, 0.5, 1, 1.5])
    ax.set_xlabel("FlyDSL speedup · shape labels are M × N × K", labelpad=10)
    for y, value in zip(positions, values):
        ax.text(max(value, 1) + 0.025, y, f"{value:.2f}×", va="center", fontsize=11)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    wins, ties, losses = sum(speed > 1.01), sum((speed >= 0.99) & (speed <= 1.01)), sum(speed < 0.99)
    fig.text(
        0.04,
        0.028,
        f"All 15 shapes shown · {wins} wins / {ties} ties / {losses} losses (±1% tie band) · dashed line = baseline",
        fontsize=11,
    )
    print(f"Dense: geomean vs faster baseline {geomean(speed):.4f}; {wins}/{ties}/{losses}")
    save(fig, "flydsl-dense-gemm-performance")


def grouped_gemm():
    data = rows("grouped_gemm.csv")
    suites = ["standard", "kn_variants", "ragged_m"]
    labels = ["Standard suite\n14 shapes", "K/N variants\n5 shapes", "Ragged M\n5 shapes"]
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    fig.subplots_adjust(left=0.20, right=0.93, top=0.76, bottom=0.19)
    title(
        fig,
        "Grouped GEMM: throughput across MoE workload suites",
        "BF16 · geometric-mean speedup over each baseline · all cases in each suite",
    )
    y = np.arange(len(suites))
    for baseline, color, offset, label in [("triton", BLUE, -0.18, "vs Triton"), ("aten", ORANGE, 0.18, "vs ATen")]:
        values = [
            geomean([float(r["flydsl_tflops"]) / float(r[f"{baseline}_tflops"]) for r in data if r["suite"] == suite])
            for suite in suites
        ]
        ax.barh(y + offset, values, height=0.30, color=color, label=label)
        for yy, value in zip(y + offset, values):
            ax.text(value + 0.035, yy, f"{value:.2f}×", va="center", fontsize=12)
        print(f"Grouped vs {baseline}: " + ", ".join(f"{s}={v:.4f}" for s, v in zip(suites, values)))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    style(ax, 2.5, [0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("FlyDSL geometric-mean speedup", labelpad=10)
    fig.legend(
        handles=[Patch(color=BLUE, label="vs Triton"), Patch(color=ORANGE, label="vs ATen")],
        loc="upper right",
        bbox_to_anchor=(0.94, 0.845),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.04,
        0.035,
        "The standard suite includes three cases where another backend is faster · dashed line = baseline",
        fontsize=11,
    )
    save(fig, "flydsl-grouped-gemm-performance")


def rmsnorm():
    data = rows("rmsnorm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 8.4))
    fig.subplots_adjust(left=0.19, right=0.95, top=0.79, bottom=0.12, wspace=0.90)
    title(
        fig,
        "RMSNorm: aligned and off-by-one hidden dimensions",
        "FP16 / BF16 / FP32 · speedup over ATen · all 22 reported cases",
    )
    colors = {"fp16": ORANGE, "bf16": BLUE, "fp32": PURPLE}
    for ax, odd, heading in zip(axes, [False, True], ["Aligned N", "N one above an aligned size"]):
        subset = [r for r in data if bool(int(r["n"]) % 8) == odd]
        values = [float(r["aten_us"]) / float(r["flydsl_us"]) for r in subset]
        positions = np.arange(len(subset))
        ax.barh(positions, values, height=0.62, color=[colors[r["dtype"]] for r in subset])
        ax.set_yticks(positions, [f"{r['dtype'].upper()}  {r['m']} × {r['n']}" for r in subset], fontsize=10.5)
        ax.set_ylim(len(subset) - 0.3, -0.7)
        style(ax, 4.15, [0, 1, 2, 3, 4])
        ax.set_title(heading, fontsize=13, fontweight="bold", loc="left", pad=14)
        ax.set_xlabel("FlyDSL speedup", labelpad=10)
        for y, value in zip(positions, values):
            ax.text(value + 0.065, y, f"{value:.2f}×", va="center", fontsize=10.5)
        print(f"RMSNorm {heading}: {min(values):.4f}–{max(values):.4f}")
    fig.text(
        0.04, 0.030, "Labels: dtype and M × N · each case is one warm timing run · dashed line = ATen", fontsize=11
    )
    save(fig, "flydsl-rmsnorm-performance")


def topk():
    data = rows("topk.csv")
    bands = ["2,4,8,16", "64-256", "257-383", "384-831", "832-1024"]
    labels = [
        "Register · K = {2, 4, 8, 16}\n10 shapes",
        "Radix · K = 64–256\n6 shapes",
        "Radix · K = 257–383\n6 shapes",
        "Radix · K = 384–831\n6 shapes",
        "Radix · K = 832–1024\n5 shapes",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    fig.subplots_adjust(left=0.31, right=0.93, top=0.78, bottom=0.16)
    title(
        fig,
        "TopK: speedups for register and radix-select kernels",
        "FP32 · geometric-mean speedup over ATen in the same determinism mode",
    )
    y = np.arange(len(bands))
    handles = []
    for mode, color, offset, label in [
        ("nondeterministic", ORANGE, -0.18, "Determinism off"),
        ("deterministic", BLUE, 0.18, "Determinism on"),
    ]:
        values = [
            geomean([float(r[f"aten_{mode}_us"]) / float(r[f"flydsl_{mode}_us"]) for r in data if r["k_band"] == band])
            for band in bands
        ]
        ax.barh(y + offset, values, height=0.30, color=color)
        for yy, value in zip(y + offset, values):
            ax.text(value + 0.065, yy, f"{value:.2f}×", va="center", fontsize=11)
        handles.append(Patch(color=color, label=label))
        print(f"TopK {mode}: " + ", ".join(f"{b}={v:.4f}" for b, v in zip(bands, values)))
    ax.set_yticks(y, labels, fontsize=11)
    ax.invert_yaxis()
    style(ax, 5.5, [0, 1, 2, 3, 4, 5])
    ax.set_xlabel("FlyDSL geometric-mean speedup", labelpad=10)
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.94, 0.86), frameon=False, ncol=2)
    fig.text(
        0.04,
        0.030,
        "All 33 reported shapes · register tie ordering can differ from ATen · dashed line = baseline",
        fontsize=11,
    )
    save(fig, "flydsl-topk-performance")


if __name__ == "__main__":
    dense_gemm()
    grouped_gemm()
    rmsnorm()
    topk()
