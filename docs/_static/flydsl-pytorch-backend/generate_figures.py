"""Rebuild the architecture overview and published performance charts.

Run with Python, Matplotlib, and NumPy. No GPU or PyTorch installation is needed.
Charts use the accompanying CSV files. Sources and limits are in README.md.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

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


def architecture():
    fig, ax = plt.subplots(figsize=(12.6, 8.6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    title(
        fig,
        "Two paths from PyTorch to FlyDSL kernels",
        "Automatic dispatch in eager mode · autotuning with torch.compile",
    )

    def box(x, y, width, height, fill="white", edge="#CBD5DF"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=1.1",
                linewidth=1.2,
                edgecolor=edge,
                facecolor=fill,
                zorder=2,
            )
        )

    def label(x, y, text, size=13, bold=False, color=INK):
        ax.text(
            x,
            y,
            text,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            color=color,
            ha="center",
            va="center",
            zorder=3,
        )

    def line(points):
        x, y = zip(*points)
        ax.plot(x, y, color="#7D8E9E", linewidth=1.4, zorder=3)

    def arrow(start, end):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color="#7D8E9E",
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    box(18, 78, 64, 10)
    label(50, 84.5, "PyTorch application", size=15, bold=True)
    label(50, 80.6, "Matrix multiplication · normalization · selection", size=12)

    # Each mode has its own selection policy; both execute on the same GPU.
    box(4, 19, 44, 53, fill="#F6F8FB", edge=GRID)
    box(52, 19, 44, 53, fill="#F6F8FB", edge=GRID)
    line([(50, 78), (50, 75), (26, 75)])
    line([(50, 75), (74, 75)])
    arrow((26, 75), (26, 72))
    arrow((74, 75), (74, 72))

    label(26, 67.7, "Eager execution", size=16, bold=True)
    label(26, 63.4, "RMSNorm · TopK", size=12)
    arrow((26, 61), (26, 59))
    box(9, 49, 34, 10)
    label(26, 55.4, "Dispatch by input support", bold=True)
    label(26, 51.7, "Shape · dtype · layout", size=11.5, color="#526477")
    line([(26, 49), (26, 46.5), (16.5, 46.5)])
    line([(26, 46.5), (35.5, 46.5)])
    arrow((16.5, 46.5), (16.5, 39))
    arrow((35.5, 46.5), (35.5, 39))
    # Light backgrounds keep branch labels legible over the connectors.
    for x, text in [(16.5, "Eligible inputs"), (35.5, "Other inputs")]:
        ax.text(
            x,
            43,
            text,
            fontsize=11,
            ha="center",
            va="center",
            bbox={"facecolor": "#F6F8FB", "edgecolor": "none", "pad": 2},
            zorder=4,
        )
    box(8, 31, 17, 8, fill="#FFF0EB", edge=ORANGE)
    box(27, 31, 17, 8)
    label(16.5, 35, "FlyDSL kernel", bold=True, color=ORANGE)
    label(35.5, 35, "ATen kernel", bold=True)
    line([(16.5, 31), (16.5, 25.5), (26, 25.5)])
    line([(35.5, 31), (35.5, 25.5), (26, 25.5)])
    arrow((26, 25.5), (26, 14))

    label(74, 67.7, "torch.compile", size=16, bold=True)
    label(74, 63.4, "Dense / grouped GEMM · MXFP8 / MXFP4", size=11.5)
    arrow((74, 61), (74, 59))
    box(56, 43, 36, 16)
    label(74, 55.4, "Benchmark eligible candidates", size=12.5, bold=True)
    for x, name in [(57, "ATen"), (69, "Triton"), (81, "FlyDSL")]:
        flydsl = name == "FlyDSL"
        box(x, 47, 10, 5, fill="#FFF0EB" if flydsl else "#F6F8FB", edge=ORANGE if flydsl else GRID)
        label(x + 5, 49.5, name, size=12, bold=True, color=ORANGE if flydsl else INK)
    label(74, 44.8, "Candidates depend on the operation", size=10.5, color="#526477")
    arrow((74, 43), (74, 37))
    box(57, 28, 34, 9)
    label(74, 32.5, "Run fastest measured kernel", size=12.5, bold=True)
    arrow((74, 28), (74, 14))

    box(18, 4, 64, 10, fill="#EDF3F8", edge=BLUE)
    label(50, 9, "AMD MI350-series GPU (gfx950)", size=15, bold=True)
    save(fig, "flydsl-pytorch-architecture")


def kernel_scheduling():
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.87, bottom=0.05, hspace=0.24, wspace=0.12)
    title(
        fig,
        "Kernel scheduling at a glance",
        "The supported operators use different work partitioning, data movement, and selection strategies",
    )

    def prepare(ax, heading):
        ax.set(xlim=(0, 100), ylim=(0, 100))
        ax.set_facecolor("#F6F8FB")
        ax.axis("off")
        ax.text(3, 94, heading, fontsize=14, fontweight="bold", va="top")

    def box(ax, x, y, width, height, text, fill="white", edge="#CBD5DF", size=10, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=1.2",
                linewidth=1.1,
                edgecolor=edge,
                facecolor=fill,
            )
        )
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight="bold" if bold else "normal",
        )

    def arrow(ax, start, end):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#7D8E9E",
                shrinkA=1,
                shrinkB=1,
            )
        )

    def grid_rect(ax, x, y, width, height, rows, cols, fill, edge=BLUE, linewidth=1.1):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=linewidth))
        for row in range(1, rows):
            yy = y + height * row / rows
            ax.plot([x, x + width], [yy, yy], color=edge, linewidth=0.65)
        for col in range(1, cols):
            xx = x + width * col / cols
            ax.plot([xx, xx], [y, y + height], color=edge, linewidth=0.65)

    dense, grouped, rms, topk_ax = axes.flat

    prepare(dense, "A · Dense / MXFP HTI tile mapping")
    dense.text(3, 84, "A[BM, BK] × B[BK, BN] → C[BM, BN]", fontsize=11, fontweight="bold")
    grid_rect(dense, 5, 38, 17, 38, 4, 2, "#DCEAF5")
    grid_rect(dense, 33, 55, 28, 18, 2, 4, "#FFF0D3", edge="#D9A441")
    grid_rect(dense, 70, 36, 27, 42, 2, 2, "#FDE2DA", edge=ORANGE, linewidth=1.5)
    dense.text(13.5, 79, "A tile", ha="center", fontsize=10, fontweight="bold")
    dense.text(47, 76, "B tile", ha="center", fontsize=10, fontweight="bold")
    dense.text(83.5, 81, "C tile · 256 × 256", ha="center", fontsize=10, fontweight="bold")
    dense.text(2, 57, "BM", ha="right", va="center", fontsize=9.5)
    dense.text(13.5, 33, "BK", ha="center", fontsize=9.5)
    dense.text(30, 64, "BK", ha="right", va="center", fontsize=9.5)
    dense.text(47, 50, "BN", ha="center", fontsize=9.5)
    dense.text(67, 57, "BM", ha="right", va="center", fontsize=9.5)
    dense.text(83.5, 31, "BN", ha="center", fontsize=9.5)
    dense.text(27, 62, "×", fontsize=18, fontweight="bold", ha="center", va="center")
    dense.text(65, 62, "→", fontsize=18, fontweight="bold", ha="center", va="center")
    for row in range(1, 2):
        yy = 57 + 21 * row / 2
        dense.plot([70, 83.5], [yy, yy], color=PURPLE, linewidth=0.7)
    for col in range(1, 4):
        xx = 70 + 13.5 * col / 4
        dense.plot([xx, xx], [57, 78], color=PURPLE, linewidth=0.7)
    dense.text(76.75, 67.5, "C00\n2 × 4 waves", ha="center", va="center", fontsize=7.5, fontweight="bold")
    dense.text(90.25, 67.5, "C01", ha="center", va="center", fontsize=8.5, fontweight="bold")
    dense.text(76.75, 46.5, "C10", ha="center", va="center", fontsize=8.5, fontweight="bold")
    dense.text(90.25, 46.5, "C11", ha="center", va="center", fontsize=8.5, fontweight="bold")
    box(dense, 5, 12, 37, 12, "K tile t / t+1\nstaged in LDS", fill="#FFF7E8", edge="#D9A441", size=9.5)
    box(dense, 56, 12, 39, 12, "MFMA current pair\nprefetch next pair", fill="#FFF0EB", edge=ORANGE, size=9.5)
    arrow(dense, (42, 18), (56, 18))
    dense.text(3, 4, "HTI keeps C00–C11 resident; MXFP co-stages E8M0 block scales.", fontsize=10)

    prepare(grouped, "B · Grouped GEMM persistent tiles")
    grouped.text(3, 84, "Aᵍ[M_g, K] × Bᵍ[K, N] for each expert g", fontsize=11, fontweight="bold")
    expert_specs = [
        (5, 54, 17, 25, 4, "E0 · M₀"),
        (29, 60, 17, 19, 3, "E1 · M₁"),
        (53, 75, 17, 2, 1, "E2 · M₂=0"),
        (77, 48, 17, 31, 5, "E3 · M₃"),
    ]
    for x, y, width, height, rows_count, label in expert_specs:
        fill, edge = ("#E9EDF1", GRAY) if "M₂" in label else ("#DCEAF5", BLUE)
        grid_rect(grouped, x, y, width, height, rows_count, 2, fill, edge=edge)
        grouped.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": fill, "edgecolor": "none", "pad": 1},
        )
    grouped.text(3, 42, "Flatten valid matrix tiles into one global stream", fontsize=10.5, fontweight="bold")
    for x, label in [(3, "E0·0"), (19, "E0·1"), (35, "E1·0"), (51, "E1·1"), (67, "E3·0"), (83, "E3·1")]:
        box(grouped, x, 29, 13, 9, label, fill="#FFF0EB", edge=ORANGE, size=9.5)
    arrow(grouped, (13, 54), (10, 39))
    arrow(grouped, (37, 60), (42, 39))
    arrow(grouped, (85, 48), (88, 39))
    box(grouped, 5, 12, 27, 9, "WG0: 0 → 3 → 6", fill="white", edge="#CBD5DF", size=9.5)
    box(grouped, 37, 12, 27, 9, "WG1: 1 → 4 → 7", fill="white", edge="#CBD5DF", size=9.5)
    box(grouped, 69, 12, 27, 9, "WG2: 2 → 5 → 8", fill="white", edge="#CBD5DF", size=9.5)
    grouped.text(3, 4, "Persistent workgroups cross expert boundaries; M₂=0 contributes no tiles.", fontsize=10)

    prepare(rms, "C · RMSNorm")
    box(rms, 3, 68, 23, 13, "One input row", fill="#EDF3F8", edge=BLUE, bold=True)
    box(rms, 38, 68, 25, 13, "128-bit loads\nregister values", fill="white", edge="#CBD5DF")
    box(rms, 74, 68, 23, 13, "Apply rstd\n× weight", fill="#FFF0EB", edge=ORANGE)
    arrow(rms, (26, 74.5), (38, 74.5))
    arrow(rms, (63, 74.5), (74, 74.5))
    rms.text(3, 55, "Wave64 partial sum-of-squares", fontsize=10.5, fontweight="bold")
    for x, label in [(5, "W0"), (27, "W1"), (49, "W2"), (71, "W3")]:
        box(rms, x, 39, 18, 11, label, fill="#E9E2F4", edge=PURPLE, bold=True)
    box(rms, 18, 19, 28, 12, "LDS partials", fill="#FFF7E8", edge="#D9A441")
    box(rms, 55, 19, 28, 12, "Wave 0 final reduce", fill="#E9E2F4", edge=PURPLE)
    arrow(rms, (50, 39), (35, 31))
    arrow(rms, (46, 25), (55, 25))
    rms.text(3, 5, "One thread block per row; resident values produce output and backward rstd.", fontsize=10)

    prepare(topk_ax, "D · TopK")
    topk_ax.text(3, 83, "Small fixed K", fontsize=10.5, fontweight="bold")
    box(topk_ax, 3, 64, 18, 13, "K = 2/4/8/16", fill="#EDF3F8", edge=BLUE)
    box(topk_ax, 29, 64, 20, 13, "Lane-local\nbitonic sort", fill="white", edge="#CBD5DF")
    box(topk_ax, 57, 64, 20, 13, "Butterfly\ntop-K merge", fill="#E9E2F4", edge=PURPLE)
    box(topk_ax, 85, 64, 12, 13, "K pairs", fill="#FFF0EB", edge=ORANGE)
    for start, end in [((21, 70.5), (29, 70.5)), ((49, 70.5), (57, 70.5)), ((77, 70.5), (85, 70.5))]:
        arrow(topk_ax, start, end)
    topk_ax.text(3, 49, "Larger K", fontsize=10.5, fontweight="bold")
    box(topk_ax, 3, 29, 18, 13, "Full row", fill="#EDF3F8", edge=BLUE)
    box(topk_ax, 29, 29, 20, 13, "Four radix\nbyte passes", fill="white", edge="#CBD5DF")
    box(topk_ax, 57, 29, 20, 13, "Threshold\ncandidates", fill="#E9E2F4", edge=PURPLE)
    box(topk_ax, 85, 29, 12, 13, "K pairs", fill="#FFF0EB", edge=ORANGE)
    for start, end in [((21, 35.5), (29, 35.5)), ((49, 35.5), (57, 35.5)), ((77, 35.5), (85, 35.5))]:
        arrow(topk_ax, start, end)
    topk_ax.text(3, 6, "Deterministic: prefix-sum slots · Off: atomic slot allocation.", fontsize=10)

    save(fig, "flydsl-kernel-scheduling")


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
        "Dense GEMM: performance across BF16 matrix shapes",
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
        f"15 shapes · {wins} wins / {ties} ties / {losses} losses (±1%) · "
        "orange = win · gray = tie/loss · blue = geometric mean · dashed = faster baseline",
        fontsize=10.5,
    )
    print(f"Dense: geomean vs faster baseline {geomean(speed):.4f}; {wins}/{ties}/{losses}")
    save(fig, "flydsl-dense-gemm-results")


def mxfp_gemm():
    data = rows("mxfp_gemm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 10.4), sharey=True)
    fig.subplots_adjust(left=0.215, right=0.97, top=0.855, bottom=0.11, wspace=0.22)
    title(
        fig,
        "MXFP scaled GEMM: low-precision performance across shapes",
        "MI355X · source updated Sep 20 · NT layout · speedup over ATen · 17 shapes per format",
    )
    for ax, fmt in zip(axes, ["mxfp8", "mxfp4"]):
        subset = [r for r in data if r["format"] == fmt]
        labels = [" × ".join(r[k] for k in ("m", "n", "k")) for r in subset]
        positions = np.r_[np.arange(len(subset)), len(subset) + 0.65]
        speed = [float(r["flydsl_tflops"]) / float(r["aten_tflops"]) for r in subset]
        values = np.r_[speed, geomean(speed)]
        ax.barh(positions, values, height=0.54, color=[ORANGE] * len(speed) + [BLUE])
        for y, value in zip(positions, values):
            ax.text(max(value, 1) + 0.04, y, f"{value:.2f}×", va="center", fontsize=11)
        print(f"{fmt.upper()} vs ATen: geomean={geomean(speed):.4f}, range={min(speed):.4f}–{max(speed):.4f}")
        ax.set_yticks(positions, labels + ["Geometric mean"], fontsize=11)
        ax.set_ylim(positions[-1] + 0.7, -0.8)
        style(ax, 3.2, [0, 1, 2, 3])
        ax.set_xlabel("FlyDSL speedup", labelpad=10)
        ax.set_title(fmt.upper(), fontsize=15, fontweight="bold", loc="left", pad=14)
        if ax is axes[0]:
            ax.get_yticklabels()[-1].set_fontweight("bold")
    fig.text(0.04, 0.025, "Shape labels: M × N × K · dashed line = ATen", fontsize=11)
    save(fig, "flydsl-mxfp-gemm-results")


def grouped_gemm():
    data = rows("grouped_gemm.csv")
    suites = ["standard", "kn_variants", "ragged_m"]
    headings = [
        "Uniform groups\nG × M × K × N",
        "Projection dimensions\nG = 8, M = 512 per group",
        "Ragged expert loads\nG = 8, K = N = 4096 · labels list M",
    ]
    fig = plt.figure(figsize=(13.6, 10.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.55, wspace=0.82)
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.subplots_adjust(left=0.14, right=0.97, top=0.83, bottom=0.10)
    title(
        fig,
        "Grouped GEMM: per-case performance across MoE workloads",
        "BF16 · speedup over the faster ATen/Triton baseline at each case · all 24 cases",
    )

    def case_label(suite, case):
        if suite == "standard":
            return " × ".join(case.removeprefix("g").split("x"))
        if suite == "kn_variants":
            return case.replace(" x ", " × ")
        return case

    for ax, suite, heading in zip(axes, suites, headings):
        subset = [r for r in data if r["suite"] == suite]
        speed = [float(r["flydsl_tflops"]) / max(float(r["triton_tflops"]), float(r["aten_tflops"])) for r in subset]
        positions = np.r_[np.arange(len(subset)), len(subset) + 0.65]
        values = np.r_[speed, geomean(speed)]
        colors = [ORANGE if value > 1.01 else GRAY for value in speed] + [BLUE]
        labels = [case_label(suite, r["case"]) for r in subset] + ["Geometric mean"]

        ax.barh(positions, values, height=0.58, color=colors)
        ax.set_yticks(positions, labels, fontsize=8.5 if suite == "ragged_m" else 9.5)
        ax.set_ylim(positions[-1] + 0.7, -0.8)
        style(ax, 1.55, [0, 0.5, 1, 1.5])
        ax.set_xlabel("FlyDSL speedup", labelpad=8)
        ax.set_title(heading, fontsize=13, fontweight="bold", loc="left", pad=12)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        for y, value in zip(positions, values):
            ax.text(max(value, 1) + 0.025, y, f"{value:.2f}×", va="center", fontsize=10)

        wins = sum(value > 1.01 for value in speed)
        ties = sum(0.99 <= value <= 1.01 for value in speed)
        losses = len(speed) - wins - ties
        print(f"Grouped {suite} vs faster baseline: geomean={geomean(speed):.4f}; {wins}/{ties}/{losses}")

    fig.text(
        0.04,
        0.022,
        "Orange = FlyDSL win · gray = tie/loss (±1% tie band) · blue = geometric mean · dashed line = faster baseline",
        fontsize=11,
    )
    save(fig, "flydsl-grouped-gemm-workloads-performance")


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
        0.04,
        0.030,
        "Labels: dtype and M × N · one run per case, with 10 warmup + 50 timed iterations · dashed = ATen",
        fontsize=10.5,
    )
    save(fig, "flydsl-rmsnorm-results")


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
        "FP32 · geometric-mean speedup over ATen · same PyTorch deterministic-algorithms setting",
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
        "33 shapes · determinism changes radix tie gathering; register tie order may differ from ATen · dashed = baseline",
        fontsize=10.5,
    )
    save(fig, "flydsl-topk-results")


def vllm_e2e():
    data = rows("vllm_e2e.csv")
    models = [
        ("Qwen3-32B", "Qwen3 32B", BLUE),
        ("Llama-3.1-8B-Instruct", "Llama 3.1 8B", PURPLE),
        ("Llama-3.3-70B-Instruct", "Llama 3.3 70B", ORANGE),
    ]
    concurrency = [8, 16, 32, 64, 128, 256]
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 9.2))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.78, bottom=0.10, hspace=0.42)
    title(
        fig,
        "vLLM end-to-end: whole-request speedup",
        "MI355X · one GPU · ISL 256 / OSL 512 · positive values favor FlyDSL",
    )
    y = np.arange(len(concurrency))
    height = 0.22
    handles = []

    for ax, precision, heading, limits, ticks in [
        (axes[0], "bf16", "BF16", (-3, 15), [-2, 0, 5, 10, 15]),
        (axes[1], "mxfp8", "MXFP8 A8W8", (0, 115), [0, 25, 50, 75, 100]),
    ]:
        lookup = {
            (r["model"], int(r["concurrency"])): float(r["whole_request_speedup_pct"])
            for r in data
            if r["precision"] == precision
        }
        for model_index, (model, label, color) in enumerate(models):
            values = [lookup[(model, batch)] for batch in concurrency]
            offset = (model_index - 1) * height
            bars = ax.barh(y + offset, values, height=height, color=color)
            if ax is axes[0]:
                handles.append(Patch(color=color, label=label))
            for bar, value in zip(bars, values):
                if precision == "bf16" and abs(value) < 1:
                    continue
                pad = 0.25 if precision == "bf16" else 1.2
                ax.text(
                    value + pad if value >= 0 else value - pad,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=9.5,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.axvline(0, color=INK, linestyle=(0, (4, 4)), linewidth=1.2)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks, [f"{tick:g}%" for tick in ticks])
        ax.set_yticks(y, concurrency)
        ax.set_ylim(len(concurrency) - 0.55, -0.55)
        ax.set_xlabel("Whole-request speedup", labelpad=8)
        ax.set_ylabel("Concurrent requests", labelpad=8)
        ax.set_title(heading, fontsize=15, fontweight="bold", loc="left", pad=12)

        all_values = [float(r["whole_request_speedup_pct"]) for r in data if r["precision"] == precision]
        print(f"vLLM {precision.upper()}: range={min(all_values):.1f}%–{max(all_values):.1f}%")

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.86), ncol=3, frameon=False)
    fig.text(
        0.04,
        0.025,
        "BF16 baseline: ATen + Triton · MXFP8 baseline: ATen · treatment adds FlyDSL · TP = 1",
        fontsize=11,
    )
    save(fig, "flydsl-vllm-e2e-results")


if __name__ == "__main__":
    architecture()
    kernel_scheduling()
    dense_gemm()
    mxfp_gemm()
    grouped_gemm()
    rmsnorm()
    topk()
    vllm_e2e()
