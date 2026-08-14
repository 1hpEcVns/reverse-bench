#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "brute": "#d62728",
    "std": "#ff7f0e",
    "avx2": "#2ca02c",
    "treap": "#1f77b4",
    "splay": "#9467bd",
    "sqrt": "#8c564b",
    "sqrt_bitset": "#17becf",
    "sqrt_bitset2": "#bcbd22",
}
MARKERS = {
    "brute": "o",
    "std": "^",
    "avx2": "s",
    "treap": "D",
    "splay": "v",
    "sqrt": "X",
    "sqrt_bitset": "P",
    "sqrt_bitset2": "h",
}
METHODS = ["brute", "std", "avx2", "treap", "splay", "sqrt", "sqrt_bitset", "sqrt_bitset2"]


def load(path):
    rows = {"mode": [], "n": [], "L": [], "method": [], "val": []}
    with open(path) as f:
        for line in f:
            mode, n, L, method, val = line.strip().split(",")
            rows["mode"].append(mode)
            rows["n"].append(int(n))
            rows["L"].append(int(L))
            rows["method"].append(method)
            rows["val"].append(float(val))
    return rows


class Table:
    """轻量数据表：index + 每列 numpy 数组，替代 pandas DataFrame。"""

    def __init__(self, xs, data):
        self.index = np.asarray(xs, dtype=np.int64)
        self.cols = {
            col: np.array([d[x] for x in xs], dtype=np.float64)
            for col, d in data.items()
        }

    @property
    def columns(self):
        return list(self.cols.keys())

    def __getitem__(self, col):
        return self.cols[col]

    def min(self, cols):
        return np.minimum.reduce([self.cols[c] for c in cols])


def pivot(df, mode):
    key = "n" if mode == "batch" else "L"
    xs = sorted({df[key][i] for i in range(len(df["mode"])) if df["mode"][i] == mode})
    data = {m: {} for m in METHODS}
    for i in range(len(df["mode"])):
        if df["mode"][i] == mode:
            data[df["method"][i]][df[key][i]] = df["val"][i]
    return Table(xs, data)


def pivot_bsweep(df):
    ns = sorted({df["n"][i] for i in range(len(df["mode"])) if df["mode"][i] == "bsweep"})
    bs = sorted({df["L"][i] for i in range(len(df["mode"])) if df["mode"][i] == "bsweep"})
    out = {}
    for n in ns:
        data = {m: {} for m in ["sqrt", "sqrt_bitset", "sqrt_bitset2"]}
        for i in range(len(df["mode"])):
            if df["mode"][i] == "bsweep" and df["n"][i] == n:
                data[df["method"][i]][df["L"][i]] = df["val"][i]
        out[n] = Table(bs, data)
    return out


def brute_best(sub):
    return sub.min(["brute", "std", "avx2"])


def tree_best(sub):
    return sub.min(["treap", "splay"])


def crossover(xs, a, b):
    """最后一个 a 赢的 x / 第一个 b 赢的 x"""
    last = first = None
    for i in range(len(xs) - 1):
        if a[i] < b[i] and not (a[i + 1] < b[i + 1]):
            last, first = int(xs[i]), int(xs[i + 1])
    if last is None:
        return None
    return last, first


def draw_all(ax, sub, ylabel, title):
    for m in sub.columns:
        ax.plot(sub.index, sub[m], marker=MARKERS[m], markersize=4, linewidth=1.3,
                color=COLORS[m], label=m)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("L (reversal length)" if ylabel.startswith("ns") else "n (array size)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")


def draw_best(ax, sub, ylabel, title, tag):
    xs = sub.index
    bb = brute_best(sub)
    tb = tree_best(sub)
    ax.plot(xs, bb, marker="o", markersize=4, linewidth=1.5, color="#d62728",
            label="brute best (scalar/std/AVX2)")
    ax.plot(xs, tb, marker="s", markersize=4, linewidth=1.5, color="#1f77b4",
            label="tree best (treap/splay)")
    ax.plot(xs, sub["sqrt"], marker="X", markersize=4, linewidth=1.5, color="#8c564b",
            label="sqrt (lazy)")
    ax.plot(xs, sub["sqrt_bitset"], marker="P", markersize=4, linewidth=1.5,
            color="#17becf", linestyle="--", label="sqrt_bitset (SIMD tag toggle)")
    ax.plot(xs, sub["sqrt_bitset2"], marker="h", markersize=4, linewidth=1.5,
            color="#bcbd22", linestyle="-.", label="sqrt_bitset2 (2-level B-tree)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("L (reversal length)" if ylabel.startswith("ns") else "n (array size)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    cb = crossover(xs, bb, tb)
    note = f"{tag}: brute <= {cb[0]} / tree >= {cb[1]}" if cb else f"{tag}: no brute/tree crossover"
    ax.annotate(note, xy=(0.03, 0.82), xytext=(0.03, 0.82),
                textcoords="axes fraction", fontsize=9, color="#1f77b4")
    return cb


def ratio_axes(ax, sub_cpp, sub_rs, method, title):
    xs = sub_cpp.index
    r = sub_cpp[method] / sub_rs[method]
    ax.plot(xs, r, marker="o", markersize=4, linewidth=1.3, color=COLORS[method])
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_title(title)
    ax.set_xlabel("L")
    ax.set_ylabel("C++ / Rust")
    ax.grid(True, which="both", alpha=0.25)
    ax.annotate("C++ faster" if r[-1] > 1 else "Rust faster",
                xy=(0.03, 0.9), xytext=(0.03, 0.9),
                textcoords="axes fraction", fontsize=8, color="gray")


def draw_bsweep(ax, sub, title):
    xs = sub.index
    for m in ["sqrt", "sqrt_bitset", "sqrt_bitset2"]:
        ax.plot(xs, sub[m], marker=MARKERS[m], markersize=5, linewidth=1.5,
                color=COLORS[m], label=m)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1] + [64, 128, 256, 512, 1024, 2048, 4096])
    ax.set_xticklabels(["auto"] + ["64", "128", "256", "512", "1k", "2k", "4k"])
    ax.set_title(title)
    ax.set_xlabel("block size B")
    ax.set_ylabel("ms / run (5000 ops + output)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


def main():
    cpp = load("results.csv")
    rs = load("results_rs.csv")
    lc, lr = pivot(cpp, "length"), pivot(rs, "length")
    bc, br = pivot(cpp, "batch"), pivot(rs, "batch")
    bsc, bsr = pivot_bsweep(cpp), pivot_bsweep(rs)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, lc, "ns / op", "C++23 - single reversal (N=2^20)")
    fig.tight_layout()
    fig.savefig("cpp23_length.webp", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, lr, "ns / op", "Rust - single reversal (N=2^20)")
    fig.tight_layout()
    fig.savefig("rust_length.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    draw_best(axes[0], lc, "ns / op", "C++23 best", "C++23")
    draw_best(axes[1], lr, "ns / op", "Rust best", "Rust")
    fig.suptitle("Brute best vs tree best vs sqrt / sqrt_bitset (single reversal)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("best_length.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, m in zip(axes.flat, ["brute", "std", "avx2", "treap", "splay", "sqrt"]):
        ratio_axes(ax, lc, lr, m, f"{m} C++/Rust ratio (length)")
    fig.suptitle("C++ vs Rust - per-op ratio", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("cpp_vs_rust_length.webp", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, bc, "ms / run", "C++23 - 5000 random reversals + output (ms)")
    fig.tight_layout()
    fig.savefig("cpp23_batch.webp", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, br, "ms / run", "Rust - 5000 random reversals + output (ms)")
    fig.tight_layout()
    fig.savefig("rust_batch.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    draw_best(axes[0], bc, "ms / run", "C++23 best", "C++23")
    draw_best(axes[1], br, "ms / run", "Rust best", "Rust")
    fig.suptitle("Brute best vs tree best vs sqrt / sqrt_bitset (5000 ops + output)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("best_batch.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    draw_bsweep(axes[0][0], bsc[1 << 16], "C++23 block-size sweep, n=2^16")
    draw_bsweep(axes[0][1], bsr[1 << 16], "Rust block-size sweep, n=2^16")
    draw_bsweep(axes[1][0], bsc[1 << 20], "C++23 block-size sweep, n=2^20")
    draw_bsweep(axes[1][1], bsr[1 << 20], "Rust block-size sweep, n=2^20")
    fig.suptitle("Block-size sweep: sqrt / sqrt_bitset / sqrt_bitset2", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("bsweep_batch.webp", dpi=150)
    plt.close(fig)

    print("== length crossover (L, N=2^20) ==")
    for tag, sub in [("C++23", lc), ("Rust", lr)]:
        print(f"{tag}: brute_best vs tree_best -> {crossover(sub.index, brute_best(sub), tree_best(sub))}")
        print(f"{tag}: brute_best vs sqrt -> {crossover(sub.index, brute_best(sub), sub['sqrt'])}")
        print(f"{tag}: brute_best vs sqrt_bitset -> {crossover(sub.index, brute_best(sub), sub['sqrt_bitset'])}")
        print(f"{tag}: brute_best vs sqrt_bitset2 -> {crossover(sub.index, brute_best(sub), sub['sqrt_bitset2'])}")
    print("== batch crossover (n, m=5000) ==")
    for tag, sub in [("C++23", bc), ("Rust", br)]:
        print(f"{tag}: brute_best vs tree_best -> {crossover(sub.index, brute_best(sub), tree_best(sub))}")
        print(f"{tag}: brute_best vs sqrt -> {crossover(sub.index, brute_best(sub), sub['sqrt'])}")
        print(f"{tag}: brute_best vs sqrt_bitset -> {crossover(sub.index, brute_best(sub), sub['sqrt_bitset'])}")
        print(f"{tag}: brute_best vs sqrt_bitset2 -> {crossover(sub.index, brute_best(sub), sub['sqrt_bitset2'])}")
    print("== bsweep best B (ms, 5000 ops + output) ==")
    for tag, d in [("C++23", bsc), ("Rust", bsr)]:
        for n in [1 << 16, 1 << 20]:
            for m in ["sqrt", "sqrt_bitset", "sqrt_bitset2"]:
                sub = d[n]
                best_i = int(np.argmin(sub[m]))
                print(f"{tag} n={n}: {m} best B={sub.index[best_i]} ({sub[m][best_i]:.3f} ms)")


if __name__ == "__main__":
    main()
