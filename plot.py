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
}
MARKERS = {
    "brute": "o",
    "std": "^",
    "avx2": "s",
    "treap": "D",
    "splay": "v",
    "sqrt": "X",
}


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
    data = {m: {} for m in ["brute", "std", "avx2", "treap", "splay", "sqrt"]}
    for i in range(len(df["mode"])):
        if df["mode"][i] == mode:
            data[df["method"][i]][df[key][i]] = df["val"][i]
    return Table(xs, data)


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
    sq = sub["sqrt"]
    ax.plot(xs, bb, marker="o", markersize=4, linewidth=1.5, color="#d62728",
            label="brute best (scalar/std/AVX2)")
    ax.plot(xs, tb, marker="s", markersize=4, linewidth=1.5, color="#1f77b4",
            label="tree best (treap/splay)")
    ax.plot(xs, sq, marker="X", markersize=4, linewidth=1.5, color="#8c564b",
            label="sqrt (块状链表)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("L (reversal length)" if ylabel.startswith("ns") else "n (array size)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    cb = crossover(xs, bb, tb)
    note = f"{tag}: brute ≤ {cb[0]} / tree ≥ {cb[1]}" if cb else f"{tag}: no brute/tree crossover"
    ax.annotate(note, xy=(0.03, 0.88), xytext=(0.03, 0.88),
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


def main():
    cpp = load("results.csv")
    rs = load("results_rs.csv")
    lc, lr = pivot(cpp, "length"), pivot(rs, "length")
    bc, br = pivot(cpp, "batch"), pivot(rs, "batch")

    # length: 单次反转 per-op 成本
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, lc, "ns / op", "C++23 — 单次区间反转成本 (N=2^20)")
    fig.tight_layout()
    fig.savefig("cpp23_length.webp", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, lr, "ns / op", "Rust — 单次区间反转成本 (N=2^20)")
    fig.tight_layout()
    fig.savefig("rust_length.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cb1 = draw_best(axes[0], lc, "ns / op", "C++23 best", "C++23")
    cb2 = draw_best(axes[1], lr, "ns / op", "Rust best", "Rust")
    fig.suptitle("Brute best vs balanced-tree best vs sqrt (single reversal)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("best_length.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, m in zip(axes.flat, ["brute", "std", "avx2", "treap", "splay", "sqrt"]):
        ratio_axes(ax, lc, lr, m, f"{m} C++/Rust ratio (length)")
    fig.suptitle("C++ vs Rust — 单次反转 per-op 比值", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("cpp_vs_rust_length.webp", dpi=150)
    plt.close(fig)

    # batch: 完整流水线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, bc, "ms / run", "C++23 — 5000 次随机反转 + 输出 (ms)")
    fig.tight_layout()
    fig.savefig("cpp23_batch.webp", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    draw_all(ax, br, "ms / run", "Rust — 5000 次随机反转 + 输出 (ms)")
    fig.tight_layout()
    fig.savefig("rust_batch.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cb3 = draw_best(axes[0], bc, "ms / run", "C++23 best", "C++23")
    cb4 = draw_best(axes[1], br, "ms / run", "Rust best", "Rust")
    fig.suptitle("Brute best vs balanced-tree best vs sqrt (5000 random reversals + output)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("best_batch.webp", dpi=150)
    plt.close(fig)

    print("== length 模式临界点（L：反转长度，N=2^20） ==")
    for tag, sub in [("C++23", lc), ("Rust", lr)]:
        cb = crossover(sub.index, brute_best(sub), tree_best(sub))
        print(f"{tag}: brute_best vs tree_best -> {cb}")
        sq_cross = crossover(sub.index, brute_best(sub), sub["sqrt"])
        print(f"{tag}: brute_best vs sqrt -> {sq_cross}")
    print("== batch 模式临界点（n：数组大小，m=5000） ==")
    for tag, sub in [("C++23", bc), ("Rust", br)]:
        cb = crossover(sub.index, brute_best(sub), tree_best(sub))
        print(f"{tag}: brute_best vs tree_best -> {cb}")
        sq_cross = crossover(sub.index, brute_best(sub), sub["sqrt"])
        print(f"{tag}: brute_best vs sqrt -> {sq_cross}")


if __name__ == "__main__":
    main()
