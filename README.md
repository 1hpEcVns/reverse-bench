# 区间反转（range reverse）Benchmark：SIMD vs Splay / Treap / 块状链表

对同一个「区间反转」操作，对比 7 种实现，C++23 与 Rust edition 2024 各一份。
核心问题：**reverse 是不是 SIMD 友好的？数据结构（splay/treap/块状链表）
在什么规模下反超暴力？块状链表的块长怎么调？懒标记能不能用 mm256/bitset
思路优化？**

## 方法

| 方法 | 说明 | 单次复杂度 |
| --- | --- | --- |
| `brute` | 手写标量 `while (l < r) swap` | O(len) |
| `std` | C++ `std::reverse` / Rust `slice::reverse` | O(len) |
| `avx2` | 手写 AVX2：两端各 load 一个 `__m256i`，`vpermd` 反转 8 个 u32 后 store | O(len/8)，带宽受限 |
| `treap` | FHQ 隐式 treap + 懒反转标记，arena u32 下标 | O(log n) |
| `splay` | 隐式 splay + 懒反转标记（哨兵头尾，rank l / rank r+2 夹出区间） | O(log n) 摊还 |
| `sqrt` | 懒标记块状链表：整块只翻懒标记 + 块列表整体反转，两端散块实翻转；**底层散块/物化用 AVX2**；块长可调，过小合并、过大拆分 | O(B + n/B) |
| `sqrt_bitset` | **bitset 思想优化版**：懒标记打包进 u32 块描述符最高位，中间整块区间翻转 = `vpxor 0x80000000`（8 块/指令），块顺序反转 = 同一个 `reverse_avx2`（vpermd）作用于 desc，标记随块走；块本体放 arena + 空闲回收 | O(B + n/B)，常数更小 |
| `sqrt_bitset2` | **两层版（bitset 套 bitset = 高度 2 的序列 B 树）**：内层块描述符（pool_id|REV）+ 外层超块描述符（sb_id|ORD|CONT），中间整块区间 = 外层 vpermd 反转超块顺序 + vpxor 翻两个标记位；超块 >2W 拆、<W/2 并；块级小块合并保持块数有界 | O(n/B/W + W) 定位，常数大 |

数据元素统一为 `u32`。

三种模式：

| 模式 | 测什么 | 输出 |
| --- | --- | --- |
| `length` | N=2^20 固定，扫反转长度 L=32..2^20，每 (方法,L) 校准到约 3 ms/轮，再跑 9 轮取中位数 | ns/op |
| `batch` | 扫 n=1024..2^20，固定 5000 个均匀随机区间反转 + 最终整序列加权 checksum（含建树/初始化） | ms/整条流水线 |
| `bsweep` | n=2^16 / 2^20，固定 5000 个随机反转，扫块长 B=64..4096（0=auto） | ms/整条流水线 |

编译/测量方法与 fenwick_bench 一致：

- C++23：`g++ -O3 -march=native -std=c++23`
- Rust：`rustc --edition=2024 -O -C target-cpu=native`
- 固定单 P-core（`taskset -c 0`），C++ 与 Rust 串行跑，避免抢核
- RNG 两侧同一 splitmix64；每个 (方法, L/n) 先与暴力**全数组逐元素对拍**
  （length 8 个操作、batch 200 个操作），不一致直接 abort
- 计时用「按位置的加权 checksum」（位置相关权重来自 RNG）做 black-box：
  普通求和是反转不变量，编译器会把它当常量把整段反转优化掉
  （实测 splay 整段反转被优化后出现 4 ns/op 的假数据）

## 汇编验证：反转循环有没有自动向量化？

`bash asm_check.sh` 把最小反转循环分别用 GCC 15.3 / LLVM 1.97 编译
（`-O3 -march=native` / `-O -C target-cpu=native`），统计反转 shuffle 指令：

| 实现 | 指令 | GCC# | LLVM# |
| --- | --- | ---: | ---: |
| 标量交换循环 | `vpermd` / `vpermps`（32b 反转 shuffle） | 4 | 6 |
| 标量交换循环 | `vpshufd`（128b shuffle） | 2 | 2 |
| `std::reverse` / `slice::reverse` | `vpermd` / `vpermps` | 4 | 6 |
| `std::reverse` / `slice::reverse` | `vpshufd` | 2 | 2 |
| 手写 AVX2 | `vpermd` / `vpermps` | 4 | 6 |
| 手写 AVX2 | `vpshufd` | 2 | 2 |

结论：**两个编译器都能把最朴素的标量交换循环自动向量化成 32 位反转 shuffle**
（GCC 用 `vpermd`，LLVM 用同功能的 `vpermps`），reverse 本质上是一对
`load → 反转 lane → store`，和 memcpy 同级别的访存模式。实测差距来自
benchmark 上下文里的随机边界/尾部和循环形态，手写 AVX2 仍有明显收益
（见下）。

## 本机结果（i9-13950HX，g++ 15.3，rustc 1.97）

### batch 模式：5000 次随机反转 + 输出（ms/轮）

| n | 方法 | C++23 | Rust | Rust/C++ |
| ---: | --- | ---: | ---: | ---: |
| 1024 | brute | 0.36 | 0.37 | 1.03 |
| 1024 | std | 0.14 | 0.14 | 0.96 |
| 1024 | avx2 | 0.13 | 0.13 | 1.01 |
| 1024 | treap | 1.89 | 1.92 | 1.02 |
| 1024 | splay | 1.58 | 1.63 | 1.03 |
| 1024 | sqrt | 0.74 | 0.96 | 1.30 |
| 1024 | sqrt_bitset | 0.74 | 1.02 | 1.38 |
| 1024 | sqrt_bitset2 | 0.92 | 1.16 | 1.26 |
| 65536 | brute | 23.97 | 24.18 | 1.01 |
| 65536 | std | 5.91 | 5.98 | 1.01 |
| 65536 | avx2 | 6.04 | 6.20 | 1.03 |
| 65536 | treap | 5.77 | 5.88 | 1.02 |
| 65536 | splay | 3.52 | 3.92 | 1.11 |
| 65536 | sqrt | 2.58 | 2.54 | 0.98 |
| 65536 | sqrt_bitset | **2.39** | **2.54** | 1.07 |
| 65536 | sqrt_bitset2 | 3.28 | 3.80 | 1.16 |
| 1048576 | brute | 391.4 | 395.8 | 1.01 |
| 1048576 | std | 126.7 | 127.3 | 1.00 |
| 1048576 | avx2 | 128.7 | 129.4 | 1.01 |
| 1048576 | treap | 42.3 | 40.7 | 0.96 |
| 1048576 | splay | 13.8 | 18.0 | 1.30 |
| 1048576 | sqrt | 8.6 | 6.9 | 0.80 |
| 1048576 | sqrt_bitset | **6.2** | **5.9** | 0.96 |
| 1048576 | sqrt_bitset2 | 8.0 | 9.8 | 1.23 |

临界点（最后一个暴力赢的规模 / 第一个对方赢的规模）：

| 对比 | C++23 | Rust |
| --- | ---: | ---: |
| length：暴力最优 vs 树最优（L，N=2^20） | 8192 / 16384 | 8192 / 16384 |
| length：暴力最优 vs sqrt（L） | 16384 / 32768 | 8192 / 16384 |
| length：暴力最优 vs sqrt_bitset（L） | 8192 / 16384 | 8192 / 16384 |
| batch：暴力最优 vs 树最优（n，m=5000） | 32768 / 65536 | 32768 / 65536 |
| batch：暴力最优 vs sqrt（n） | 16384 / 32768 | 16384 / 32768 |
| batch：暴力最优 vs sqrt_bitset（n） | 16384 / 32768 | 16384 / 32768 |

### length 模式：单次反转 per-op（ns，N=2^20，节选）

| L | brute | std | avx2 | treap | splay | sqrt | sqrt_bitset |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | 3690 | 1129 | 1132 | 1059 | 785 | 1154 | 951 |
| 65536 | 14543 | 4446 | 4429 | 1319 | 807 | 1240 | 987 |
| 1048576 | 238645 | 90640 | 92090 | 118 | **4.4** | 1310 | 553 |

其中 `sqrt_bitset2`（两层版）的对应数值：L=16384 时 1003 ns、L=65536 时
1211 ns、L=2^20（整段）时 **213 ns**——整段反转时它只用外层 16 个超块
描述符做 vpermd + vpxor，比单层 1024 个块描述符快 2.6 倍；但随机区间下
被端部物化/拆块/重定位的开销吃掉了。

### bsweep：块长 B 对分块性能的影响（C++23，ms/轮；A/B = sqrt / sqrt_bitset）

| B | n=2^16 sqrt / bitset | n=2^20 sqrt / bitset |
| ---: | ---: | ---: |
| 64 | 4.55 / 2.42 | 69.76 / 26.64 |
| 128 | 2.93 / **1.87** | 36.44 / 14.29 |
| 256 | **2.51** / 2.01 | 19.51 / 8.68 |
| 512 | 2.85 / 2.58 | 11.49 / 6.36 |
| 1024 | 3.48 / 3.36 | **8.12** / **5.92** |
| 2048 | 4.89 / 4.89 | 8.97 / 7.74 |
| 4096 | 8.24 / 8.38 | 13.98 / 13.46 |
| 0（auto=2√n） | 2.80 / 2.66 | 8.08 / **5.90** |

`sqrt_bitset2` 的 bsweep（C++，ms）：n=2^16 最优 B=auto（3.38），n=2^20
最优 B=128（6.72）。

## 结论

### 1. reverse 非常 SIMD 友好

- 反转一个连续区间 = 两端各一次 load、反转 lane、交叉 store，
  和 memcpy 一样是纯带宽问题，没有数据依赖。
- 手写 AVX2（8×u32/次）在 L=2^20 时 ~92 µs，暴力标量 ~239 µs，
  **约 2.6 倍加速**；有效带宽 ~90 GB/s（大 L）到 ~110 GB/s（L2 内）。
- 两个编译器连朴素标量循环都能自动向量化（`vpermd`/`vpermps`），
  但 benchmark 上下文里手写版仍然显著更快（batch n=1M：129 ms vs 391 ms）。

### 2. 懒标记可以用 mm256/bitset 思路优化，实测再快 ~25–30%

`sqrt_bitset` 的关键改动（这就是你说的 bitset 思想）：

1. **懒标记位图化**：每个块 1 bit，但不是独立位图，而是打包进 u32 块描述符的
   最高位（`desc = pool_id | rev<<31`）。中间整块的区间翻转 = 一次
   `vpxor 0x80000000`，8 个块的标记一条 SIMD 指令，不再逐块 `^= 1`。
2. **标记随块走**：块顺序反转直接对 desc 数组做 `reverse_avx2`（vpermd），
   rev 位跟着块一起移动，不需要独立的 bitset 移位/反转（试过独立位图，
   插入/删除/区间反转都要处理位移动，且「先反转再整体翻转」会让标记串位，
   打包进描述符是最稳的）。
3. **块长调节保留**：>2B 拆、<B/2 合并，B 可扫；arena 槽位用 free-list 回收，
   长时间运行不会膨胀。

实测收益（batch，5000 次随机反转 + 输出）：

- n=2^20：C++ 8.6 → 6.2 ms（**-28%**），Rust 6.9 → 5.9 ms（-14%）；
- n=2^16：C++ 2.51 → 1.87 ms（-25%，B=128 最优）；
- length 单次反转 L=2^20：1375 → 542 ns（整段反转的 desc/szs 都是 u32，
  vpermd + vpxor 一次扫完）；
- B 偏小时收益更大（n=2^20, B=64：69.8 → 26.6 ms），因为小块数多、
  标记翻转从 O(#blocks) 字节操作变成 O(#blocks/8) 向量操作。
- 块长仍然要调：n=2^20 最优 B=1024（5.9 ms），B=64 要 26.6 ms（4.5 倍差距）；
  n=2^16 最优 B=128–256。

### 2b. bitset 套 bitset = 序列 B 树逻辑，但这个规模下不划算

把懒标记再套一层（`sqrt_bitset2`：内层块 + 外层超块，每层都用
`vpermd`/`vpxor`，超块按容量拆/并）——**本质上就是高度 2 的序列 B 树**
（块=叶子，超块=内部节点，节点存 total 做搜索，每层懒反转标记）。实测结论：

- batch n=2^20：C++ 8.0 ms、Rust 9.8 ms，**比单层 sqrt_bitset 慢 27–60%**；
  n=2^16 慢 27–47%。多出来的每步成本：端部超块 materialize、块级拆分、
  每次操作 4–7 次 find、超块/块两级拆并。
- 唯一亮点是整段反转（L=2^20）：213 ns vs 单层 553 ns，外层只有 ~16 个
  描述符，vpermd/vpxor 一次扫完。
- 结论：n ≤ 2^20、块数 ≤ 几千时，**单层平铺 + SIMD 标记位就是最优形态**；
  两层结构要到块数很大（n 到 2^24+ 或 B 很小时）find 的 O(#blocks) 才会
  成为瓶颈，那时才值得把根也换成真指针 B 树。

### 3. 数据结构反超点：长度 ~8K–16K，batch n ~32K–64K

- 单次反转：L ≤ 8192 暴力最优（~570 ns 的 AVX2），L ≥ 16384 树/分块最优
  （splay ~800 ns，sqrt_bitset ~940 ns）。
- 5000 次反转完整流水线：n ≤ 16K 暴力最优，n ≥ 64K 树/块状胜出。
- **batch 最终赢家是 bitset 懒标记块状链表（sqrt_bitset）**：n=2^20 时 6.2 ms，
  比 splay（13.8 ms）、treap（42.3 ms）、暴力最优（127 ms）都快。
- treap 在 batch 里明显慢于 splay：FHQ 的 split×3 + merge×2 路径比 splay
  的两次 kth+两次 splay 更长，且递归调用有额外开销。

### 4. 一个反直觉现象：splay 整段反转会“特化”成 O(1)

length 模式 L=2^20 时 splay 只有 ~4.4 ns/op：第一次操作后树变成
`head → tail → 整棵元素子树`，之后的整段反转退化为 2 次节点访问 + 1 次懒标记
翻转。这不是测量错误，而是 splay 局部性（边界节点被 splay 到根附近）带来的
真实行为；随机区间没有这个特化（L=32~64K 时稳定在 ~550–880 ns）。

### 5. C++ vs Rust

- 暴力三件套几乎完全一致（±5%）；sqrt_bitset 也基本一致（±10% 以内）。
- treap/splay 在 batch n=1M 时 Rust 慢 4–30%，主要是递归和 parent 指针链
  的调度差异。

## 除了暴力，还有哪些数据结构？

实际实现并测了 4 类：

1. **隐式 treap（FHQ）**：O(log n) 期望，好写、可持久化，常数最大。
2. **隐式 splay**：O(log n) 摊还，常数小，对边界区间有局部性红利，迭代实现无递归深度风险。
3. **懒标记块状链表（sqrt / sqrt_bitset）**：O(B + n/B)，实测 batch 场景最快；
   懒标记用 bitset/SIMD 打包后常数再降 25–30%，块长要按 n 调（≈2√n）。
4. **两层块状链表（sqrt_bitset2）**：bitset 套 bitset = 高度 2 的序列 B 树，
   实测这个规模下不如单层（见 2b），只有整段反转占优。
5. ~~无懒标记物理 SIMD 块状链表~~：实测过但已删除——整块也物理反转会让
   单次长区间退化成 O(len) SIMD，batch 打不过暴力（n=2^20 约 145 ms vs 暴力
   127 ms），懒标记才是分块 O(B+n/B) 的核心。

没单独实现、但值得知道的同类：

- **Rope（平衡二叉树/绳）**：本质和隐式 treap 同构（分裂+合并+懒标记），
  常数取决于节点大小；没有懒反转的 rope（如 GNU `__gnu_cxx::rope`）区间反转
  要逐叶重建，会更慢，不值得上。
- **WBLT / 隐式 AVL / 红黑**：和 treap/splay 同族的隐式序列，O(log n)，
  常数在两者之间；没有理由在 CP 里替代 splay。
- **可持久化隐式 treap**：如果还要撤销/版本回溯才需要，单次反转的 COW
  节点数 O(log n)。
- **双向链表 + 懒方向**：整段反转可 O(1)（头尾互换 + 方向位翻转），
  但**区间**反转需要把区间内每个节点的方向翻转——除非再叠一层懒标记结构，
  那就绕回 treap/splay 了。

一句话：小长度用 AVX2 暴力；大量随机区间反转用 **sqrt_bitset**（懒标记
位图化 + SIMD 翻转，块长 ≈ 2√n）或 splay；要持久化/撤销用可持久化 treap。

## 复现

```bash
nix develop            # gcc/rustc/python+matplotlib
make bench-all         # C++23 results.csv + Rust results_rs.csv
make plot              # 重测 + 生成 8 张 webp 图并输出临界点
bash asm_check.sh      # 自动向量化/SIMD 指令检查
```

也可以单独跑某个语言或模式：

```bash
./bench length > /tmp/length.csv      # 只跑 length 模式
./bench batch  > /tmp/batch.csv       # 只跑 batch 模式
./bench bsweep > /tmp/bsweep.csv      # 只跑块长扫描
```

原始数据：`results.csv`（C++23）、`results_rs.csv`（Rust）。

## P3391 【模板】文艺平衡树：最优提交

`p3391_submit.cpp` 用本仓库实测最快的 `sqrt_bitset` 直接提交：

```bash
g++ -O2 -std=c++17 p3391_submit.cpp -o p3391          # 纯标量（通用）
g++ -O2 -mavx2 -std=c++17 p3391_submit.cpp -o p3391   # 有 AVX2 的评测机更快
```

实测（n=m=1e5，i9-13950HX，3 次取最小）：

| 数据形态 | sqrt_bitset 标量 | sqrt_bitset AVX2 | splay（p3391_splay.cpp） |
| --- | ---: | ---: | ---: |
| 随机区间 | 105 ms | **74 ms** | 96 ms |
| 全段反转 [1,n] | 72 ms | 58 ms | **25 ms** |
| 固定 [1,n/2] | 128 ms | 71 ms | **20 ms** |
| 固定 [n/4,3n/4] | 58 ms | 47 ms | **30 ms** |
| 随机小区间 (64) | **48 ms** | 52 ms | 77 ms |

全部远低于 P3391 的时限（通常 1–2 s），两份代码都已与暴力对拍。
随机/小区间数据 `sqrt_bitset` 最快；重复固定边界的测试 splay 靠“边界节点
被 splay 到根”的局部性更快，可按评测数据形态二选一。

## 目录

- `bench.cpp` / `bench.rs`：同构的 7 方法 × 3 模式 benchmark
- `asm_check.sh`：SIMD 指令统计表
- `plot.py`：生成 `*_length.webp`、`*_batch.webp`、`best_*.webp`、`bsweep_batch.webp`、
  `cpp_vs_rust_length.webp`
