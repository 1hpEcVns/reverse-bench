// reverse_bench: 区间反转（range reverse）的 SIMD / 数据结构对比
// 方法：
//   brute  - 手写标量交换循环
//   std    - std::reverse
//   avx2   - 手写 AVX2（vpermd 整向量反向交换）
//   treap  - FHQ 隐式 treap，懒反转标记，O(log n)
//   splay  - 隐式 splay，懒反转标记，O(log n) 摊还
//   sqrt   - 块状链表（unrolled list），整块翻转标记 + 两端散块实翻转
//
// 模式：
//   length - N=2^20 固定，扫反转长度 L（32..2^20），每方法每 L 校准到约 3 ms
//            再跑 9 轮取中位数，输出 ns/op
//   batch  - 扫 n（1024..2^20），固定 5000 个随机区间反转 + 最终输出 checksum，
//            输出整条流水线 ms/轮（含建树/初始化）
//
// 编译：g++ -O3 -march=native -std=c++23 bench.cpp -o bench
// 测量：taskset -c 0 ./bench > results.csv

#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using usize = std::size_t;

static constexpr usize ROUNDS = 9;
static constexpr double TARGET_MS = 3.0;  // 校准目标：约 3 ms / 轮
static constexpr u64 Q_CAP = 8'000'000;
static constexpr u32 OPS = 5000;          // batch 模式的随机反转次数
static constexpr u32 LEN_N = 1u << 20;    // length 模式固定数组大小

// ---------------- RNG（splitmix64，C++/Rust 两侧完全一致） ----------------
static inline u64 splitmix64(u64& x) {
    x += 0x9e3779b97f4a7c15ULL;
    u64 z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline u32 rnd(u64& s, u32 mod) {
    return (u32)((splitmix64(s) >> 32) % mod);
}

// ---------------- 计时 / black box ----------------
static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static inline void black_box_u64(u64 x) {
    asm volatile("" : "+r"(x) : : "memory");
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------- 暴力：标量 / std::reverse / AVX2 ----------------
static inline void reverse_scalar(u32* a, usize l, usize r) {  // [l, r)
    while (l < r) {
        --r;
        u32 t = a[l];
        a[l] = a[r];
        a[r] = t;
        ++l;
    }
}

static inline void reverse_std(u32* a, usize l, usize r) {  // [l, r)
    std::reverse(a + l, a + r);
}

static inline void reverse_avx2(u32* a, usize l, usize r) {  // [l, r)
    const __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    while (l + 8 <= r) {  // 不能用 r - l >= 8：最后一次迭代后 l > r 会无符号下溢
        // 先 load 两端，再 store 两端；区间不足 16 时中间重叠也安全。
        __m256i L = _mm256_loadu_si256((const __m256i*)(a + l));
        __m256i R = _mm256_loadu_si256((const __m256i*)(a + r - 8));
        _mm256_storeu_si256((__m256i*)(a + l),
                            _mm256_permutevar8x32_epi32(R, idx));
        _mm256_storeu_si256((__m256i*)(a + r - 8),
                            _mm256_permutevar8x32_epi32(L, idx));
        l += 8;
        r -= 8;
    }
    reverse_scalar(a, l, r);
}

// ---------------- FHQ 隐式 treap（懒反转） ----------------
struct Treap {
    struct Node {
        u32 l, r;      // 孩子下标，0 = 空
        u32 pri, sz;   // 优先级（小顶堆）、子树大小
        u32 val;
        u8 rev;
    };
    std::vector<Node> p;
    u32 root = 0;
    u64 rng;

    Treap(const u32* a, usize n, u64 seed) : rng(seed) {
        p.reserve(n + 1);
        p.push_back(Node{});  // 下标 0 = 空节点
        std::vector<u32> st;
        st.reserve(n);
        for (usize i = 0; i < n; ++i) {
            u32 x = new_node(a[i]);
            u32 last = 0;
            while (!st.empty() && p[st.back()].pri > p[x].pri) {
                last = st.back();
                st.pop_back();
            }
            if (st.empty())
                root = x;
            else
                p[st.back()].r = x;
            p[x].l = last;
            st.push_back(x);
        }
        // 创建顺序倒序 pull 不行：左孩子下标更小，此时还没被 pull。
        // 改为后序遍历（孩子先于父亲）一次性求 size。
        std::vector<u32> order;
        order.reserve(n);
        std::vector<u32> stk2;
        stk2.reserve(n);
        stk2.push_back(root);
        while (!stk2.empty()) {
            u32 t = stk2.back();
            stk2.pop_back();
            order.push_back(t);
            if (p[t].l) stk2.push_back(p[t].l);
            if (p[t].r) stk2.push_back(p[t].r);
        }
        for (auto it = order.rbegin(); it != order.rend(); ++it) pull(*it);
    }

    u32 new_node(u32 v) {
        Node nd{};
        nd.pri = (u32)(splitmix64(rng) >> 32);
        nd.sz = 1;
        nd.val = v;
        p.push_back(nd);
        return (u32)p.size() - 1;
    }

    inline u32 sz(u32 x) { return x ? p[x].sz : 0; }

    inline void pull(u32 x) { p[x].sz = 1 + sz(p[x].l) + sz(p[x].r); }

    inline void push(u32 x) {
        if (x && p[x].rev) {
            std::swap(p[x].l, p[x].r);
            p[p[x].l].rev ^= 1;
            p[p[x].r].rev ^= 1;
            p[x].rev = 0;
        }
    }

    void split(u32 t, u32 k, u32& a, u32& b) {  // 前 k 个给 a
        if (!t) {
            a = b = 0;
            return;
        }
        push(t);
        u32 ls = sz(p[t].l);
        if (k <= ls) {
            split(p[t].l, k, a, p[t].l);
            b = t;
            pull(t);
        } else {
            split(p[t].r, k - ls - 1, p[t].r, b);
            a = t;
            pull(t);
        }
    }

    u32 merge(u32 a, u32 b) {
        if (!a || !b) return a ? a : b;
        if (p[a].pri < p[b].pri) {
            push(a);
            p[a].r = merge(p[a].r, b);
            pull(a);
            return a;
        } else {
            push(b);
            p[b].l = merge(a, p[b].l);
            pull(b);
            return b;
        }
    }

    void reverse(u32 l, u32 r) {  // 元素下标 [l, r]（含端点）
        u32 a, b, c;
        split(root, l, a, b);
        split(b, r - l + 1, b, c);
        p[b].rev ^= 1;
        root = merge(a, merge(b, c));
    }

    void collect(std::vector<u32>& out) {
        out.clear();
        u32 x = root;
        std::vector<u32> stk;
        stk.reserve(64);
        while (x || !stk.empty()) {
            while (x) {
                push(x);
                stk.push_back(x);
                x = p[x].l;
            }
            x = stk.back();
            stk.pop_back();
            out.push_back(p[x].val);
            x = p[x].r;
        }
    }
};

// ---------------- 隐式 splay（懒反转） ----------------
struct Splay {
    struct Node {
        u32 l, r, p;
        u32 sz;
        u32 val;
        u8 rev;
    };
    std::vector<Node> p;  // 1..n+2：哨兵头 + n 个元素 + 哨兵尾；0 = 空
    u32 root = 0;
    u32 n = 0;

    Splay(const u32* a, usize n, u64) {
        this->n = (u32)n;
        p.resize(n + 3);
        for (u32 i = 1; i <= n + 2; ++i)
            p[i].val = (i >= 2 && i <= n + 1) ? a[i - 2] : 0;
        root = build(1, (u32)n + 2, 0);
    }

    u32 build(u32 L, u32 R, u32 par) {
        if (L > R) return 0;
        u32 m = (L + R) >> 1;
        p[m].p = par;
        p[m].l = build(L, m - 1, m);
        p[m].r = build(m + 1, R, m);
        pull(m);
        return m;
    }

    inline void pull(u32 x) { p[x].sz = 1 + p[p[x].l].sz + p[p[x].r].sz; }

    inline void push(u32 x) {
        if (x && p[x].rev) {
            std::swap(p[x].l, p[x].r);
            p[p[x].l].rev ^= 1;
            p[p[x].r].rev ^= 1;
            p[x].rev = 0;
        }
    }

    void rotate(u32 x) {
        u32 y = p[x].p;
        u32 z = p[y].p;
        if (p[y].l == x) {
            p[y].l = p[x].r;
            if (p[x].r) p[p[x].r].p = y;
            p[x].r = y;
        } else {
            p[y].r = p[x].l;
            if (p[x].l) p[p[x].l].p = y;
            p[x].l = y;
        }
        p[y].p = x;
        p[x].p = z;
        if (z) {
            if (p[z].l == y)
                p[z].l = x;
            else
                p[z].r = x;
        }
        pull(y);
        pull(x);
    }

    // 前置条件：root..x 路径上的 rev 都已 push（调用方先 kth）
    void splay(u32 x, u32 goal) {
        while (p[x].p != goal) {
            u32 y = p[x].p;
            u32 z = p[y].p;
            if (z != goal) {
                if ((p[y].l == x) == (p[z].l == y))
                    rotate(y);
                else
                    rotate(x);
            }
            rotate(x);
        }
        if (!goal) root = x;
    }

    u32 kth(u32 k) {  // 所有 n+2 个节点中的第 k 个（0-based）
        u32 x = root;
        for (;;) {
            push(x);
            u32 ls = p[p[x].l].sz;
            if (k < ls)
                x = p[x].l;
            else if (k == ls)
                return x;
            else {
                k -= ls + 1;
                x = p[x].r;
            }
        }
    }

    void reverse(u32 l, u32 r) {  // 元素下标 [l, r]（含端点）
        // 哨兵头 rank 0，元素 i 的 rank = i+1，哨兵尾 rank = n+1。
        // 翻转 [l, r] 需要 splay rank l（区间前一个节点）和 rank r+2（区间后一个）。
        u32 A = kth(l);
        splay(A, 0);
        u32 B = kth(r + 2);  // 先 splay A，再从 A 向下推路径，保证 B 路径已 push
        splay(B, root);
        u32 mid = p[p[root].r].l;
        p[mid].rev ^= 1;
    }

    void collect(std::vector<u32>& out) {
        out.clear();
        u32 x = root;
        std::vector<u32> stk;
        stk.reserve(64);
        while (x || !stk.empty()) {
            while (x) {
                push(x);
                stk.push_back(x);
                x = p[x].l;
            }
            x = stk.back();
            stk.pop_back();
            // 跳过两个哨兵节点（下标 1 和 n+2）
            if (x != 1 && x != n + 2) out.push_back(p[x].val);
            x = p[x].r;
        }
    }
};

// ---------------- 块状链表（sqrt decomposition） ----------------
struct Sqrt {
    struct Block {
        std::vector<u32> v;
        u8 rev;
    };
    std::vector<Block> b;
    u32 B;

    Sqrt(const u32* a, usize n, u64, u32 block_size = 0) {
        B = block_size ? block_size
                       : std::max(64u, std::min(1024u, (u32)(std::sqrt((double)n) * 2.0)));
        for (usize i = 0; i < n;) {
            Block bl;
            bl.rev = 0;
            usize e = std::min(n, i + B);
            bl.v.assign(a + i, a + e);
            b.push_back(std::move(bl));
            i = e;
        }
    }

    std::pair<u32, u32> find(u32 k) {  // 逻辑位置 -> (块, 块内偏移)
        for (u32 i = 0; i < (u32)b.size(); ++i) {
            u32 s = (u32)b[i].v.size();
            if (k < s) return {i, k};
            k -= s;
        }
        return {0, 0};
    }

    inline void materialize(u32 i) {
        if (b[i].rev) {
            reverse_avx2(b[i].v.data(), 0, b[i].v.size());
            b[i].rev = 0;
        }
    }

    void split_block(u32 i) {
        u32 s = (u32)b[i].v.size();
        if (s <= 2 * B) return;
        u32 half = s / 2;
        Block nb;
        nb.rev = 0;
        nb.v.assign(b[i].v.begin() + half, b[i].v.end());
        b[i].v.resize(half);
        b.insert(b.begin() + i + 1, std::move(nb));
    }

    // 块长调节：过小的块与邻居合并（合并后仍超 2B 由 split_block 处理）。
    void merge_small(u32 i) {
        if (b[i].v.size() >= B / 2) return;
        if (i > 0 && b[i - 1].v.size() + b[i].v.size() <= 2 * B) {
            materialize(i - 1);  // 邻居可能是带懒标记的中间块，先物理化
            materialize(i);
            b[i - 1].v.insert(b[i - 1].v.end(), b[i].v.begin(), b[i].v.end());
            b.erase(b.begin() + i);
            return;
        }
        if (i + 1 < b.size() && b[i].v.size() + b[i + 1].v.size() <= 2 * B) {
            materialize(i + 1);
            materialize(i);
            b[i].v.insert(b[i].v.end(), b[i + 1].v.begin(), b[i + 1].v.end());
            b.erase(b.begin() + i + 1);
        }
    }

    void rebalance(u32 i) {
        if (b[i].v.size() > 2 * B)
            split_block(i);
        else
            merge_small(i);
    }

    void reverse(u32 l, u32 r) {  // 元素下标 [l, r]（含端点）
        auto [bi, oi] = find(l);
        auto [bj, oj] = find(r);
        if (bi == bj) {
            materialize(bi);
            reverse_avx2(b[bi].v.data(), oi, (usize)oj + 1);
            return;
        }
        materialize(bi);
        materialize(bj);
        u32 nr = oj + 1;  // 右散块参与翻转的前缀长度
        std::vector<u32> rp(b[bj].v.begin(), b[bj].v.begin() + nr);
        reverse_avx2(rp.data(), 0, rp.size());
        std::vector<u32> lp(b[bi].v.begin() + oi, b[bi].v.end());
        reverse_avx2(lp.data(), 0, lp.size());
        b[bi].v.resize(oi);
        b[bi].v.insert(b[bi].v.end(), rp.begin(), rp.end());
        std::vector<u32> tail(b[bj].v.begin() + oj + 1, b[bj].v.end());
        b[bj].v = std::move(lp);
        b[bj].v.insert(b[bj].v.end(), tail.begin(), tail.end());
        // 中间整块：先反转块列表顺序，再逐块翻转（懒标记）
        std::reverse(b.begin() + bi + 1, b.begin() + bj);
        for (u32 i = bi + 1; i < bj; ++i) b[i].rev ^= 1;
        // 先处理大下标，避免增删块导致另一个下标失效
        rebalance(bj);
        rebalance(bi);
    }

    void collect(std::vector<u32>& out) const {
        out.clear();
        for (const Block& bl : b) {
            if (bl.rev)
                out.insert(out.end(), bl.v.rbegin(), bl.v.rend());
            else
                out.insert(out.end(), bl.v.begin(), bl.v.end());
        }
    }
};

// ---------------- 物理 SIMD 块状链表（无懒标记） ----------------
// 上层（整块）与下层（散块）用同一个 reverse_avx2 内核，块内容始终物理有序；
// 整块反转 = 逐块 SIMD 就地反转 + 块列表顺序反转（O(#blocks) 指针搬移）。
// ---------------- bitset 思想懒标记块状链表（sqrt_bitset） ----------------
// 懒标记打包进 u32 块描述符的最高位（desc = pool_id | rev<<31）：
//  - 中间整块的区间翻转 = 一次 vpxor 0x80000000（每 8 块一条 SIMD 指令）；
//  - 块顺序反转 = 同一个 reverse_avx2（vpermd）直接作用于 desc，标记随块走；
//  - 块本体放 arena（pool 只增不减），desc/szs 是 u32 数组，split/merge 只是
//    vector insert/erase，不再有独立的 bit insert/remove。
struct SqrtBitset {
    struct Block {
        std::vector<u32> v;
    };
    static constexpr u32 REV = 0x80000000u;
    std::vector<Block> pool;  // arena：block id = 下标，删除后留空位
    std::vector<u32> desc;    // 逻辑块顺序：pool_id | (懒反转 << 31)
    std::vector<u32> szs;     // 与 desc 平行的块大小（find 用，避免二次间接）
    std::vector<u32> free_ids;  // 合并后回收的 pool 槽位
    u32 B;

    SqrtBitset(const u32* a, usize n, u64, u32 block_size = 0) {
        B = block_size ? block_size
                       : std::max(64u, std::min(1024u, (u32)(std::sqrt((double)n) * 2.0)));
        for (usize i = 0; i < n;) {
            Block bl;
            usize e = std::min(n, i + B);
            bl.v.assign(a + i, a + e);
            pool.push_back(std::move(bl));
            desc.push_back((u32)pool.size() - 1);
            szs.push_back((u32)(e - i));
            i = e;
        }
    }

    std::pair<u32, u32> find(u32 k) const {  // 逻辑位置 -> (位置, 块内偏移)
        for (u32 i = 0; i < (u32)desc.size(); ++i) {
            u32 s = szs[i];
            if (k < s) return {i, k};
            k -= s;
        }
        return {0, 0};
    }

    inline bool bit_test(u32 pos) const { return desc[pos] & REV; }

    // 区间懒标记翻转：整块区间 = vpxor 0x80000000（bitset 思想，8 块/指令）
    void toggle_range(u32 l, u32 r) {
        const __m256i m = _mm256_set1_epi32((int)REV);
        u32 i = l;
        for (; i + 8 <= r; i += 8) {
            __m256i x = _mm256_loadu_si256((const __m256i*)(desc.data() + i));
            x = _mm256_xor_si256(x, m);
            _mm256_storeu_si256((__m256i*)(desc.data() + i), x);
        }
        for (; i < r; ++i) desc[i] ^= REV;
    }

    inline void materialize(u32 pos) {
        if (desc[pos] & REV) {
            reverse_avx2(pool[desc[pos] & ~REV].v.data(), 0, szs[pos]);
            desc[pos] &= ~REV;
        }
    }

    void split_block(u32 pos) {
        u32 s = szs[pos];
        if (s <= 2 * B) return;
        u32 half = s / 2;
        u32 id = desc[pos] & ~REV;  // 端块必已 materialize（rev=0）
        Block nb;
        nb.v.assign(pool[id].v.begin() + half, pool[id].v.end());
        pool[id].v.resize(half);
        szs[pos] = half;
        u32 nid;
        if (!free_ids.empty()) {
            nid = free_ids.back();
            free_ids.pop_back();
            pool[nid].v = std::move(nb.v);
        } else {
            pool.push_back(std::move(nb));
            nid = (u32)pool.size() - 1;
        }
        desc.insert(desc.begin() + pos + 1, nid);
        szs.insert(szs.begin() + pos + 1, s - half);
    }

    void merge_small(u32 pos) {
        u32 s = szs[pos];
        if (s >= B / 2) return;
        if (pos > 0 && szs[pos - 1] + s <= 2 * B) {
            materialize(pos - 1);
            materialize(pos);
            u32 idL = desc[pos - 1] & ~REV;
            u32 idR = desc[pos] & ~REV;
            Block& L = pool[idL];
            Block& R = pool[idR];
            L.v.insert(L.v.end(), R.v.begin(), R.v.end());
            szs[pos - 1] += s;
            std::vector<u32>().swap(R.v);
            free_ids.push_back(idR);
            desc.erase(desc.begin() + pos);
            szs.erase(szs.begin() + pos);
            return;
        }
        if (pos + 1 < desc.size() && s + szs[pos + 1] <= 2 * B) {
            materialize(pos);
            materialize(pos + 1);
            u32 idL = desc[pos] & ~REV;
            u32 idR = desc[pos + 1] & ~REV;
            Block& L = pool[idL];
            Block& R = pool[idR];
            L.v.insert(L.v.end(), R.v.begin(), R.v.end());
            szs[pos] += szs[pos + 1];
            std::vector<u32>().swap(R.v);
            free_ids.push_back(idR);
            desc.erase(desc.begin() + pos + 1);
            szs.erase(szs.begin() + pos + 1);
        }
    }

    void rebalance(u32 pos) {
        if (szs[pos] > 2 * B)
            split_block(pos);
        else
            merge_small(pos);
    }

    void reverse(u32 l, u32 r) {  // 元素下标 [l, r]（含端点）
        auto [bi, oi] = find(l);
        auto [bj, oj] = find(r);
        if (bi == bj) {
            materialize(bi);
            reverse_avx2(pool[desc[bi] & ~REV].v.data(), oi, (usize)oj + 1);
            return;
        }
        materialize(bi);
        materialize(bj);
        Block& A = pool[desc[bi] & ~REV];
        Block& D = pool[desc[bj] & ~REV];
        u32 nr = oj + 1;  // 右散块参与翻转的前缀长度
        std::vector<u32> rp(D.v.begin(), D.v.begin() + nr);
        reverse_avx2(rp.data(), 0, rp.size());
        std::vector<u32> lp(A.v.begin() + oi, A.v.end());
        reverse_avx2(lp.data(), 0, lp.size());
        std::vector<u32> tail(D.v.begin() + nr, D.v.end());
        A.v.resize(oi);
        A.v.insert(A.v.end(), rp.begin(), rp.end());
        D.v = std::move(lp);
        D.v.insert(D.v.end(), tail.begin(), tail.end());
        szs[bi] = (u32)A.v.size();
        szs[bj] = (u32)D.v.size();
        // 中间块：vpermd 反转 desc/szs（标记随块走），再 vpxor 区间翻转标记
        reverse_avx2(desc.data(), bi + 1, bj);
        reverse_avx2(szs.data(), bi + 1, bj);
        toggle_range(bi + 1, bj);
        rebalance(bj);
        rebalance(bi);
    }

    void collect(std::vector<u32>& out) const {
        out.clear();
        for (u32 i = 0; i < (u32)desc.size(); ++i) {
            const std::vector<u32>& v = pool[desc[i] & ~REV].v;
            if (desc[i] & REV)
                out.insert(out.end(), v.rbegin(), v.rend());
            else
                out.insert(out.end(), v.begin(), v.end());
        }
    }
};

// ---------------- 两层 bitset 块状链表（sqrt_bitset2） ----------------
// 「bitset 套 bitset」：
//   内层：每块 1 bit 内容反转标记，打包进 u32 块描述符（pool_id | REV）。
//   外层：每超块 2 bit（块序反转 ORD / 整块内容反转 CONT），打包进外层
//         u32 描述符（sb_id | ORD | CONT），超块槽位是动态数组。
// 中间整块区间 = 外层 vpermd 反转超块顺序 + vpxor 翻 ORD/CONT 两个位
// （#superblocks/8 条指令）；端部散块先 materialize 到槽位级再按块处理。
// find 先跳超块 total（与槽位顺序无关）再扫 ≤W 个槽位：
// O(#superblocks + W)。超块 >2W 拆、<W/2 并；块 arena + 两级 free-list。
struct SqrtBitset2 {
    static constexpr u32 REV = 0x80000000u;   // 内层：块内容反转
    static constexpr u32 ORD = 0x40000000u;   // 外层：超块内块序反转
    static constexpr u32 CONT = 0x80000000u;  // 外层：超块内所有块内容反转
    static constexpr u32 XMASK = ~(ORD | CONT);
    static constexpr u32 W = 64;              // 超块目标容量（拆 >2W / 并 <W/2）

    struct Block {
        std::vector<u32> v;
    };
    struct SB {
        std::vector<u32> descs;  // pool_id | REV
        std::vector<u32> szs;
        u32 total = 0;
    };

    std::vector<Block> pool;
    std::vector<u32> free_ids;
    std::vector<SB> sbpool;
    std::vector<u32> sb_free;
    std::vector<u32> outer;  // sb_id | ORD | CONT
    u32 B;

    struct Loc {
        u32 opos, slot, off;
    };

    SqrtBitset2(const u32* a, usize n, u64, u32 block_size = 0) {
        B = block_size ? block_size
                       : std::max(64u, std::min(1024u, (u32)(std::sqrt((double)n) * 2.0)));
        for (usize i = 0; i < n;) {
            Block bl;
            usize e = std::min(n, i + B);
            bl.v.assign(a + i, a + e);
            pool.push_back(std::move(bl));
            i = e;
        }
        SB sb;
        u32 total = 0;
        for (u32 i = 0; i < (u32)pool.size(); ++i) {
            if (sb.descs.size() == W) {
                sb.total = total;
                sbpool.push_back(std::move(sb));
                sb = SB{};
                total = 0;
            }
            sb.descs.push_back(i);
            sb.szs.push_back((u32)pool[i].v.size());
            total += (u32)pool[i].v.size();
        }
        if (!sb.descs.empty() || pool.empty()) {
            sb.total = total;
            sbpool.push_back(std::move(sb));
        }
        for (u32 i = 0; i < (u32)sbpool.size(); ++i) outer.push_back(i);
    }

    inline SB& S(u32 opos) { return sbpool[outer[opos] & XMASK]; }
    inline const SB& S(u32 opos) const { return sbpool[outer[opos] & XMASK]; }

    static void toggle_mask(u32* a, u32 l, u32 r, u32 mask) {
        const __m256i m = _mm256_set1_epi32((int)mask);
        u32 i = l;
        for (; i + 8 <= r; i += 8) {
            __m256i x = _mm256_loadu_si256((const __m256i*)(a + i));
            x = _mm256_xor_si256(x, m);
            _mm256_storeu_si256((__m256i*)(a + i), x);
        }
        for (; i < r; ++i) a[i] ^= mask;
    }

    // 把超块的 ORD/CONT 懒标记落到槽位级（ORD -> 物理反转槽位顺序，
    // CONT -> 逐块内容位取反）
    void materialize_sb(u32 opos) {
        u32 d = outer[opos];
        if (d & (ORD | CONT)) {
            SB& s = sbpool[d & XMASK];
            if (d & ORD) {
                reverse_avx2(s.descs.data(), 0, s.descs.size());
                reverse_avx2(s.szs.data(), 0, s.szs.size());
            }
            if (d & CONT) toggle_mask(s.descs.data(), 0, (u32)s.descs.size(), REV);
            outer[opos] = d & XMASK;
        }
    }

    Loc find(u32 k) const {
        u32 oi = 0;
        for (;;) {
            const SB& s = S(oi);
            if (k < s.total) break;
            k -= s.total;
            ++oi;
        }
        const SB& s = S(oi);
        u32 n = (u32)s.descs.size();
        u32 slot = 0;
        if (outer[oi] & ORD) {
            u32 j = n;
            while (j > 0) {
                u32 sz = s.szs[--j];
                if (k < sz) {
                    slot = j;
                    break;
                }
                k -= sz;
            }
        } else {
            u32 j = 0;
            while (j < n) {
                u32 sz = s.szs[j];
                if (k < sz) {
                    slot = j;
                    break;
                }
                k -= sz;
                ++j;
            }
        }
        return {oi, slot, k};
    }

    // 元素级拆分：把超块 opos 内 slot 处的块在偏移 off 处拆成两块。
    // 前置：超块已 materialize；off ∈ (0, sz)。
    void split_block_at(u32 opos, u32 slot, u32 off) {
        SB& s = S(opos);
        u32 sz = s.szs[slot];
        if (off <= 0 || off >= sz) return;
        u32 id = s.descs[slot] & ~REV;
        if (s.descs[slot] & REV) {  // 块内容若懒反转，先物化
            reverse_avx2(pool[id].v.data(), 0, sz);
            s.descs[slot] &= ~REV;
        }
        u32 nid;
        if (!free_ids.empty()) {
            nid = free_ids.back();
            free_ids.pop_back();
            pool[nid].v = std::vector<u32>();
        } else {
            pool.push_back(Block{});
            nid = (u32)pool.size() - 1;
        }
        pool[nid].v.assign(pool[id].v.begin() + off, pool[id].v.end());
        pool[id].v.resize(off);
        s.szs[slot] = off;
        s.descs.insert(s.descs.begin() + slot + 1, nid);
        s.szs.insert(s.szs.begin() + slot + 1, sz - off);
        if (s.descs.size() > 2 * W) split_superblock(opos);
    }

    // 物化单块内容（把块的懒反转落到物理 buffer）
    void materialize_block(SB& s, u32 slot) {
        if (s.descs[slot] & REV) {
            u32 id = s.descs[slot] & ~REV;
            reverse_avx2(pool[id].v.data(), 0, s.szs[slot]);
            s.descs[slot] &= ~REV;
        }
    }

    // 小块合并：触发条件是 sz < B（而不是 < B/2），合并后 ≤ 2B 就拼，
    // 否则端块拆分产生的小块遇到满块邻居永远合不进去，块数会无界增长。
    void merge_small_block(u32 opos, u32 slot) {
        SB& s = S(opos);
        u32 cnt = (u32)s.descs.size();
        if (slot >= cnt) return;
        u32 sz = s.szs[slot];
        if (sz >= B) return;
        if (slot > 0 && s.szs[slot - 1] + sz <= 2 * B) {
            materialize_block(s, slot - 1);
            materialize_block(s, slot);
            u32 lid = s.descs[slot - 1] & ~REV;
            u32 rid = s.descs[slot] & ~REV;
            pool[lid].v.insert(pool[lid].v.end(), pool[rid].v.begin(), pool[rid].v.end());
            s.szs[slot - 1] += sz;
            std::vector<u32>().swap(pool[rid].v);
            free_ids.push_back(rid);
            s.descs.erase(s.descs.begin() + slot);
            s.szs.erase(s.szs.begin() + slot);
            return;
        }
        if (slot + 1 < cnt && sz + s.szs[slot + 1] <= 2 * B) {
            materialize_block(s, slot);
            materialize_block(s, slot + 1);
            u32 lid = s.descs[slot] & ~REV;
            u32 rid = s.descs[slot + 1] & ~REV;
            pool[lid].v.insert(pool[lid].v.end(), pool[rid].v.begin(), pool[rid].v.end());
            s.szs[slot] += s.szs[slot + 1];
            std::vector<u32>().swap(pool[rid].v);
            free_ids.push_back(rid);
            s.descs.erase(s.descs.begin() + slot + 1);
            s.szs.erase(s.szs.begin() + slot + 1);
        }
    }

    // 端超块内的小块合并扫描（保持块数有界）
    void merge_small_scan(u32 opos) {
        SB& s = S(opos);
        u32 k = 0;
        while (k < (u32)s.descs.size()) {
            if (s.szs[k] < B) {
                u32 before = (u32)s.descs.size();
                merge_small_block(opos, k);  // 可能吞掉 k 或 k+1
                if ((u32)s.descs.size() == before) ++k;  // 没合并成必须前进
            } else {
                ++k;
            }
        }
    }

    void split_superblock(u32 opos) {
        u32 sid = outer[opos] & XMASK;
        SB& s = sbpool[sid];
        if (s.descs.size() <= 2 * W) return;
        u32 half = (u32)s.descs.size() / 2;
        u32 nid;
        if (!sb_free.empty()) {
            nid = sb_free.back();
            sb_free.pop_back();
            sbpool[nid].descs.clear();
            sbpool[nid].szs.clear();
            sbpool[nid].total = 0;
        } else {
            sbpool.push_back(SB{});
            nid = (u32)sbpool.size() - 1;
        }
        // push_back 可能让 sbpool 重分配，旧引用失效，重新按下标取
        SB& ss = sbpool[sid];
        SB& n = sbpool[nid];
        n.descs.assign(ss.descs.begin() + half, ss.descs.end());
        n.szs.assign(ss.szs.begin() + half, ss.szs.end());
        u32 t = 0;
        for (u32 x : n.szs) t += x;
        n.total = t;
        ss.descs.resize(half);
        ss.szs.resize(half);
        ss.total -= t;
        outer.insert(outer.begin() + opos + 1, nid);
    }

    void merge_superblock(u32 opos) {
        SB& s = S(opos);
        if (s.descs.size() >= W / 2) return;
        if (opos > 0) {
            SB& L = S(opos - 1);
            if (L.descs.size() + s.descs.size() <= 2 * W) {
                materialize_sb(opos - 1);
                materialize_sb(opos);
                u32 sid = outer[opos] & XMASK;
                L.descs.insert(L.descs.end(), s.descs.begin(), s.descs.end());
                L.szs.insert(L.szs.end(), s.szs.begin(), s.szs.end());
                L.total += s.total;
                s.descs.clear();
                s.szs.clear();
                s.total = 0;
                sb_free.push_back(sid);
                outer.erase(outer.begin() + opos);
                return;
            }
        }
        if (opos + 1 < outer.size()) {
            SB& R = S(opos + 1);
            if (s.descs.size() + R.descs.size() <= 2 * W) {
                materialize_sb(opos);
                materialize_sb(opos + 1);
                u32 rid = outer[opos + 1] & XMASK;
                SB& ss = S(opos);
                ss.descs.insert(ss.descs.end(), R.descs.begin(), R.descs.end());
                ss.szs.insert(ss.szs.end(), R.szs.begin(), R.szs.end());
                ss.total += R.total;
                R.descs.clear();
                R.szs.clear();
                R.total = 0;
                sb_free.push_back(rid);
                outer.erase(outer.begin() + opos + 1);
            }
        }
    }

    void rebalance_sb(u32 opos) {
        u32 n = (u32)S(opos).descs.size();
        if (n > 2 * W)
            split_superblock(opos);
        else if (n < W / 2)
            merge_superblock(opos);
    }

    // 同一超块内的块区间反转（块序 + 每块内容）
    void reverse_slot_range(u32 opos, u32 si, u32 sj) {
        SB& s = S(opos);
        reverse_avx2(s.descs.data(), si, sj + 1);
        reverse_avx2(s.szs.data(), si, sj + 1);
        toggle_mask(s.descs.data(), si, sj + 1, REV);
    }

    void reverse(u32 l, u32 r) {  // 元素下标 [l, r]（含端点）
        Loc a = find(l);
        Loc b = find(r);
        materialize_sb(a.opos);
        if (b.opos != a.opos) materialize_sb(b.opos);
        // 物化把 ORD 落到物理槽位、CONT 落到块位；之后物理==逻辑，
        // 而 find 在 ORD 置位时返回的是物理槽位，必须先重新定位。
        a = find(l);
        b = find(r);
        // 端部按元素边界拆块（可能引发超块拆分，之后重新定位）
        if (a.off > 0) split_block_at(a.opos, a.slot, a.off);
        Loc b2 = find(r);
        {
            const SB& s = S(b2.opos);
            if (b2.off + 1 < s.szs[b2.slot]) split_block_at(b2.opos, b2.slot, b2.off + 1);
        }
        Loc a2 = find(l);
        Loc b3 = find(r);
        if (a2.opos == b3.opos) {
            materialize_sb(a2.opos);
            reverse_slot_range(a2.opos, a2.slot, b3.slot);
            return;
        }
        u32 oi = a2.opos, oj = b3.opos;
        u32 si = a2.slot, sj = b3.slot;
        SB& A = S(oi);
        SB& B = S(oj);
        u32 ac = (u32)A.descs.size(), bc = (u32)B.descs.size();
        u32 npre = si;             // A 前缀块数
        u32 nrb = sj + 1;          // B 的 [0..sj] 块数
        u32 nla = ac - si;         // A 的 [si..) 块数
        u32 ntail = bc - sj - 1;   // B 后缀块数
        // rp_chunk = B[0..sj] 块序 + 内容反转
        std::vector<u32> rp_d(B.descs.begin(), B.descs.begin() + nrb);
        std::vector<u32> rp_s(B.szs.begin(), B.szs.begin() + nrb);
        reverse_avx2(rp_d.data(), 0, rp_d.size());
        reverse_avx2(rp_s.data(), 0, rp_s.size());
        toggle_mask(rp_d.data(), 0, (u32)rp_d.size(), REV);
        // lp_chunk = A[si..) 块序 + 内容反转
        std::vector<u32> lp_d(A.descs.begin() + si, A.descs.end());
        std::vector<u32> lp_s(A.szs.begin() + si, A.szs.end());
        reverse_avx2(lp_d.data(), 0, lp_d.size());
        reverse_avx2(lp_s.data(), 0, lp_s.size());
        toggle_mask(lp_d.data(), 0, (u32)lp_d.size(), REV);
        // tail = B[sj+1..)
        std::vector<u32> tail_d(B.descs.begin() + sj + 1, B.descs.end());
        std::vector<u32> tail_s(B.szs.begin() + sj + 1, B.szs.end());
        // 中间：外层顺序反转 + ORD/CONT 两个标记位（vpermd + vpxor）
        reverse_avx2(outer.data(), oi + 1, oj);
        toggle_mask(outer.data(), oi + 1, oj, ORD);
        toggle_mask(outer.data(), oi + 1, oj, CONT);
        // 写回 A = prefix + rp_chunk
        A.descs.resize(npre + nrb);
        A.szs.resize(npre + nrb);
        std::copy(rp_d.begin(), rp_d.end(), A.descs.begin() + npre);
        std::copy(rp_s.begin(), rp_s.end(), A.szs.begin() + npre);
        u32 t = 0;
        for (u32 x : A.szs) t += x;
        A.total = t;
        // 写回 B = lp_chunk + tail
        B.descs.resize(nla + ntail);
        B.szs.resize(nla + ntail);
        std::copy(lp_d.begin(), lp_d.end(), B.descs.begin());
        std::copy(lp_s.begin(), lp_s.end(), B.szs.begin());
        std::copy(tail_d.begin(), tail_d.end(), B.descs.begin() + nla);
        std::copy(tail_s.begin(), tail_s.end(), B.szs.begin() + nla);
        t = 0;
        for (u32 x : B.szs) t += x;
        B.total = t;
        // 小块合并（保持块数有界），再做超块结构维护
        merge_small_scan(oi);
        merge_small_scan(oj);
        // 结构维护：先大下标
        rebalance_sb(oj);
        rebalance_sb(oi);
    }

    void collect(std::vector<u32>& out) const {
        out.clear();
        for (u32 o = 0; o < (u32)outer.size(); ++o) {
            u32 od = outer[o];
            const SB& s = sbpool[od & XMASK];
            u32 n = (u32)s.descs.size();
            for (u32 k = 0; k < n; ++k) {
                u32 slot = (od & ORD) ? n - 1 - k : k;
                u32 id = s.descs[slot] & ~REV;
                bool rev = ((s.descs[slot] & REV) != 0) ^ ((od & CONT) != 0);
                const std::vector<u32>& v = pool[id].v;
                if (rev)
                    out.insert(out.end(), v.rbegin(), v.rend());
                else
                    out.insert(out.end(), v.begin(), v.end());
            }
        }
    }
};

// ---------------- 方法分发 ----------------
enum Method : int { M_BRUTE = 0, M_STD, M_AVX2, M_TREAP, M_SPLAY, M_SQRT, M_SQRT_BITSET, M_SQRT_BITSET2, M_COUNT };
static const char* METHOD_NAME[M_COUNT] = {"brute", "std", "avx2",
                                           "treap", "splay", "sqrt", "sqrt_bitset", "sqrt_bitset2"};

static inline void apply_reverse(int m, u32* a, u32 l, u32 r) {  // [l, r] 含端点
    switch (m) {
    case M_BRUTE: reverse_scalar(a, l, (usize)r + 1); break;
    case M_STD: reverse_std(a, l, (usize)r + 1); break;
    case M_AVX2: reverse_avx2(a, l, (usize)r + 1); break;
    default: break;
    }
}

struct DS {
    int kind;
    Treap* t = nullptr;
    Splay* s = nullptr;
    Sqrt* q = nullptr;
    SqrtBitset* qb = nullptr;
    SqrtBitset2* qb2 = nullptr;
    std::vector<u32> arr;

    DS(int m, const u32* a, usize n, u64 seed, u32 block_size = 0) : kind(m) {
        if (m == M_TREAP) t = new Treap(a, n, seed);
        if (m == M_SPLAY) s = new Splay(a, n, seed);
        if (m == M_SQRT) q = new Sqrt(a, n, seed, block_size);
        if (m == M_SQRT_BITSET) qb = new SqrtBitset(a, n, seed, block_size);
        if (m == M_SQRT_BITSET2) qb2 = new SqrtBitset2(a, n, seed, block_size);
        if (m == M_BRUTE || m == M_STD || m == M_AVX2) {
            arr.assign(a, a + n);
        }
    }
    ~DS() {
        delete t;
        delete s;
        delete q;
        delete qb;
        delete qb2;
    }
    void reverse(u32 l, u32 r) {
        if (t) t->reverse(l, r);
        if (s) s->reverse(l, r);
        if (q) q->reverse(l, r);
        if (qb) qb->reverse(l, r);
        if (qb2) qb2->reverse(l, r);
        if (!arr.empty()) apply_reverse(kind, arr.data(), l, r);
    }
    void collect(std::vector<u32>& out) {
        if (t) {
            t->collect(out);
            return;
        }
        if (s) {
            s->collect(out);
            return;
        }
        if (q) {
            q->collect(out);
            return;
        }
        if (qb) {
            qb->collect(out);
            return;
        }
        if (qb2) {
            qb2->collect(out);
            return;
        }
        out = arr;
    }
    // 按位置的加权 checksum：反转会改变序列，因此不会被编译器当常量优化掉。
    u64 weighted(const std::vector<u64>& w, std::vector<u32>& out) {
        collect(out);
        u64 sum = 0;
        for (usize i = 0; i < out.size(); ++i) sum += (u64)out[i] * w[i];
        return sum;
    }
};

// ---------------- 正确性验证 ----------------
static void verify_length(int m, u32 n, u32 L) {
    if (m == M_BRUTE) return;
    std::vector<u32> a(n), b(n);
    u64 seed = 0x9e3779b97f4a7c15ULL ^ L ^ (u64)m;
    for (u32 i = 0; i < n; ++i) {
        u32 v = (u32)(splitmix64(seed) & 0xffff);
        a[i] = b[i] = v;
    }
    u64 s = seed ^ 0x123456789abcdef0ULL;
    u32 cnt = 8;
    while (cnt--) {
        u32 l = rnd(s, n - L + 1);
        u32 r = l + L;  // [l, r) 半开
        reverse_scalar(a.data(), l, r);
        DS ds(m, b.data(), n, seed);
        ds.reverse(l, r - 1);
        std::vector<u32> got(n);
        ds.collect(got);
        if (got != a) {
            std::fprintf(stderr, "VERIFY FAIL length L=%u m=%s\n", L, METHOD_NAME[m]);
            std::abort();
        }
        b = a;  // 让双方状态同步，继续下一轮
    }
}

static void verify_batch(int m, u32 n, const std::vector<u32>& tmpl, u32 block_size = 0) {
    if (m == M_BRUTE) return;
    std::vector<u32> a = tmpl;
    DS ds(m, tmpl.data(), n, 0xdeadbeefULL ^ (u64)m ^ n, block_size);
    u64 s = 0xabcdef0123456789ULL ^ (u64)m ^ n;
    for (u32 i = 0; i < 200; ++i) {
        u32 l = rnd(s, n);
        u32 r = rnd(s, n);
        if (l > r) std::swap(l, r);
        reverse_scalar(a.data(), l, (usize)r + 1);
        ds.reverse(l, r);
    }
    std::vector<u32> got(n);
    ds.collect(got);
    if (got != a) {
        std::fprintf(stderr, "VERIFY FAIL batch n=%u m=%s\n", n, METHOD_NAME[m]);
        std::abort();
    }
}

// ---------------- length 模式 ----------------
static double time_length_once(int m, const std::vector<u32>& tmpl, u32 L,
                               u64 q, u64 seed, const std::vector<u64>& w) {
    DS ds(m, tmpl.data(), (usize)tmpl.size(), seed ^ 0x5eed);
    u64 s = seed;
    double t0 = now_ms();
    for (u64 i = 0; i < q; ++i) {
        u32 l = rnd(s, LEN_N - L + 1);
        ds.reverse(l, l + L - 1);
    }
    double t1 = now_ms();
    // 加权 checksum 不计入计时，但放在计时后执行 + black_box，
    // 防止整段 ops 被优化掉（普通求和是反转不变量，编译器会把它当常量）。
    std::vector<u32> out(tmpl.size());
    u64 c = ds.weighted(w, out);
    black_box_u64(c);
    return t1 - t0;
}

static void measure_length() {
    std::vector<u32> tmpl(LEN_N);
    u64 seed = 0x5a5a5a5a5a5a5a5aULL;
    for (u32 i = 0; i < LEN_N; ++i) tmpl[i] = (u32)(splitmix64(seed) & 0xffff);
    std::vector<u64> w(LEN_N);
    u64 wseed = 0x7777777777777777ULL;
    for (u32 i = 0; i < LEN_N; ++i) w[i] = splitmix64(wseed);

    for (u32 L = 32; L <= LEN_N; L <<= 1) {
        for (int m = 0; m < M_COUNT; ++m) {
            verify_length(m, LEN_N, L);
            // 校准：q 从 1 起倍增，直到一轮 >= 3 ms
            u64 q = 1;
            while (q <= Q_CAP) {
                double t = time_length_once(m, tmpl, L, q, 0x1111 ^ (u64)m ^ L, w);
                if (t >= TARGET_MS) break;
                q <<= 1;
            }
            if (q > Q_CAP) q = Q_CAP;
            std::vector<double> times;
            for (u32 r = 0; r < ROUNDS; ++r)
                times.push_back(time_length_once(
                    m, tmpl, L, q, 0x2222 ^ (u64)m ^ L ^ ((u64)r << 32), w));
            double ns = median(times) / (double)q * 1e6;
            std::printf("length,%u,%u,%s,%.3f\n", LEN_N, L, METHOD_NAME[m], ns);
            std::fflush(stdout);
        }
    }
}

// ---------------- batch 模式 ----------------
static double time_batch_once(int m, u32 n, const std::vector<u32>& tmpl,
                              u64 seed, const std::vector<u64>& w, u32 block_size = 0) {
    double t0 = now_ms();
    DS ds(m, tmpl.data(), n, seed ^ 0xbeef, block_size);
    u64 s = seed;
    for (u32 i = 0; i < OPS; ++i) {
        u32 l = rnd(s, n);
        u32 r = rnd(s, n);
        if (l > r) std::swap(l, r);
        ds.reverse(l, r);
    }
    std::vector<u32> out(n);
    u64 c = ds.weighted(w, out);
    double t1 = now_ms();
    black_box_u64(c);
    return t1 - t0;
}

static void measure_batch() {
    for (u32 n = 1024; n <= (1u << 20); n <<= 1) {
        std::vector<u32> tmpl(n);
        u64 seed = 0xabcde12345ULL ^ n;
        for (u32 i = 0; i < n; ++i) tmpl[i] = (u32)(splitmix64(seed) & 0xffff);
        std::vector<u64> w(n);
        u64 wseed = 0x8888888888888888ULL ^ n;
        for (u32 i = 0; i < n; ++i) w[i] = splitmix64(wseed);
        for (int m = 0; m < M_COUNT; ++m) verify_batch(m, n, tmpl);
        for (int m = 0; m < M_COUNT; ++m) {
            std::vector<double> times;
            for (u32 r = 0; r < ROUNDS; ++r)
                times.push_back(time_batch_once(
                    m, n, tmpl, 0x3333 ^ (u64)m ^ n ^ ((u64)r << 32), w));
            double ms = median(times);
            std::printf("batch,%u,0,%s,%.3f\n", n, METHOD_NAME[m], ms);
            std::fflush(stdout);
        }
    }
}

// ---------------- bsweep 模式：固定 batch 场景扫块长 B ----------------
static void measure_bsweep() {
    const u32 BS[] = {64, 128, 256, 512, 1024, 2048, 4096, 0};  // 0 = auto
    for (u32 n : {1u << 16, 1u << 20}) {
        std::vector<u32> tmpl(n);
        u64 seed = 0x0badc0deULL ^ n;
        for (u32 i = 0; i < n; ++i) tmpl[i] = (u32)(splitmix64(seed) & 0xffff);
        std::vector<u64> w(n);
        u64 wseed = 0x9999999999999999ULL ^ n;
        for (u32 i = 0; i < n; ++i) w[i] = splitmix64(wseed);
        for (u32 B : BS) {
            for (int m : {M_SQRT, M_SQRT_BITSET, M_SQRT_BITSET2}) {
                verify_batch(m, n, tmpl, B);
                std::vector<double> times;
                for (u32 r = 0; r < ROUNDS; ++r)
                    times.push_back(time_batch_once(
                        m, n, tmpl, 0x4444 ^ (u64)m ^ n ^ B ^ ((u64)r << 32), w, B));
                double ms = median(times);
                std::printf("bsweep,%u,%u,%s,%.3f\n", n, B, METHOD_NAME[m], ms);
                std::fflush(stdout);
            }
        }
    }
}

int main(int argc, char** argv) {
    const bool only_length = argc > 1 && std::string(argv[1]) == "length";
    const bool only_batch = argc > 1 && std::string(argv[1]) == "batch";
    const bool only_bsweep = argc > 1 && std::string(argv[1]) == "bsweep";
    if (!only_batch && !only_bsweep) measure_length();
    if (!only_length && !only_bsweep) measure_batch();
    if (only_bsweep) measure_bsweep();
    return 0;
}
