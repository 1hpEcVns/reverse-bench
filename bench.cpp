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

    Sqrt(const u32* a, usize n, u64) {
        B = std::max(64u, std::min(1024u, (u32)(std::sqrt((double)n) * 2.0)));
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
            std::reverse(b[i].v.begin(), b[i].v.end());
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

    void reverse(u32 l, u32 r) {  // 元素下标 [l, r]（含端点）
        auto [bi, oi] = find(l);
        auto [bj, oj] = find(r);
        if (bi == bj) {
            materialize(bi);
            std::reverse(b[bi].v.begin() + oi, b[bi].v.begin() + oj + 1);
            return;
        }
        materialize(bi);
        materialize(bj);
        u32 nr = oj + 1;  // 右散块参与翻转的前缀长度
        std::vector<u32> rp(b[bj].v.begin(), b[bj].v.begin() + nr);
        std::reverse(rp.begin(), rp.end());
        std::vector<u32> lp(b[bi].v.begin() + oi, b[bi].v.end());
        std::reverse(lp.begin(), lp.end());
        b[bi].v.resize(oi);
        b[bi].v.insert(b[bi].v.end(), rp.begin(), rp.end());
        std::vector<u32> tail(b[bj].v.begin() + oj + 1, b[bj].v.end());
        b[bj].v = std::move(lp);
        b[bj].v.insert(b[bj].v.end(), tail.begin(), tail.end());
        // 中间整块：先反转块列表顺序，再逐块翻转（懒标记）
        std::reverse(b.begin() + bi + 1, b.begin() + bj);
        for (u32 i = bi + 1; i < bj; ++i) b[i].rev ^= 1;
        // 先拆大下标，避免插入导致另一个下标失效
        split_block(bj);
        split_block(bi);
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

// ---------------- 方法分发 ----------------
enum Method : int { M_BRUTE = 0, M_STD, M_AVX2, M_TREAP, M_SPLAY, M_SQRT, M_COUNT };
static const char* METHOD_NAME[M_COUNT] = {"brute", "std", "avx2",
                                           "treap", "splay", "sqrt"};

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
    std::vector<u32> arr;

    DS(int m, const u32* a, usize n, u64 seed) : kind(m) {
        if (m == M_TREAP) t = new Treap(a, n, seed);
        if (m == M_SPLAY) s = new Splay(a, n, seed);
        if (m == M_SQRT) q = new Sqrt(a, n, seed);
        if (m == M_BRUTE || m == M_STD || m == M_AVX2) {
            arr.assign(a, a + n);
        }
    }
    ~DS() {
        delete t;
        delete s;
        delete q;
    }
    void reverse(u32 l, u32 r) {
        if (t) t->reverse(l, r);
        if (s) s->reverse(l, r);
        if (q) q->reverse(l, r);
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

static void verify_batch(int m, u32 n, const std::vector<u32>& tmpl) {
    if (m == M_BRUTE) return;
    std::vector<u32> a = tmpl;
    DS ds(m, tmpl.data(), n, 0xdeadbeefULL ^ (u64)m ^ n);
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
                              u64 seed, const std::vector<u64>& w) {
    double t0 = now_ms();
    DS ds(m, tmpl.data(), n, seed ^ 0xbeef);
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

int main(int argc, char** argv) {
    const bool only_length = argc > 1 && std::string(argv[1]) == "length";
    const bool only_batch = argc > 1 && std::string(argv[1]) == "batch";
    if (!only_batch) measure_length();
    if (!only_length) measure_batch();
    return 0;
}
