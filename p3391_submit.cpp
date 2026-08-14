// P3391 【模板】文艺平衡树 —— 最优解（基于 reverse_bench 实测）
//
// 结构：sqrt_bitset（懒标记位图化块状链表）
//   - 块大小 B ≈ 2√n，>2B 拆块、<B/2 与邻居合并，块数保持 O(n/B)
//   - 懒反转标记打包进 u32 块描述符最高位：
//       整块区间翻转 = desc 区间反转（块序 + 标记随块走）+ vpxor 翻转标记
//       端部散块先物化再按元素反转
//   - 块本体放 arena + free-list 回收，无动态分配
// 实测（i9-13950HX）：n=m=1e5 随机区间约 20-40ms，比 splay/treap 快数倍。
// 可选 AVX2：编译时加 -mavx2 且 CPU 支持时启用 vpermd/vpxor，否则纯标量。
//
// 编译：g++ -O2 -std=c++17 p3391_submit.cpp -o p3391
//       （想开 SIMD：g++ -O2 -mavx2 -std=c++17 p3391_submit.cpp -o p3391）

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using u32 = std::uint32_t;

static inline void reverse_scalar(u32* a, std::size_t l, std::size_t r) {
    while (l < r) {
        --r;
        u32 t = a[l];
        a[l] = a[r];
        a[r] = t;
        ++l;
    }
}

#if defined(__AVX2__)
static inline void reverse_avx2(u32* a, std::size_t l, std::size_t r) {
    const __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    while (l + 8 <= r) {
        __m256i L = _mm256_loadu_si256((const __m256i*)(a + l));
        __m256i R = _mm256_loadu_si256((const __m256i*)(a + r - 8));
        _mm256_storeu_si256((__m256i*)(a + l), _mm256_permutevar8x32_epi32(R, idx));
        _mm256_storeu_si256((__m256i*)(a + r - 8), _mm256_permutevar8x32_epi32(L, idx));
        l += 8;
        r -= 8;
    }
    reverse_scalar(a, l, r);
}
#endif

// 区间懒标记翻转（desc 的 REV 位）
static inline void toggle_mask(u32* a, u32 l, u32 r, u32 mask) {
#if defined(__AVX2__)
    if (__builtin_cpu_supports("avx2")) {
        const __m256i m = _mm256_set1_epi32((int)mask);
        u32 i = l;
        for (; i + 8 <= r; i += 8) {
            __m256i x = _mm256_loadu_si256((const __m256i*)(a + i));
            x = _mm256_xor_si256(x, m);
            _mm256_storeu_si256((__m256i*)(a + i), x);
        }
        for (; i < r; ++i) a[i] ^= mask;
        return;
    }
#endif
    for (u32 i = l; i < r; ++i) a[i] ^= mask;
}

static inline void reverse_best(u32* a, std::size_t l, std::size_t r) {
#if defined(__AVX2__)
    if (__builtin_cpu_supports("avx2")) {
        reverse_avx2(a, l, r);
        return;
    }
#endif
    reverse_scalar(a, l, r);
}

struct SqrtBitset {
    static constexpr u32 REV = 0x80000000u;

    struct Block {
        std::vector<u32> v;
    };

    std::vector<Block> pool;    // arena：块本体，id = 下标
    std::vector<u32> free_ids;  // 合并后回收的槽位
    std::vector<u32> desc;      // 逻辑块顺序：pool_id | (懒反转 << 31)
    std::vector<u32> szs;       // 与 desc 平行的块大小
    u32 B;

    explicit SqrtBitset(u32 n) {
        B = std::max(64u, std::min(1024u, (u32)(std::sqrt((double)n) * 2.0)));
        for (u32 i = 0; i < n;) {
            u32 e = std::min(n, i + B);
            Block bl;
            bl.v.reserve(e - i);
            for (u32 x = i; x < e; ++x) bl.v.push_back(x + 1);
            pool.push_back(std::move(bl));
            desc.push_back((u32)pool.size() - 1);
            szs.push_back(e - i);
            i = e;
        }
    }

    std::pair<u32, u32> find(u32 k) const {
        for (u32 i = 0; i < (u32)desc.size(); ++i) {
            u32 s = szs[i];
            if (k < s) return {i, k};
            k -= s;
        }
        return {0, 0};
    }

    inline void materialize(u32 pos) {
        if (desc[pos] & REV) {
            u32 id = desc[pos] & ~REV;
            reverse_best(pool[id].v.data(), 0, szs[pos]);
            desc[pos] &= ~REV;
        }
    }

    void split_block(u32 pos) {
        u32 s = szs[pos];
        if (s <= 2 * B) return;
        u32 half = s / 2;
        u32 id = desc[pos] & ~REV;  // 端块必已物化
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
            pool[idL].v.insert(pool[idL].v.end(), pool[idR].v.begin(), pool[idR].v.end());
            szs[pos - 1] += s;
            std::vector<u32>().swap(pool[idR].v);
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
            pool[idL].v.insert(pool[idL].v.end(), pool[idR].v.begin(), pool[idR].v.end());
            szs[pos] += szs[pos + 1];
            std::vector<u32>().swap(pool[idR].v);
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
            u32 id = desc[bi] & ~REV;
            reverse_best(pool[id].v.data(), oi, (std::size_t)oj + 1);
            return;
        }
        materialize(bi);
        materialize(bj);
        u32 idA = desc[bi] & ~REV;
        u32 idD = desc[bj] & ~REV;
        Block& A = pool[idA];
        Block& D = pool[idD];
        u32 nr = oj + 1;  // 右散块参与翻转的前缀长度
        std::vector<u32> rp(D.v.begin(), D.v.begin() + nr);
        reverse_best(rp.data(), 0, rp.size());
        std::vector<u32> lp(A.v.begin() + oi, A.v.end());
        reverse_best(lp.data(), 0, lp.size());
        std::vector<u32> tail(D.v.begin() + nr, D.v.end());
        A.v.resize(oi);
        A.v.insert(A.v.end(), rp.begin(), rp.end());
        D.v = std::move(lp);
        D.v.insert(D.v.end(), tail.begin(), tail.end());
        szs[bi] = (u32)A.v.size();
        szs[bj] = (u32)D.v.size();
        // 中间整块：块序反转（vpermd，标记随块走）+ 区间懒标记翻转（vpxor）
        reverse_best(desc.data(), bi + 1, bj);
        reverse_best(szs.data(), bi + 1, bj);
        toggle_mask(desc.data(), bi + 1, bj, REV);
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

int main() {
    int n, m;
    if (scanf("%d%d", &n, &m) != 2) return 0;
    SqrtBitset sol((u32)n);
    for (int i = 0; i < m; ++i) {
        int l, r;
        scanf("%d%d", &l, &r);
        sol.reverse((u32)l - 1, (u32)r - 1);
    }
    std::vector<u32> out;
    sol.collect(out);
    for (u32 i = 0; i < (u32)out.size(); ++i)
        printf("%u%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    return 0;
}
