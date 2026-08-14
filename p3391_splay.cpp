// P3391 【模板】文艺平衡树 —— 标准 splay 参考实现（bench 同款，已对拍）
// 实测（n=m=1e5）：随机区间 ~96ms；固定边界区间（如反复翻转 [1,n/2]）
// ~20ms（边界节点被 splay 到根附近的局部性红利）。
// 编译：g++ -O2 -std=c++17 p3391_splay.cpp -o p3391_splay

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using usize = std::size_t;

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

    explicit Splay(const u32* a, usize n) : n((u32)n) {
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
        u32 A = kth(l);
        splay(A, 0);
        u32 B = kth(r + 2);
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
            if (x != 1 && x != n + 2) out.push_back(p[x].val);
            x = p[x].r;
        }
    }
};

int main() {
    int n, m;
    if (scanf("%d%d", &n, &m) != 2) return 0;
    std::vector<u32> a(n);
    for (int i = 0; i < n; ++i) a[i] = i + 1;
    Splay sp(a.data(), n);
    for (int i = 0; i < m; ++i) {
        int l, r;
        scanf("%d%d", &l, &r);
        sp.reverse((u32)l - 1, (u32)r - 1);
    }
    std::vector<u32> out;
    sp.collect(out);
    for (u32 i = 0; i < out.size(); ++i)
        printf("%u%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    return 0;
}
