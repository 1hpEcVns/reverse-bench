// reverse_bench: 区间反转（range reverse）的 SIMD / 数据结构对比（Rust edition 2024）
// 与 bench.cpp 完全同构：
//   brute / std / avx2 / treap / splay / sqrt，length + batch 两种模式。
// 编译：rustc --edition=2024 -O -C target-cpu=native bench.rs -o bench_rs
#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

type U32 = u32;

const ROUNDS: usize = 9;
const TARGET_MS: f64 = 3.0;
const Q_CAP: u64 = 8_000_000;
const OPS: u32 = 5000;
const LEN_N: u32 = 1 << 20;

// ---------------- RNG（splitmix64，与 C++ 完全一致） ----------------
#[inline(always)]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline(always)]
fn rnd(s: &mut u64, modu: u32) -> u32 {
    ((splitmix64(s) >> 32) % modu as u64) as u32
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ---------------- 暴力：标量 / slice::reverse / AVX2 ----------------
#[inline(always)]
unsafe fn reverse_scalar(a: &mut [u32], mut l: usize, mut r: usize) {
    while l < r {
        r -= 1;
        let p = a.as_mut_ptr();
        let x = *p.add(l);
        let y = *p.add(r);
        *p.add(l) = y;
        *p.add(r) = x;
        l += 1;
    }
}

#[inline(always)]
fn reverse_std(a: &mut [u32], l: usize, r: usize) {
    a[l..r].reverse();
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn reverse_avx2(a: &mut [u32], mut l: usize, mut r: usize) {
    let idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    while l + 8 <= r {
        // 不能用 r - l >= 8：最后一次迭代后 l > r 会无符号下溢
        let base = a.as_mut_ptr();
        let lv = _mm256_loadu_si256(base.add(l) as *const __m256i);
        let rv = _mm256_loadu_si256(base.add(r - 8) as *const __m256i);
        _mm256_storeu_si256(
            base.add(l) as *mut __m256i,
            _mm256_permutevar8x32_epi32(rv, idx),
        );
        _mm256_storeu_si256(
            base.add(r - 8) as *mut __m256i,
            _mm256_permutevar8x32_epi32(lv, idx),
        );
        l += 8;
        r -= 8;
    }
    reverse_scalar(a, l, r);
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
unsafe fn reverse_avx2(a: &mut [u32], l: usize, r: usize) {
    reverse_scalar(a, l, r);
}

// ---------------- FHQ 隐式 treap（懒反转） ----------------
#[derive(Clone, Copy, Default)]
struct TreapNode {
    l: u32,
    r: u32,
    pri: u32,
    sz: u32,
    val: u32,
    rev: u8,
}

struct Treap {
    p: Vec<TreapNode>,
    root: u32,
    rng: u64,
}

impl Treap {
    fn new(a: &[u32], seed: u64) -> Treap {
        let mut t = Treap {
            p: Vec::with_capacity(a.len() + 1),
            root: 0,
            rng: seed,
        };
        t.p.push(TreapNode::default()); // 下标 0 = 空
        let mut st: Vec<u32> = Vec::with_capacity(a.len());
        for &v in a {
            let x = t.new_node(v);
            let mut last = 0;
            while let Some(&top) = st.last() {
                let tp = unsafe { t.get(top) };
                let xp = unsafe { t.get(x) };
                if tp.pri <= xp.pri {
                    break;
                }
                last = top;
                st.pop();
            }
            if let Some(&top) = st.last() {
                unsafe { (*t.get_mut(top)).r = x };
            } else {
                t.root = x;
            }
            unsafe { (*t.get_mut(x)).l = last };
            st.push(x);
        }
        // 创建顺序倒序 pull 不行：左孩子下标更小，此时还没被 pull。
        // 改为后序遍历（孩子先于父亲）一次性求 size。
        let mut order = Vec::with_capacity(a.len());
        let mut stk2 = vec![t.root];
        while let Some(x) = stk2.pop() {
            order.push(x);
            let nd = unsafe { t.get(x) };
            if nd.l != 0 {
                stk2.push(nd.l);
            }
            if nd.r != 0 {
                stk2.push(nd.r);
            }
        }
        for &x in order.iter().rev() {
            t.pull(x);
        }
        t
    }

    #[inline(always)]
    fn new_node(&mut self, v: u32) -> u32 {
        let nd = TreapNode {
            pri: (splitmix64(&mut self.rng) >> 32) as u32,
            sz: 1,
            val: v,
            ..Default::default()
        };
        self.p.push(nd);
        (self.p.len() - 1) as u32
    }

    #[inline(always)]
    unsafe fn get(&self, i: u32) -> TreapNode {
        *self.p.get_unchecked(i as usize)
    }

    #[inline(always)]
    unsafe fn get_mut(&mut self, i: u32) -> *mut TreapNode {
        self.p.as_mut_ptr().add(i as usize)
    }

    #[inline(always)]
    fn sz(&self, x: u32) -> u32 {
        if x == 0 {
            0
        } else {
            unsafe { self.get(x).sz }
        }
    }

    #[inline(always)]
    fn pull(&mut self, x: u32) {
        if x != 0 {
            let nd = unsafe { self.get(x) };
            let sz = 1 + self.sz(nd.l) + self.sz(nd.r);
            unsafe { (*self.get_mut(x)).sz = sz };
        }
    }

    #[inline(always)]
    fn push(&mut self, x: u32) {
        if x == 0 {
            return;
        }
        unsafe {
            let rev = (*self.get_mut(x)).rev;
            if rev != 0 {
                let nd = self.get_mut(x);
                let l = (*nd).l;
                let r = (*nd).r;
                (*nd).l = r;
                (*nd).r = l;
                (*nd).rev = 0;
                if r != 0 {
                    (*self.get_mut(r)).rev ^= 1;
                }
                if l != 0 {
                    (*self.get_mut(l)).rev ^= 1;
                }
            }
        }
    }

    unsafe fn split(&mut self, t: u32, k: u32) -> (u32, u32) {
        if t == 0 {
            return (0, 0);
        }
        self.push(t);
        let ls = self.sz(self.get(t).l);
        if k <= ls {
            let (a, b) = self.split(self.get(t).l, k);
            (*self.get_mut(t)).l = b;
            self.pull(t);
            (a, t)
        } else {
            let (a, b) = self.split(self.get(t).r, k - ls - 1);
            (*self.get_mut(t)).r = a;
            self.pull(t);
            (t, b)
        }
    }

    unsafe fn merge(&mut self, a: u32, b: u32) -> u32 {
        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }
        if self.get(a).pri < self.get(b).pri {
            self.push(a);
            let m = self.merge(self.get(a).r, b);
            (*self.get_mut(a)).r = m;
            self.pull(a);
            a
        } else {
            self.push(b);
            let m = self.merge(a, self.get(b).l);
            (*self.get_mut(b)).l = m;
            self.pull(b);
            b
        }
    }

    fn reverse(&mut self, l: u32, r: u32) {
        unsafe {
            let (a, b) = self.split(self.root, l);
            let (mid, c) = self.split(b, r - l + 1);
            (*self.get_mut(mid)).rev ^= 1;
            let mc = self.merge(mid, c);
            self.root = self.merge(a, mc);
        }
    }

    fn collect(&mut self, out: &mut Vec<u32>) {
        out.clear();
        let mut stk: Vec<u32> = Vec::with_capacity(64);
        let mut x = self.root;
        loop {
            while x != 0 {
                self.push(x);
                stk.push(x);
                x = unsafe { self.get(x).l };
            }
            match stk.pop() {
                None => break,
                Some(t) => {
                    out.push(unsafe { self.get(t).val });
                    x = unsafe { self.get(t).r };
                }
            }
        }
    }
}

// ---------------- 隐式 splay（懒反转） ----------------
#[derive(Clone, Copy, Default)]
struct SplayNode {
    l: u32,
    r: u32,
    p: u32,
    sz: u32,
    val: u32,
    rev: u8,
}

struct Splay {
    p: Vec<SplayNode>,
    root: u32,
    n: u32,
}

impl Splay {
    fn new(a: &[u32], _seed: u64) -> Splay {
        let n = a.len() as u32;
        let mut p = vec![SplayNode::default(); (n + 3) as usize];
        for i in 1..=(n + 2) {
            p[i as usize].val = if i >= 2 && i <= n + 1 {
                a[(i - 2) as usize]
            } else {
                0
            };
        }
        let mut s = Splay { p, root: 0, n };
        s.root = s.build(1, n + 2, 0);
        s
    }

    fn build(&mut self, l: u32, r: u32, par: u32) -> u32 {
        if l > r {
            return 0;
        }
        let m = (l + r) >> 1;
        unsafe {
            (*self.get_mut(m)).p = par;
            let lc = self.build(l, m - 1, m);
            let rc = self.build(m + 1, r, m);
            (*self.get_mut(m)).l = lc;
            (*self.get_mut(m)).r = rc;
        }
        self.pull(m);
        m
    }

    #[inline(always)]
    unsafe fn get(&self, i: u32) -> SplayNode {
        *self.p.get_unchecked(i as usize)
    }

    #[inline(always)]
    unsafe fn get_mut(&mut self, i: u32) -> *mut SplayNode {
        self.p.as_mut_ptr().add(i as usize)
    }

    #[inline(always)]
    fn pull(&mut self, x: u32) {
        if x != 0 {
            let nd = unsafe { self.get(x) };
            let sz = 1 + unsafe { self.get(nd.l) }.sz + unsafe { self.get(nd.r) }.sz;
            unsafe { (*self.get_mut(x)).sz = sz };
        }
    }

    #[inline(always)]
    fn push(&mut self, x: u32) {
        if x == 0 {
            return;
        }
        unsafe {
            if (*self.get_mut(x)).rev != 0 {
                let nd = self.get_mut(x);
                let l = (*nd).l;
                let r = (*nd).r;
                (*nd).l = r;
                (*nd).r = l;
                (*nd).rev = 0;
                if r != 0 {
                    (*self.get_mut(r)).rev ^= 1;
                }
                if l != 0 {
                    (*self.get_mut(l)).rev ^= 1;
                }
            }
        }
    }

    fn rotate(&mut self, x: u32) {
        unsafe {
            let y = self.get(x).p;
            let z = self.get(y).p;
            let nd = self.get(x);
            if self.get(y).l == x {
                (*self.get_mut(y)).l = nd.r;
                if nd.r != 0 {
                    (*self.get_mut(nd.r)).p = y;
                }
                (*self.get_mut(x)).r = y;
            } else {
                (*self.get_mut(y)).r = nd.l;
                if nd.l != 0 {
                    (*self.get_mut(nd.l)).p = y;
                }
                (*self.get_mut(x)).l = y;
            }
            (*self.get_mut(y)).p = x;
            (*self.get_mut(x)).p = z;
            if z != 0 {
                if self.get(z).l == y {
                    (*self.get_mut(z)).l = x;
                } else {
                    (*self.get_mut(z)).r = x;
                }
            }
            self.pull(y);
            self.pull(x);
        }
    }

    // 前置条件：root..x 路径上的 rev 都已 push（调用方先 kth）
    fn splay(&mut self, x: u32, goal: u32) {
        loop {
            let px = unsafe { self.get(x).p };
            if px == goal {
                break;
            }
            let y = px;
            let z = unsafe { self.get(y).p };
            if z != goal {
                let same = unsafe { self.get(y).l == x && self.get(z).l == y }
                    || unsafe { self.get(y).r == x && self.get(z).r == y };
                if same {
                    self.rotate(y);
                } else {
                    self.rotate(x);
                }
            }
            self.rotate(x);
        }
        if goal == 0 {
            self.root = x;
        }
    }

    fn kth(&mut self, mut k: u32) -> u32 {
        let mut x = self.root;
        loop {
            self.push(x);
            let ls = unsafe { self.get(unsafe { self.get(x).l }).sz };
            if k < ls {
                x = unsafe { self.get(x).l };
            } else if k == ls {
                return x;
            } else {
                k -= ls + 1;
                x = unsafe { self.get(x).r };
            }
        }
    }

    fn reverse(&mut self, l: u32, r: u32) {
        // 哨兵头 rank 0，元素 i 的 rank = i+1，哨兵尾 rank = n+1。
        // 翻转 [l, r] 需要 splay rank l（区间前一个节点）和 rank r+2（区间后一个）。
        let a = self.kth(l);
        self.splay(a, 0);
        let b = self.kth(r + 2);
        self.splay(b, self.root);
        let mid = unsafe { self.get(self.get(self.root).r).l };
        unsafe { (*self.get_mut(mid)).rev ^= 1 };
    }

    fn collect(&mut self, out: &mut Vec<u32>) {
        out.clear();
        let mut stk: Vec<u32> = Vec::with_capacity(64);
        let mut x = self.root;
        loop {
            while x != 0 {
                self.push(x);
                stk.push(x);
                x = unsafe { self.get(x).l };
            }
            match stk.pop() {
                None => break,
                Some(t) => {
                    // 跳过两个哨兵节点（下标 1 和 n+2）
                    if t != 1 && t != self.n + 2 {
                        out.push(unsafe { self.get(t).val });
                    }
                    x = unsafe { self.get(t).r };
                }
            }
        }
    }
}

// ---------------- 块状链表（sqrt decomposition） ----------------
struct Sqrt {
    blocks: Vec<(Vec<u32>, bool)>,
    b: u32,
}

impl Sqrt {
    fn new(a: &[u32], _seed: u64, block_size: u32) -> Sqrt {
        let n = a.len() as f64;
        let auto = ((n.sqrt() * 2.0) as u32).clamp(64, 1024);
        let b = (if block_size != 0 { block_size } else { auto }) as usize;
        let mut blocks = Vec::new();
        let mut i = 0usize;
        while i < a.len() {
            let e = (i + b).min(a.len());
            blocks.push((a[i..e].to_vec(), false));
            i = e;
        }
        Sqrt {
            blocks,
            b: b as u32,
        }
    }

    fn find(&self, mut k: u32) -> (u32, u32) {
        for (i, (v, _)) in self.blocks.iter().enumerate() {
            let s = v.len() as u32;
            if k < s {
                return (i as u32, k);
            }
            k -= s;
        }
        (0, 0)
    }

    #[inline(always)]
    fn materialize(&mut self, i: u32) {
        let (v, rev) = &mut self.blocks[i as usize];
        if *rev {
            let len = v.len();
            unsafe { reverse_avx2(v, 0, len) };
            *rev = false;
        }
    }

    fn split_block(&mut self, i: u32) {
        let s = self.blocks[i as usize].0.len();
        if s <= 2 * self.b as usize {
            return;
        }
        let half = s / 2;
        let nb = (self.blocks[i as usize].0.split_off(half), false);
        self.blocks.insert(i as usize + 1, nb);
    }

    // 块长调节：过小的块与邻居合并（合并前先 materialize，保证物理顺序一致）。
    fn merge_small(&mut self, i: u32) {
        let sz = self.blocks[i as usize].0.len();
        if sz >= self.b as usize / 2 {
            return;
        }
        let ii = i as usize;
        if ii > 0 {
            let left = self.blocks[ii - 1].0.len();
            if left + sz <= 2 * self.b as usize {
                self.materialize((ii - 1) as u32);
                self.materialize(i);
                let mut rhs = std::mem::take(&mut self.blocks[ii].0);
                self.blocks[ii - 1].0.append(&mut rhs);
                self.blocks[ii - 1].1 = false;
                self.blocks.remove(ii);
                return;
            }
        }
        if ii + 1 < self.blocks.len() {
            let right = self.blocks[ii + 1].0.len();
            if sz + right <= 2 * self.b as usize {
                self.materialize(i);
                self.materialize((ii + 1) as u32);
                let mut rhs = std::mem::take(&mut self.blocks[ii + 1].0);
                self.blocks[ii].0.append(&mut rhs);
                self.blocks[ii].1 = false;
                self.blocks.remove(ii + 1);
            }
        }
    }

    fn rebalance(&mut self, i: u32) {
        if self.blocks[i as usize].0.len() > 2 * self.b as usize {
            self.split_block(i);
        } else {
            self.merge_small(i);
        }
    }

    fn reverse(&mut self, l: u32, r: u32) {
        let (bi, oi) = self.find(l);
        let (bj, oj) = self.find(r);
        if bi == bj {
            self.materialize(bi);
            let (v, _) = &mut self.blocks[bi as usize];
            unsafe { reverse_avx2(v, oi as usize, oj as usize + 1) };
            return;
        }
        self.materialize(bi);
        self.materialize(bj);
        let nr = (oj + 1) as usize;
        let mut rp = self.blocks[bj as usize].0[..nr].to_vec();
        let rp_len = rp.len();
        unsafe { reverse_avx2(&mut rp, 0, rp_len) };
        let mut lp = self.blocks[bi as usize].0[oi as usize..].to_vec();
        let lp_len = lp.len();
        unsafe { reverse_avx2(&mut lp, 0, lp_len) };
        let tail = self.blocks[bj as usize].0[(oj + 1) as usize..].to_vec();
        self.blocks[bi as usize].0.truncate(oi as usize);
        self.blocks[bi as usize].0.extend_from_slice(&rp);
        self.blocks[bj as usize].0 = lp;
        self.blocks[bj as usize].0.extend_from_slice(&tail);
        // 中间整块：先反转块列表顺序，再逐块翻转（懒标记）
        self.blocks[(bi + 1) as usize..bj as usize].reverse();
        for i in (bi + 1)..bj {
            self.blocks[i as usize].1 ^= true;
        }
        // 先处理大下标，避免增删块导致另一个下标失效
        self.rebalance(bj);
        self.rebalance(bi);
    }

    fn collect(&self, out: &mut Vec<u32>) {
        out.clear();
        for (v, rev) in &self.blocks {
            if *rev {
                for &x in v.iter().rev() {
                    out.push(x);
                }
            } else {
                for &x in v {
                    out.push(x);
                }
            }
        }
    }
}

// ---------------- bitset 思想懒标记块状链表（sqrt_bitset） ----------------
// 懒标记打包进 u32 块描述符的最高位（desc = pool_id | rev<<31）：
//  - 中间整块的区间翻转 = 一次 vpxor 0x80000000（每 8 块一条 SIMD 指令）；
//  - 块顺序反转 = 同一个 reverse_avx2（vpermd）直接作用于 desc，标记随块走；
//  - 块本体放 arena（pool 只增不减），desc/szs 是 u32 数组，split/merge 只是
//    vector insert/erase，不再有独立的 bit insert/remove。
struct SqrtBitset {
    pool: Vec<Vec<u32>>,  // arena：block id = 下标，删除后留空位
    desc: Vec<u32>,       // 逻辑块顺序：pool_id | (懒反转 << 31)
    szs: Vec<u32>,        // 与 desc 平行的块大小（find 用，避免二次间接）
    free_ids: Vec<u32>,   // 合并后回收的 pool 槽位
    b: u32,
}

impl SqrtBitset {
    const REV: u32 = 0x8000_0000;

    fn new(a: &[u32], _seed: u64, block_size: u32) -> SqrtBitset {
        let n = a.len() as f64;
        let auto = ((n.sqrt() * 2.0) as u32).clamp(64, 1024);
        let b = (if block_size != 0 { block_size } else { auto }) as usize;
        let mut pool = Vec::new();
        let mut desc = Vec::new();
        let mut szs = Vec::new();
        let free_ids = Vec::new();
        let mut i = 0usize;
        while i < a.len() {
            let e = (i + b).min(a.len());
            pool.push(a[i..e].to_vec());
            desc.push((pool.len() - 1) as u32);
            szs.push((e - i) as u32);
            i = e;
        }
        SqrtBitset {
            pool,
            desc,
            szs,
            free_ids,
            b: b as u32,
        }
    }

    fn find(&self, mut k: u32) -> (u32, u32) {
        for (i, &s) in self.szs.iter().enumerate() {
            if k < s {
                return (i as u32, k);
            }
            k -= s;
        }
        (0, 0)
    }

    #[inline(always)]
    fn bit_test(&self, pos: u32) -> bool {
        self.desc[pos as usize] & Self::REV != 0
    }

    // 区间懒标记翻转：整块区间 = vpxor 0x80000000（bitset 思想，8 块/指令）
    fn toggle_range(&mut self, l: u32, r: u32) {
        unsafe {
            let m = _mm256_set1_epi32(Self::REV as i32);
            let mut i = l as usize;
            let rr = r as usize;
            while i + 8 <= rr {
                let p = self.desc.as_mut_ptr().add(i);
                let x = _mm256_loadu_si256(p as *const __m256i);
                let y = _mm256_xor_si256(x, m);
                _mm256_storeu_si256(p as *mut __m256i, y);
                i += 8;
            }
            for d in &mut self.desc[i..rr] {
                *d ^= Self::REV;
            }
        }
    }

    fn materialize(&mut self, pos: u32) {
        if self.desc[pos as usize] & Self::REV != 0 {
            let id = (self.desc[pos as usize] & !Self::REV) as usize;
            let v = &mut self.pool[id];
            let len = v.len();
            unsafe { reverse_avx2(v, 0, len) };
            self.desc[pos as usize] &= !Self::REV;
        }
    }

    fn split_block(&mut self, pos: u32) {
        let s = self.szs[pos as usize];
        if s <= 2 * self.b {
            return;
        }
        let half = s / 2;
        let id = (self.desc[pos as usize] & !Self::REV) as usize; // 端块必已物化
        let nb = self.pool[id].split_off(half as usize);
        self.szs[pos as usize] = half;
        let nid = if let Some(f) = self.free_ids.pop() {
            self.pool[f as usize] = nb;
            f
        } else {
            self.pool.push(nb);
            (self.pool.len() - 1) as u32
        };
        self.desc.insert(pos as usize + 1, nid);
        self.szs.insert(pos as usize + 1, s - half);
    }

    fn merge_small(&mut self, pos: u32) {
        let s = self.szs[pos as usize];
        if s >= self.b / 2 {
            return;
        }
        let p = pos as usize;
        if p > 0 && self.szs[p - 1] + s <= 2 * self.b {
            self.materialize(pos - 1);
            self.materialize(pos);
            let idr = self.desc[p] & !Self::REV;
            let mut rhs = std::mem::take(&mut self.pool[idr as usize]);
            let lid = (self.desc[p - 1] & !Self::REV) as usize;
            self.pool[lid].append(&mut rhs);
            self.szs[p - 1] += s;
            self.free_ids.push(idr);
            self.desc.remove(p);
            self.szs.remove(p);
            return;
        }
        if p + 1 < self.desc.len() && s + self.szs[p + 1] <= 2 * self.b {
            self.materialize(pos);
            self.materialize(pos + 1);
            let idr = self.desc[p + 1] & !Self::REV;
            let mut rhs = std::mem::take(&mut self.pool[idr as usize]);
            let lid = (self.desc[p] & !Self::REV) as usize;
            self.pool[lid].append(&mut rhs);
            self.szs[p] += self.szs[p + 1];
            self.free_ids.push(idr);
            self.desc.remove(p + 1);
            self.szs.remove(p + 1);
        }
    }

    fn rebalance(&mut self, pos: u32) {
        if self.szs[pos as usize] > 2 * self.b {
            self.split_block(pos);
        } else {
            self.merge_small(pos);
        }
    }

    fn reverse(&mut self, l: u32, r: u32) {
        let (bi, oi) = self.find(l);
        let (bj, oj) = self.find(r);
        if bi == bj {
            self.materialize(bi);
            let id = (self.desc[bi as usize] & !Self::REV) as usize;
            let v = &mut self.pool[id];
            unsafe { reverse_avx2(v, oi as usize, oj as usize + 1) };
            return;
        }
        self.materialize(bi);
        self.materialize(bj);
        let ai = (self.desc[bi as usize] & !Self::REV) as usize;
        let di = (self.desc[bj as usize] & !Self::REV) as usize;
        let nr = (oj + 1) as usize;
        let mut rp = self.pool[di][..nr].to_vec();
        let rp_len = rp.len();
        unsafe { reverse_avx2(&mut rp, 0, rp_len) };
        let mut lp = self.pool[ai][oi as usize..].to_vec();
        let lp_len = lp.len();
        unsafe { reverse_avx2(&mut lp, 0, lp_len) };
        let tail = self.pool[di][nr..].to_vec();
        self.pool[ai].truncate(oi as usize);
        self.pool[ai].extend_from_slice(&rp);
        self.pool[di] = lp;
        self.pool[di].extend_from_slice(&tail);
        self.szs[bi as usize] = self.pool[ai].len() as u32;
        self.szs[bj as usize] = self.pool[di].len() as u32;
        // 中间块：vpermd 反转 desc/szs（标记随块走），再 vpxor 区间翻转标记
        unsafe {
            reverse_avx2(&mut self.desc, (bi + 1) as usize, bj as usize);
            reverse_avx2(&mut self.szs, (bi + 1) as usize, bj as usize);
        }
        self.toggle_range(bi + 1, bj);
        self.rebalance(bj);
        self.rebalance(bi);
    }

    fn collect(&self, out: &mut Vec<u32>) {
        out.clear();
        for i in 0..self.desc.len() {
            let d = self.desc[i];
            let v = &self.pool[(d & !Self::REV) as usize];
            if d & Self::REV != 0 {
                out.extend(v.iter().rev());
            } else {
                out.extend_from_slice(v);
            }
        }
    }
}

// ---------------- 两层 bitset 块状链表（sqrt_bitset2） ----------------
// 「bitset 套 bitset」= 高度 2 的序列 B 树逻辑：
//   内层（叶子）：每块 1 bit 内容反转标记，打包进 u32 块描述符（pool_id | REV）。
//   外层（内部节点）：每超块 2 bit（块序反转 ORD / 整块内容反转 CONT），
//         打包进外层 u32 描述符（sb_id | ORD | CONT），槽位是动态数组。
// 中间整块区间 = 外层 vpermd 反转超块顺序 + vpxor 翻 ORD/CONT 两个位；
// 端部散块先 materialize 到槽位级再按块处理。find 先跳超块 total
// （与槽位顺序无关）再扫 ≤W 个槽位：O(#superblocks + W)。
// 超块 >2W 拆、<W/2 并（B 树节点占用量维护）；块 arena + 两级 free-list。
#[derive(Clone, Copy)]
struct Loc {
    opos: u32,
    slot: u32,
    off: u32,
}

struct SB {
    descs: Vec<u32>,  // pool_id | REV
    szs: Vec<u32>,
    total: u32,
}

struct SqrtBitset2 {
    pool: Vec<Vec<u32>>,  // arena：block id = 下标，删除后留空位
    free_ids: Vec<u32>,   // 合并后回收的 pool 槽位
    sbpool: Vec<SB>,
    sb_free: Vec<u32>,
    outer: Vec<u32>,      // sb_id | ORD | CONT
    b: u32,
}

impl SqrtBitset2 {
    const REV: u32 = 0x8000_0000;   // 内层：块内容反转
    const ORD: u32 = 0x4000_0000;   // 外层：超块内块序反转
    const CONT: u32 = 0x8000_0000;  // 外层：超块内所有块内容反转
    const XMASK: u32 = !(Self::ORD | Self::CONT);
    const W: u32 = 64;              // 超块目标容量（拆 >2W / 并 <W/2）

    fn new(a: &[u32], _seed: u64, block_size: u32) -> SqrtBitset2 {
        let n = a.len() as f64;
        let auto = ((n.sqrt() * 2.0) as u32).clamp(64, 1024);
        let b = (if block_size != 0 { block_size } else { auto }) as usize;
        let mut pool = Vec::new();
        let mut i = 0usize;
        while i < a.len() {
            let e = (i + b).min(a.len());
            pool.push(a[i..e].to_vec());
            i = e;
        }
        let mut sb = SB { descs: Vec::new(), szs: Vec::new(), total: 0 };
        let mut sbpool = Vec::new();
        let mut total = 0u32;
        for (i, v) in pool.iter().enumerate() {
            if sb.descs.len() == Self::W as usize {
                sb.total = total;
                sbpool.push(sb);
                sb = SB { descs: Vec::new(), szs: Vec::new(), total: 0 };
                total = 0;
            }
            sb.descs.push(i as u32);
            sb.szs.push(v.len() as u32);
            total += v.len() as u32;
        }
        if !sb.descs.is_empty() || pool.is_empty() {
            sb.total = total;
            sbpool.push(sb);
        }
        let outer: Vec<u32> = (0..sbpool.len() as u32).collect();
        SqrtBitset2 {
            pool,
            free_ids: Vec::new(),
            sbpool,
            sb_free: Vec::new(),
            outer,
            b: b as u32,
        }
    }

    fn toggle_mask(a: &mut [u32], l: u32, r: u32, mask: u32) {
        unsafe {
            let m = _mm256_set1_epi32(mask as i32);
            let mut i = l as usize;
            let rr = r as usize;
            while i + 8 <= rr {
                let p = a.as_mut_ptr().add(i);
                let x = _mm256_loadu_si256(p as *const __m256i);
                let y = _mm256_xor_si256(x, m);
                _mm256_storeu_si256(p as *mut __m256i, y);
                i += 8;
            }
            for d in &mut a[i..rr] {
                *d ^= mask;
            }
        }
    }

    // 把超块的 ORD/CONT 懒标记落到槽位级
    fn materialize_sb(&mut self, opos: u32) {
        let d = self.outer[opos as usize];
        if d & (Self::ORD | Self::CONT) != 0 {
            let sid = (d & Self::XMASK) as usize;
            if d & Self::ORD != 0 {
                unsafe {
                    let nd = self.sbpool[sid].descs.len();
                    reverse_avx2(&mut self.sbpool[sid].descs, 0, nd);
                    let ns = self.sbpool[sid].szs.len();
                    reverse_avx2(&mut self.sbpool[sid].szs, 0, ns);
                }
            }
            if d & Self::CONT != 0 {
                let n = self.sbpool[sid].descs.len() as u32;
                Self::toggle_mask(&mut self.sbpool[sid].descs, 0, n, Self::REV);
            }
            self.outer[opos as usize] = d & Self::XMASK;
        }
    }

    fn find(&self, mut k: u32) -> Loc {
        let mut oi = 0usize;
        loop {
            let sid = (self.outer[oi] & Self::XMASK) as usize;
            let s = &self.sbpool[sid];
            if k < s.total {
                break;
            }
            k -= s.total;
            oi += 1;
        }
        let sid = (self.outer[oi] & Self::XMASK) as usize;
        let s = &self.sbpool[sid];
        let n = s.descs.len() as u32;
        let mut slot = 0u32;
        if self.outer[oi] & Self::ORD != 0 {
            let mut j = n;
            while j > 0 {
                j -= 1;
                let sz = s.szs[j as usize];
                if k < sz {
                    slot = j;
                    break;
                }
                k -= sz;
            }
        } else {
            let mut j = 0u32;
            while j < n {
                let sz = s.szs[j as usize];
                if k < sz {
                    slot = j;
                    break;
                }
                k -= sz;
                j += 1;
            }
        }
        Loc { opos: oi as u32, slot, off: k }
    }

    fn materialize_block(&mut self, opos: u32, slot: u32) {
        let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
        if self.sbpool[sid].descs[slot as usize] & Self::REV != 0 {
            let id = (self.sbpool[sid].descs[slot as usize] & !Self::REV) as usize;
            let sz = self.sbpool[sid].szs[slot as usize] as usize;
            unsafe { reverse_avx2(&mut self.pool[id], 0, sz) };
            self.sbpool[sid].descs[slot as usize] &= !Self::REV;
        }
    }

    // 元素级拆分：把超块 opos 内 slot 处的块在偏移 off 处拆成两块
    fn split_block_at(&mut self, opos: u32, slot: u32, off: u32) {
        let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
        let sz = self.sbpool[sid].szs[slot as usize];
        if off == 0 || off >= sz {
            return;
        }
        let id = (self.sbpool[sid].descs[slot as usize] & !Self::REV) as usize;
        if self.sbpool[sid].descs[slot as usize] & Self::REV != 0 {
            let len = self.pool[id].len();
            unsafe { reverse_avx2(&mut self.pool[id], 0, len) };
            self.sbpool[sid].descs[slot as usize] &= !Self::REV;
        }
        let suffix = self.pool[id].split_off(off as usize);
        let nid = if let Some(f) = self.free_ids.pop() {
            self.pool[f as usize] = suffix;
            f
        } else {
            self.pool.push(suffix);
            (self.pool.len() - 1) as u32
        };
        self.sbpool[sid].szs[slot as usize] = off;
        self.sbpool[sid].descs.insert(slot as usize + 1, nid);
        self.sbpool[sid].szs.insert(slot as usize + 1, sz - off);
        if self.sbpool[sid].descs.len() > 2 * Self::W as usize {
            self.split_superblock(opos);
        }
    }

    fn split_superblock(&mut self, opos: u32) {
        let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
        if self.sbpool[sid].descs.len() <= 2 * Self::W as usize {
            return;
        }
        let half = self.sbpool[sid].descs.len() / 2;
        let nid = if let Some(f) = self.sb_free.pop() {
            self.sbpool[f as usize].descs.clear();
            self.sbpool[f as usize].szs.clear();
            self.sbpool[f as usize].total = 0;
            f
        } else {
            self.sbpool.push(SB { descs: Vec::new(), szs: Vec::new(), total: 0 });
            (self.sbpool.len() - 1) as u32
        };
        let n = nid as usize;
        self.sbpool[n].descs = self.sbpool[sid].descs.split_off(half);
        self.sbpool[n].szs = self.sbpool[sid].szs.split_off(half);
        let t: u32 = self.sbpool[n].szs.iter().sum();
        self.sbpool[n].total = t;
        self.sbpool[sid].total -= t;
        self.outer.insert(opos as usize + 1, nid);
    }

    // 小块合并：触发条件是 sz < B，合并后 ≤ 2B 就拼（保持块数有界）
    fn merge_small_block(&mut self, opos: u32, slot: u32) {
        let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
        let cnt = self.sbpool[sid].descs.len() as u32;
        if slot >= cnt {
            return;
        }
        let sz = self.sbpool[sid].szs[slot as usize];
        if sz >= self.b {
            return;
        }
        if slot > 0 && self.sbpool[sid].szs[slot as usize - 1] + sz <= 2 * self.b {
            self.materialize_block(opos, slot - 1);
            self.materialize_block(opos, slot);
            let lid = (self.sbpool[sid].descs[slot as usize - 1] & !Self::REV) as usize;
            let rid = (self.sbpool[sid].descs[slot as usize] & !Self::REV) as usize;
            let mut rhs = std::mem::take(&mut self.pool[rid]);
            self.pool[lid].append(&mut rhs);
            self.sbpool[sid].szs[slot as usize - 1] += sz;
            self.free_ids.push(rid as u32);
            self.sbpool[sid].descs.remove(slot as usize);
            self.sbpool[sid].szs.remove(slot as usize);
            return;
        }
        if slot + 1 < cnt && sz + self.sbpool[sid].szs[slot as usize + 1] <= 2 * self.b {
            self.materialize_block(opos, slot);
            self.materialize_block(opos, slot + 1);
            let lid = (self.sbpool[sid].descs[slot as usize] & !Self::REV) as usize;
            let rid = (self.sbpool[sid].descs[slot as usize + 1] & !Self::REV) as usize;
            let mut rhs = std::mem::take(&mut self.pool[rid]);
            self.pool[lid].append(&mut rhs);
            self.sbpool[sid].szs[slot as usize] += self.sbpool[sid].szs[slot as usize + 1];
            self.free_ids.push(rid as u32);
            self.sbpool[sid].descs.remove(slot as usize + 1);
            self.sbpool[sid].szs.remove(slot as usize + 1);
        }
    }

    fn merge_small_scan(&mut self, opos: u32) {
        let mut k = 0u32;
        loop {
            let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
            if k >= self.sbpool[sid].descs.len() as u32 {
                break;
            }
            if self.sbpool[sid].szs[k as usize] < self.b {
                let before = self.sbpool[sid].descs.len();
                self.merge_small_block(opos, k);
                if self.sbpool[sid].descs.len() == before {
                    k += 1;  // 没合并成必须前进
                }
            } else {
                k += 1;
            }
        }
    }

    fn merge_superblock(&mut self, opos: u32) {
        let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
        if self.sbpool[sid].descs.len() < Self::W as usize / 2 {
            if opos > 0 {
                let lsid = (self.outer[opos as usize - 1] & Self::XMASK) as usize;
                if self.sbpool[lsid].descs.len() + self.sbpool[sid].descs.len()
                    <= 2 * Self::W as usize
                {
                    self.materialize_sb(opos - 1);
                    self.materialize_sb(opos);
                    let ssid = (self.outer[opos as usize] & Self::XMASK) as usize;
                    let mut rhs = std::mem::take(&mut self.sbpool[ssid].descs);
                    let mut rsz = std::mem::take(&mut self.sbpool[ssid].szs);
                    self.sbpool[lsid].descs.append(&mut rhs);
                    self.sbpool[lsid].szs.append(&mut rsz);
                    self.sbpool[lsid].total += self.sbpool[ssid].total;
                    self.sbpool[ssid].total = 0;
                    self.sb_free.push(ssid as u32);
                    self.outer.remove(opos as usize);
                    return;
                }
            }
            if opos + 1 < self.outer.len() as u32 {
                let rsid = (self.outer[opos as usize + 1] & Self::XMASK) as usize;
                if self.sbpool[sid].descs.len() + self.sbpool[rsid].descs.len()
                    <= 2 * Self::W as usize
                {
                    self.materialize_sb(opos);
                    self.materialize_sb(opos + 1);
                    let ssid = (self.outer[opos as usize] & Self::XMASK) as usize;
                    let rrid = (self.outer[opos as usize + 1] & Self::XMASK) as usize;
                    let mut rhs = std::mem::take(&mut self.sbpool[rrid].descs);
                    let mut rsz = std::mem::take(&mut self.sbpool[rrid].szs);
                    self.sbpool[ssid].descs.append(&mut rhs);
                    self.sbpool[ssid].szs.append(&mut rsz);
                    self.sbpool[ssid].total += self.sbpool[rrid].total;
                    self.sbpool[rrid].total = 0;
                    self.sb_free.push(rrid as u32);
                    self.outer.remove(opos as usize + 1);
                }
            }
        }
    }

    fn rebalance_sb(&mut self, opos: u32) {
        let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
        if self.sbpool[sid].descs.len() > 2 * Self::W as usize {
            self.split_superblock(opos);
        } else if self.sbpool[sid].descs.len() < Self::W as usize / 2 {
            self.merge_superblock(opos);
        }
    }

    // 同一超块内的块区间反转（块序 + 每块内容）
    fn reverse_slot_range(&mut self, opos: u32, si: u32, sj: u32) {
        let sid = (self.outer[opos as usize] & Self::XMASK) as usize;
        unsafe {
            reverse_avx2(&mut self.sbpool[sid].descs, si as usize, sj as usize + 1);
            reverse_avx2(&mut self.sbpool[sid].szs, si as usize, sj as usize + 1);
        }
        Self::toggle_mask(&mut self.sbpool[sid].descs, si, sj + 1, Self::REV);
    }

    fn reverse(&mut self, l: u32, r: u32) {
        let mut a = self.find(l);
        let mut b = self.find(r);
        self.materialize_sb(a.opos);
        if b.opos != a.opos {
            self.materialize_sb(b.opos);
        }
        // 物化把 ORD 落到物理槽位、CONT 落到块位；之后物理==逻辑，
        // 而 find 在 ORD 置位时返回的是物理槽位，必须先重新定位。
        a = self.find(l);
        b = self.find(r);
        if a.off > 0 {
            self.split_block_at(a.opos, a.slot, a.off);
        }
        let b2 = self.find(r);
        {
            let sid = (self.outer[b2.opos as usize] & Self::XMASK) as usize;
            if b2.off + 1 < self.sbpool[sid].szs[b2.slot as usize] {
                self.split_block_at(b2.opos, b2.slot, b2.off + 1);
            }
        }
        let a2 = self.find(l);
        let b3 = self.find(r);
        if a2.opos == b3.opos {
            self.materialize_sb(a2.opos);
            self.reverse_slot_range(a2.opos, a2.slot, b3.slot);
            return;
        }
        let oi = a2.opos;
        let oj = b3.opos;
        let si = a2.slot;
        let sj = b3.slot;
        let ai = (self.outer[oi as usize] & Self::XMASK) as usize;
        let di = (self.outer[oj as usize] & Self::XMASK) as usize;
        let ac = self.sbpool[ai].descs.len() as u32;
        let bc = self.sbpool[di].descs.len() as u32;
        let npre = si;
        let nrb = sj + 1;
        let nla = ac - si;
        let _ntail = bc - sj - 1;
        // rp_chunk = B[0..sj] 块序 + 内容反转
        let mut rp_d = self.sbpool[di].descs[..nrb as usize].to_vec();
        let mut rp_s = self.sbpool[di].szs[..nrb as usize].to_vec();
        unsafe {
            let nd = rp_d.len();
            reverse_avx2(&mut rp_d, 0, nd);
            let ns = rp_s.len();
            reverse_avx2(&mut rp_s, 0, ns);
        }
        Self::toggle_mask(&mut rp_d, 0, nrb, Self::REV);
        // lp_chunk = A[si..) 块序 + 内容反转
        let mut lp_d = self.sbpool[ai].descs[si as usize..].to_vec();
        let mut lp_s = self.sbpool[ai].szs[si as usize..].to_vec();
        unsafe {
            let nd = lp_d.len();
            reverse_avx2(&mut lp_d, 0, nd);
            let ns = lp_s.len();
            reverse_avx2(&mut lp_s, 0, ns);
        }
        Self::toggle_mask(&mut lp_d, 0, nla, Self::REV);
        // tail = B[sj+1..)
        let tail_d = self.sbpool[di].descs[sj as usize + 1..].to_vec();
        let tail_s = self.sbpool[di].szs[sj as usize + 1..].to_vec();
        // 中间：外层顺序反转 + ORD/CONT 两个标记位（vpermd + vpxor）
        unsafe {
            reverse_avx2(&mut self.outer, (oi + 1) as usize, oj as usize);
        }
        Self::toggle_mask(&mut self.outer, oi + 1, oj, Self::ORD);
        Self::toggle_mask(&mut self.outer, oi + 1, oj, Self::CONT);
        // 写回 A = prefix + rp_chunk
        self.sbpool[ai].descs.truncate(npre as usize);
        self.sbpool[ai].descs.extend_from_slice(&rp_d);
        self.sbpool[ai].szs.truncate(npre as usize);
        self.sbpool[ai].szs.extend_from_slice(&rp_s);
        self.sbpool[ai].total = self.sbpool[ai].szs.iter().sum();
        // 写回 B = lp_chunk + tail
        self.sbpool[di].descs = lp_d;
        self.sbpool[di].descs.extend_from_slice(&tail_d);
        self.sbpool[di].szs = lp_s;
        self.sbpool[di].szs.extend_from_slice(&tail_s);
        self.sbpool[di].total = self.sbpool[di].szs.iter().sum();
        // 小块合并（保持块数有界），再做超块结构维护
        self.merge_small_scan(oi);
        self.merge_small_scan(oj);
        self.rebalance_sb(oj);
        self.rebalance_sb(oi);
    }

    fn collect(&self, out: &mut Vec<u32>) {
        out.clear();
        for o in 0..self.outer.len() {
            let od = self.outer[o];
            let s = &self.sbpool[(od & Self::XMASK) as usize];
            let n = s.descs.len() as u32;
            for k in 0..n {
                let slot = if od & Self::ORD != 0 { n - 1 - k } else { k };
                let id = (s.descs[slot as usize] & !Self::REV) as usize;
                let rev = (s.descs[slot as usize] & Self::REV != 0) ^ (od & Self::CONT != 0);
                let v = &self.pool[id];
                if rev {
                    out.extend(v.iter().rev());
                } else {
                    out.extend_from_slice(v);
                }
            }
        }
    }
}

// ---------------- 方法分发 ----------------
const METHODS: [&str; 8] =
    ["brute", "std", "avx2", "treap", "splay", "sqrt", "sqrt_bitset", "sqrt_bitset2"];

#[inline(always)]
unsafe fn apply_reverse(m: usize, a: &mut [u32], l: u32, r: u32) {
    match m {
        0 => reverse_scalar(a, l as usize, r as usize + 1),
        1 => reverse_std(a, l as usize, r as usize + 1),
        2 => reverse_avx2(a, l as usize, r as usize + 1),
        _ => unreachable!(),
    }
}

enum DS {
    Array { a: Vec<u32>, kind: usize },
    Treap(Treap),
    Splay(Splay),
    Sqrt(Sqrt),
    SqrtBitset(SqrtBitset),
    SqrtBitset2(SqrtBitset2),
}

impl DS {
    fn new(m: usize, a: &[u32], seed: u64, block_size: u32) -> DS {
        match m {
            0 | 1 | 2 => DS::Array {
                a: a.to_vec(),
                kind: m,
            },
            3 => DS::Treap(Treap::new(a, seed)),
            4 => DS::Splay(Splay::new(a, seed)),
            5 => DS::Sqrt(Sqrt::new(a, seed, block_size)),
            6 => DS::SqrtBitset(SqrtBitset::new(a, seed, block_size)),
            7 => DS::SqrtBitset2(SqrtBitset2::new(a, seed, block_size)),
            _ => unreachable!(),
        }
    }

    fn reverse(&mut self, l: u32, r: u32) {
        match self {
            DS::Array { a, kind } => unsafe { apply_reverse(*kind, a, l, r) },
            DS::Treap(t) => t.reverse(l, r),
            DS::Splay(s) => s.reverse(l, r),
            DS::Sqrt(q) => q.reverse(l, r),
            DS::SqrtBitset(q) => q.reverse(l, r),
            DS::SqrtBitset2(q) => q.reverse(l, r),
        }
    }

    fn collect(&mut self, out: &mut Vec<u32>) {
        match self {
            DS::Array { a, .. } => {
                out.clear();
                out.extend_from_slice(a);
            }
            DS::Treap(t) => t.collect(out),
            DS::Splay(s) => s.collect(out),
            DS::Sqrt(q) => q.collect(out),
            DS::SqrtBitset(q) => q.collect(out),
            DS::SqrtBitset2(q) => q.collect(out),
        }
    }

    // 按位置的加权 checksum：反转会改变序列，因此不会被编译器当常量优化掉。
    fn weighted(&mut self, w: &[u64], out: &mut Vec<u32>) -> u64 {
        self.collect(out);
        out.iter().zip(w).map(|(&x, &wi)| x as u64 * wi).sum()
    }
}

// ---------------- 正确性验证 ----------------
fn verify_length(m: usize, n: u32, l: u32) {
    if m == 0 {
        return;
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ l as u64 ^ m as u64;
    let mut a: Vec<u32> = (0..n).map(|_| (splitmix64(&mut seed) & 0xffff) as u32).collect();
    let mut b = a.clone();
    let mut s = seed ^ 0x1234_5678_9abc_def0;
    let mut got = Vec::with_capacity(n as usize);
    for _ in 0..8 {
        let st = rnd(&mut s, n - l + 1);
        let r = st + l; // [st, r) 半开
        unsafe { reverse_scalar(&mut a, st as usize, r as usize) };
        let mut ds = DS::new(m, &b, seed, 0);
        ds.reverse(st, r - 1);
        ds.collect(&mut got);
        if got != a {
            eprintln!("VERIFY FAIL length L={} m={}", l, METHODS[m]);
            std::process::abort();
        }
        b = a.clone();
    }
}

fn verify_batch(m: usize, n: u32, tmpl: &[u32], block_size: u32) {
    if m == 0 {
        return;
    }
    let mut a = tmpl.to_vec();
    let mut ds = DS::new(m, tmpl, 0xdead_beef ^ m as u64 ^ n as u64, block_size);
    let mut s = 0xabcd_ef01_2345_6789u64 ^ m as u64 ^ n as u64;
    for _ in 0..200 {
        let mut l = rnd(&mut s, n);
        let mut r = rnd(&mut s, n);
        if l > r {
            std::mem::swap(&mut l, &mut r);
        }
        unsafe { reverse_scalar(&mut a, l as usize, r as usize + 1) };
        ds.reverse(l, r);
    }
    let mut got = Vec::with_capacity(n as usize);
    ds.collect(&mut got);
    if got != a {
        eprintln!("VERIFY FAIL batch n={} m={}", n, METHODS[m]);
        std::process::abort();
    }
}

// ---------------- length 模式 ----------------
fn time_length_once(m: usize, tmpl: &[u32], l: u32, q: u64, seed: u64, w: &[u64]) -> f64 {
    let mut ds = DS::new(m, tmpl, seed ^ 0x5eed, 0);
    let mut s = seed;
    let t0 = Instant::now();
    for _ in 0..q {
        let st = rnd(&mut s, LEN_N - l + 1);
        ds.reverse(st, st + l - 1);
    }
    let t1 = t0.elapsed().as_secs_f64() * 1e3;
    // 加权 checksum 不计入计时，但放在计时后执行 + black_box，
    // 防止整段 ops 被优化掉（普通求和是反转不变量，编译器会把它当常量）。
    let mut out = Vec::with_capacity(tmpl.len());
    let c = ds.weighted(w, &mut out);
    black_box(c);
    t1
}

fn measure_length() {
    let mut seed = 0x5a5a_5a5a_5a5a_5a5au64;
    let tmpl: Vec<u32> = (0..LEN_N)
        .map(|_| (splitmix64(&mut seed) & 0xffff) as u32)
        .collect();
    let mut wseed = 0x7777_7777_7777_7777u64;
    let w: Vec<u64> = (0..LEN_N).map(|_| splitmix64(&mut wseed)).collect();

    let mut l = 32u32;
    while l <= LEN_N {
        for m in 0..METHODS.len() {
            verify_length(m, LEN_N, l);
            let mut q = 1u64;
            while q <= Q_CAP {
                let t = time_length_once(m, &tmpl, l, q, 0x1111 ^ m as u64 ^ l as u64, &w);
                if t >= TARGET_MS {
                    break;
                }
                q <<= 1;
            }
            if q > Q_CAP {
                q = Q_CAP;
            }
            let mut times = Vec::new();
            for r in 0..ROUNDS {
                times.push(time_length_once(
                    m,
                    &tmpl,
                    l,
                    q,
                    0x2222 ^ m as u64 ^ l as u64 ^ ((r as u64) << 32),
                    &w,
                ));
            }
            let ns = median(times) / q as f64 * 1e6;
            println!("length,{},{},{},{:.3}", LEN_N, l, METHODS[m], ns);
        }
        l <<= 1;
    }
}

// ---------------- batch 模式 ----------------
fn time_batch_once(m: usize, n: u32, tmpl: &[u32], seed: u64, w: &[u64], block_size: u32) -> f64 {
    let t0 = Instant::now();
    let mut ds = DS::new(m, tmpl, seed ^ 0xbeef, block_size);
    let mut s = seed;
    for _ in 0..OPS {
        let mut l = rnd(&mut s, n);
        let mut r = rnd(&mut s, n);
        if l > r {
            std::mem::swap(&mut l, &mut r);
        }
        ds.reverse(l, r);
    }
    let mut out = Vec::with_capacity(n as usize);
    let c = ds.weighted(w, &mut out);
    let t1 = t0.elapsed().as_secs_f64() * 1e3;
    black_box(c);
    t1
}

fn measure_batch() {
    let mut n = 1024u32;
    while n <= (1 << 20) {
        let mut seed = 0xabcde12345u64 ^ n as u64;
        let tmpl: Vec<u32> = (0..n).map(|_| (splitmix64(&mut seed) & 0xffff) as u32).collect();
        let mut wseed = 0x8888_8888_8888_8888u64 ^ n as u64;
        let w: Vec<u64> = (0..n).map(|_| splitmix64(&mut wseed)).collect();
        for m in 0..METHODS.len() {
            verify_batch(m, n, &tmpl, 0);
        }
        for m in 0..METHODS.len() {
            let mut times = Vec::new();
            for r in 0..ROUNDS {
                times.push(time_batch_once(
                    m,
                    n,
                    &tmpl,
                    0x3333 ^ m as u64 ^ n as u64 ^ ((r as u64) << 32),
                    &w,
                    0,
                ));
            }
            let ms = median(times);
            println!("batch,{},0,{},{:.3}", n, METHODS[m], ms);
        }
        n <<= 1;
    }
}

// ---------------- bsweep 模式：固定 batch 场景扫块长 B ----------------
fn measure_bsweep() {
    let bs: [u32; 8] = [64, 128, 256, 512, 1024, 2048, 4096, 0]; // 0 = auto
    for n in [1u32 << 16, 1 << 20] {
        let mut seed = 0x0bad_c0deu64 ^ n as u64;
        let tmpl: Vec<u32> = (0..n).map(|_| (splitmix64(&mut seed) & 0xffff) as u32).collect();
        let mut wseed = 0x9999_9999_9999_9999u64 ^ n as u64;
        let w: Vec<u64> = (0..n).map(|_| splitmix64(&mut wseed)).collect();
        for &b in &bs {
            for m in [5usize, 6, 7] {
                verify_batch(m, n, &tmpl, b);
                let mut times = Vec::new();
                for r in 0..ROUNDS {
                    times.push(time_batch_once(
                        m,
                        n,
                        &tmpl,
                        0x4444 ^ m as u64 ^ n as u64 ^ b as u64 ^ ((r as u64) << 32),
                        &w,
                        b,
                    ));
                }
                let ms = median(times);
                println!("bsweep,{},{},{},{:.3}", n, b, METHODS[m], ms);
            }
        }
    }
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    let only_length = arg == "length";
    let only_batch = arg == "batch";
    let only_bsweep = arg == "bsweep";
    if !only_batch && !only_bsweep {
        measure_length();
    }
    if !only_length && !only_bsweep {
        measure_batch();
    }
    if only_bsweep {
        measure_bsweep();
    }
}
