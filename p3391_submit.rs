// P3391 【模板】文艺平衡树 —— Rust 提交版（sqrt_bitset，与 C++ 版同构）
//
// 结构：懒标记位图化块状链表
//   - 块大小 B ≈ 2√n，>2B 拆块、<B/2 与邻居合并，块数保持 O(n/B)
//   - 懒反转标记打包进 u32 块描述符最高位：
//       整块区间翻转 = desc 区间反转（块序 + 标记随块走）+ 标记位翻转
//       端部散块先物化再按元素反转
//   - 块本体放 arena + free-list 回收
// 快速 I/O：read_to_end 一次读入 + 手写解析；输出一次性 write_all。
//
// 编译：rustc -O p3391_submit.rs -o p3391

use std::io::{self, Read, Write};

const REV: u32 = 0x8000_0000;

struct SqrtBitset {
    pool: Vec<Vec<u32>>,  // arena：块本体，id = 下标
    free_ids: Vec<u32>,   // 合并后回收的槽位
    desc: Vec<u32>,       // 逻辑块顺序：pool_id | (懒反转 << 31)
    szs: Vec<u32>,        // 与 desc 平行的块大小
    b: u32,
}

impl SqrtBitset {
    fn new(n: u32) -> Self {
        let b = ((n as f64).sqrt() * 2.0) as u32;
        let b = b.clamp(64, 1024);
        let mut pool = Vec::new();
        let mut desc = Vec::new();
        let mut szs = Vec::new();
        let mut i = 0u32;
        while i < n {
            let e = (i + b).min(n);
            pool.push((i + 1..=e).collect());
            desc.push((pool.len() - 1) as u32);
            szs.push(e - i);
            i = e;
        }
        SqrtBitset {
            pool,
            free_ids: Vec::new(),
            desc,
            szs,
            b,
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

    fn materialize(&mut self, pos: u32) {
        if self.desc[pos as usize] & REV != 0 {
            let id = (self.desc[pos as usize] & !REV) as usize;
            self.pool[id].reverse();
            self.desc[pos as usize] &= !REV;
        }
    }

    fn split_block(&mut self, pos: u32) {
        let s = self.szs[pos as usize];
        if s <= 2 * self.b {
            return;
        }
        let half = s / 2;
        let id = (self.desc[pos as usize] & !REV) as usize; // 端块必已物化
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
            let idr = self.desc[p] & !REV;
            let mut rhs = std::mem::take(&mut self.pool[idr as usize]);
            let lid = (self.desc[p - 1] & !REV) as usize;
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
            let idr = self.desc[p + 1] & !REV;
            let mut rhs = std::mem::take(&mut self.pool[idr as usize]);
            let lid = (self.desc[p] & !REV) as usize;
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
            let id = (self.desc[bi as usize] & !REV) as usize;
            self.pool[id][oi as usize..=oj as usize].reverse();
            return;
        }
        self.materialize(bi);
        self.materialize(bj);
        let ai = (self.desc[bi as usize] & !REV) as usize;
        let di = (self.desc[bj as usize] & !REV) as usize;
        let nr = (oj + 1) as usize;
        let mut rp = self.pool[di][..nr].to_vec();
        rp.reverse();
        let mut lp = self.pool[ai][oi as usize..].to_vec();
        lp.reverse();
        let tail = self.pool[di][nr..].to_vec();
        self.pool[ai].truncate(oi as usize);
        self.pool[ai].extend_from_slice(&rp);
        self.pool[di] = lp;
        self.pool[di].extend_from_slice(&tail);
        self.szs[bi as usize] = self.pool[ai].len() as u32;
        self.szs[bj as usize] = self.pool[di].len() as u32;
        // 中间整块：块序反转（标记随块走）+ 区间懒标记翻转
        self.desc[(bi + 1) as usize..bj as usize].reverse();
        self.szs[(bi + 1) as usize..bj as usize].reverse();
        for d in &mut self.desc[(bi + 1) as usize..bj as usize] {
            *d ^= REV;
        }
        self.rebalance(bj);
        self.rebalance(bi);
    }

    fn collect(&self, out: &mut Vec<u32>) {
        out.clear();
        for i in 0..self.desc.len() {
            let d = self.desc[i];
            let v = &self.pool[(d & !REV) as usize];
            if d & REV != 0 {
                out.extend(v.iter().rev());
            } else {
                out.extend_from_slice(v);
            }
        }
    }
}

// ---------------- 快速 I/O ----------------
struct FastIn {
    buf: Vec<u8>,
    pos: usize,
}

impl FastIn {
    fn new() -> Self {
        let mut buf = Vec::new();
        io::stdin().lock().read_to_end(&mut buf).expect("read stdin");
        FastIn { buf, pos: 0 }
    }

    fn next_u32(&mut self) -> u32 {
        let b = &self.buf;
        let mut p = self.pos;
        while p < b.len() && b[p] <= b' ' {
            p += 1;
        }
        let mut x = 0u32;
        while p < b.len() && b[p] > b' ' {
            x = x * 10 + (b[p] - b'0') as u32;
            p += 1;
        }
        self.pos = p;
        x
    }
}

fn append_u32(out: &mut Vec<u8>, mut x: u32) {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (x % 10) as u8;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    out.extend_from_slice(&tmp[i..]);
}

fn main() {
    let mut inp = FastIn::new();
    let n = inp.next_u32();
    let m = inp.next_u32();
    let mut sol = SqrtBitset::new(n);
    for _ in 0..m {
        let l = inp.next_u32();
        let r = inp.next_u32();
        sol.reverse(l - 1, r - 1);
    }
    let mut vals = Vec::with_capacity(n as usize);
    sol.collect(&mut vals);
    let mut out = Vec::with_capacity(n as usize * 8);
    for (i, &v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(b' ');
        }
        append_u32(&mut out, v);
    }
    out.push(b'\n');
    io::stdout().lock().write_all(&out).expect("write stdout");
}
