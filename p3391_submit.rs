#![allow(dead_code)] // 模板 API: 未用到的解析方法在 CF 上不警告

//! # CF 快速 IO 最终模板 (Rust 1.89, edition 2024, x86_64)
//!
//! 单文件, 纯 std, 无 libc。**stdin 必须是普通文件** (CF 用 `< file` 重定向, 满足)。
//!
//! 读: `fstat` 拿大小 → `mmap` 零拷贝 → SIMD/SWAR 手写解析。零 read 系统调用。
//! 写: 内存缓冲 + 手写 itoa → 一次 `write` 刷出。
//! 实测: 190MiB 输入全管线 0.20s (std 写法 0.46s)。
//!
//! 用法: 贴到 `main` 前面; 本地 `#[path = "template_io_final.rs"] mod fastio;`
//!
//! ```ignore
//! let mut r = fastio::FastIn::new();
//! let mut w = fastio::FastOut::new();
//! let n = r.next_usize();
//! let mut s = 0i64;
//! for _ in 0..n { s += r.next_i64(); }
//! w.write_i64(s);
//! w.ln();
//! // 退出时 Drop 自动 flush
//! ```

use core::arch::asm;
use core::arch::x86_64 as simd;

// ---------------- 系统调用 (x86_64) ----------------

const SYS_WRITE: u64 = 1;
const SYS_FSTAT: u64 = 5;
const SYS_MMAP: u64 = 9;
const SYS_MUNMAP: u64 = 11;

const PROT_READ: u64 = 1;
const MAP_PRIVATE: u64 = 2;

/// 3 参数系统调用, 返回 i64 (负数 = -errno)
///
/// # Safety
/// 参数必须符合目标系统调用的 ABI。
#[inline(always)]
unsafe fn sc3(n: u64, a1: u64, a2: u64, a3: u64) -> i64 {
    let r: i64;
    unsafe {
        asm!(
            "syscall",
            inlateout("rax") n as i64 => r,
            in("rdi") a1,
            in("rsi") a2,
            in("rdx") a3,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    r
}

/// 6 参数系统调用 (mmap 用)
///
/// # Safety
/// 参数必须符合目标系统调用的 ABI。
#[inline(always)]
unsafe fn sc6(n: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64, a6: u64) -> i64 {
    let r: i64;
    unsafe {
        asm!(
            "syscall",
            inlateout("rax") n as i64 => r,
            in("rdi") a1,
            in("rsi") a2,
            in("rdx") a3,
            in("r10") a4,
            in("r8") a5,
            in("r9") a6,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    r
}

// ---------------- SIMD / SWAR 数字解析 ----------------
//
// 参考实现:
// - simdjson: parse_eight_digits_unrolled (SSSE3/SSE4.1)
// - atoi_simd: 16 位数字一次 `maddubs -> madd -> pack -> madd`
// - Rust std dec2flt: SWAR 8 位数字 3 次乘法

const POW10: [u64; 17] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
];

/// 一次解析 8 个 ASCII 数字 (SWAR, 3 次乘法)。
/// 输入必须是恰好 8 个数字的小端字节。
#[inline(always)]
fn parse8_swar(v: u64) -> u64 {
    const MASK: u64 = 0x0000_00FF_0000_00FF;
    const MUL1: u64 = 0x000F_4240_0000_0064;
    const MUL2: u64 = 0x0000_2710_0000_0001;
    let v = v.wrapping_sub(0x3030_3030_3030_3030);
    let v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let v1 = (v & MASK).wrapping_mul(MUL1);
    let v2 = ((v >> 16) & MASK).wrapping_mul(MUL2);
    ((v1.wrapping_add(v2) >> 32) as u32) as u64
}

/// 一次检查 8 个字节是否全是数字 (simdjson 的 is_made_of_eight_digits_fast)。
#[inline(always)]
fn is_8digits(v: u64) -> bool {
    (v & 0xF0F0_F0F0_F0F0_F0F0
        | (((v + 0x0606_0606_0606_0606) & 0xF0F0_F0F0_F0F0_F0F0) >> 4))
        == 0x3333_3333_3333_3333
}

/// 一次解析 1..=16 个 ASCII 数字 (SSSE3 + SSE4.1)。
/// `chunk` 低 `len` 个字节是数字。
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn parse16_sse_reg(chunk: simd::__m128i, len: usize) -> u64 {
    // 把数字整体移到 128 位寄存器的高位，低位补零
    let chunk = match len {
        16 => chunk,
        15 => simd::_mm_bslli_si128(chunk, 1),
        14 => simd::_mm_bslli_si128(chunk, 2),
        13 => simd::_mm_bslli_si128(chunk, 3),
        12 => simd::_mm_bslli_si128(chunk, 4),
        11 => simd::_mm_bslli_si128(chunk, 5),
        10 => simd::_mm_bslli_si128(chunk, 6),
        9 => simd::_mm_bslli_si128(chunk, 7),
        8 => simd::_mm_bslli_si128(chunk, 8),
        7 => simd::_mm_bslli_si128(chunk, 9),
        6 => simd::_mm_bslli_si128(chunk, 10),
        5 => simd::_mm_bslli_si128(chunk, 11),
        4 => simd::_mm_bslli_si128(chunk, 12),
        3 => simd::_mm_bslli_si128(chunk, 13),
        2 => simd::_mm_bslli_si128(chunk, 14),
        1 => simd::_mm_bslli_si128(chunk, 15),
        _ => return 0,
    };

    // ASCII '0'..'9' -> 0..9
    let chunk = simd::_mm_and_si128(chunk, simd::_mm_set1_epi8(0x0F));

    // 两两合并: ab -> 10*a+b
    let chunk = simd::_mm_maddubs_epi16(
        chunk,
        simd::_mm_set_epi8(1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10),
    );
    // 四四合并: 10*a+b 与 10*c+d -> 1000*a+100*b+10*c+d
    let chunk = simd::_mm_madd_epi16(chunk, simd::_mm_set_epi16(1, 100, 1, 100, 1, 100, 1, 100));
    // 压缩成 4 个 u16 组
    let chunk = simd::_mm_packus_epi32(chunk, chunk);
    // 八八合并
    let chunk = simd::_mm_madd_epi16(
        chunk,
        simd::_mm_set_epi16(0, 0, 0, 0, 1, 10000, 1, 10000),
    );
    let arr: [u32; 4] = core::mem::transmute(chunk);
    arr[0] as u64 * 100_000_000 + arr[1] as u64
}

/// 一次解析 1..=16 个 ASCII 数字 (SSSE3 + SSE4.1)。
/// `buf` 已清零，前 `len` 个字节是数字。
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn parse16_sse(buf: &[u8; 16], len: usize) -> u64 {
    let chunk = simd::_mm_loadu_si128(buf.as_ptr().cast());
    parse16_sse_reg(chunk, len)
}

/// AVX2: 一次解析 1..=16 个 ASCII 数字。
/// 数字先对齐到 128 位块的高位，再广播到 256 位寄存器的两个 128 位半区，
/// 用 `_mm256_maddubs_epi16` / `_mm256_madd_epi16` 同时算出前 8 位和后 8 位。
#[target_feature(enable = "avx2,ssse3,sse4.1")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn parse16_avx2_high(v16: simd::__m128i, len: usize) -> u64 {
    let v16 = match len {
        16 => v16,
        15 => simd::_mm_bslli_si128(v16, 1),
        14 => simd::_mm_bslli_si128(v16, 2),
        13 => simd::_mm_bslli_si128(v16, 3),
        12 => simd::_mm_bslli_si128(v16, 4),
        11 => simd::_mm_bslli_si128(v16, 5),
        10 => simd::_mm_bslli_si128(v16, 6),
        9 => simd::_mm_bslli_si128(v16, 7),
        8 => simd::_mm_bslli_si128(v16, 8),
        7 => simd::_mm_bslli_si128(v16, 9),
        6 => simd::_mm_bslli_si128(v16, 10),
        5 => simd::_mm_bslli_si128(v16, 11),
        4 => simd::_mm_bslli_si128(v16, 12),
        3 => simd::_mm_bslli_si128(v16, 13),
        2 => simd::_mm_bslli_si128(v16, 14),
        1 => simd::_mm_bslli_si128(v16, 15),
        _ => return 0,
    };
    // ASCII '0'..'9' -> 0..9
    let v16 = simd::_mm_and_si128(v16, simd::_mm_set1_epi8(0x0F));
    // 广播到两个半区：低半区算前 8 位，高半区算后 8 位
    let v = simd::_mm256_broadcastsi128_si256(v16);

    // 两两合并
    let mult = simd::_mm256_set_epi8(
        1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1,
        10, 1, 10, 1, 10, 1, 10,
    );
    let chunk = simd::_mm256_maddubs_epi16(v, mult);
    // 四四合并
    let mult = simd::_mm256_set_epi16(
        1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100,
    );
    let chunk = simd::_mm256_madd_epi16(chunk, mult);
    let chunk = simd::_mm256_packus_epi32(chunk, chunk);
    // 八八合并：两个半区分别得到 8 位数字的值
    let chunk = simd::_mm256_madd_epi16(
        chunk,
        simd::_mm256_set_epi16(
            1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000, 1, 10000,
        ),
    );
    let arr: [u32; 8] = core::mem::transmute(chunk);
    arr[0] as u64 * 100_000_000 + arr[1] as u64
}

/// AVX2: 从 32 字节寄存器中跳过 `cnt` 个空白后，用 mm256 解析前 `len` 位数字。
#[target_feature(enable = "avx2,ssse3,sse4.1")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn parse16_avx2_shifted(v32: simd::__m256i, cnt: usize, len: usize) -> u64 {
    let v16 = simd::_mm256_castsi256_si128(v32);
    // `_mm_srli_si128` 需要编译期立即数，cnt 只有 0..=15，展开成 match
    let v16 = match cnt {
        0 => v16,
        1 => simd::_mm_srli_si128(v16, 1),
        2 => simd::_mm_srli_si128(v16, 2),
        3 => simd::_mm_srli_si128(v16, 3),
        4 => simd::_mm_srli_si128(v16, 4),
        5 => simd::_mm_srli_si128(v16, 5),
        6 => simd::_mm_srli_si128(v16, 6),
        7 => simd::_mm_srli_si128(v16, 7),
        8 => simd::_mm_srli_si128(v16, 8),
        9 => simd::_mm_srli_si128(v16, 9),
        10 => simd::_mm_srli_si128(v16, 10),
        11 => simd::_mm_srli_si128(v16, 11),
        12 => simd::_mm_srli_si128(v16, 12),
        13 => simd::_mm_srli_si128(v16, 13),
        14 => simd::_mm_srli_si128(v16, 14),
        15 => simd::_mm_srli_si128(v16, 15),
        _ => return 0,
    };
    parse16_avx2_high(v16, len)
}

/// SSE4.1/SSSE3: 读取 16 字节并直接解析前 1..=16 位数字。
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn load16_parse(b: &[u8], p: usize) -> (usize, u64) {
    let v = simd::_mm_loadu_si128(b.as_ptr().add(p).cast());
    let hi = simd::_mm_cmpgt_epi8(v, simd::_mm_set1_epi8(b'9' as i8));
    let lo = simd::_mm_cmpgt_epi8(simd::_mm_set1_epi8(b'0' as i8), v);
    let bits = simd::_mm_movemask_epi8(simd::_mm_or_si128(hi, lo)) as u32;
    let len = bits.trailing_zeros().min(16) as usize;
    let val = parse16_sse_reg(v, len);
    (len, val)
}

/// AVX2 + SSSE3/SSE4.1: 读取 32 字节；若前 1..=16 位就是数字，直接解析。
#[target_feature(enable = "avx2,ssse3,sse4.1")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn load32_parse16(b: &[u8], p: usize) -> (usize, u64) {
    let v = simd::_mm256_loadu_si256(b.as_ptr().add(p).cast());
    let hi = simd::_mm256_cmpgt_epi8(v, simd::_mm256_set1_epi8(b'9' as i8));
    let lo = simd::_mm256_cmpgt_epi8(simd::_mm256_set1_epi8(b'0' as i8), v);
    let bits = simd::_mm256_movemask_epi8(simd::_mm256_or_si256(hi, lo)) as u32;
    let len = bits.trailing_zeros().min(32) as usize;
    if len <= 16 {
        let v16 = simd::_mm256_castsi256_si128(v);
        let val = parse16_avx2_high(v16, len);
        (len, val)
    } else {
        (len, 0)
    }
}

/// AVX2 + SSSE3/SSE4.1: 读取 16 字节，定位长度后用 mm256 解析。
#[target_feature(enable = "avx2,ssse3,sse4.1")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn load16_parse_avx(b: &[u8], p: usize) -> (usize, u64) {
    let v16 = simd::_mm_loadu_si128(b.as_ptr().add(p).cast());
    let hi = simd::_mm_cmpgt_epi8(v16, simd::_mm_set1_epi8(b'9' as i8));
    let lo = simd::_mm_cmpgt_epi8(simd::_mm_set1_epi8(b'0' as i8), v16);
    let bits = simd::_mm_movemask_epi8(simd::_mm_or_si128(hi, lo)) as u32;
    let len = bits.trailing_zeros().min(16) as usize;
    let val = parse16_avx2_high(v16, len);
    (len, val)
}

/// SSE2: 读取 16 字节，返回“空白位图” (1 = 空白)。
#[target_feature(enable = "sse2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn ws16(b: &[u8], p: usize) -> u32 {
    let v = simd::_mm_loadu_si128(b.as_ptr().add(p).cast());
    let x = simd::_mm_xor_si128(v, simd::_mm_set1_epi8(0x80u8 as i8));
    let m = simd::_mm_cmpgt_epi8(
        simd::_mm_set1_epi8(((b' ' as u8 ^ 0x80) + 1) as i8),
        x,
    );
    simd::_mm_movemask_epi8(m) as u32
}

/// SSE2: 读取 16 字节，返回“非数字位图” (1 = 不是 0-9)。
#[target_feature(enable = "sse2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn non_digit16(b: &[u8], p: usize) -> u32 {
    let v = simd::_mm_loadu_si128(b.as_ptr().add(p).cast());
    let hi = simd::_mm_cmpgt_epi8(v, simd::_mm_set1_epi8(b'9' as i8));
    let lo = simd::_mm_cmpgt_epi8(simd::_mm_set1_epi8(b'0' as i8), v);
    simd::_mm_movemask_epi8(simd::_mm_or_si128(hi, lo)) as u32
}

/// AVX2: 读取 32 字节，返回“空白位图” (1 = 空白)。
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn ws32(b: &[u8], p: usize) -> u32 {
    let v = simd::_mm256_loadu_si256(b.as_ptr().add(p).cast());
    avx2_ws_bits(v)
}

/// AVX2: 返回 32 字节寄存器里的“空白位图”。
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx2_ws_bits(v: simd::__m256i) -> u32 {
    let x = simd::_mm256_xor_si256(v, simd::_mm256_set1_epi8(0x80u8 as i8));
    let m = simd::_mm256_cmpgt_epi8(
        simd::_mm256_set1_epi8(((b' ' as u8 ^ 0x80) + 1) as i8),
        x,
    );
    simd::_mm256_movemask_epi8(m) as u32
}

/// AVX2: 读取 32 字节，返回“非数字位图” (1 = 不是 0-9)。
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn non_digit32(b: &[u8], p: usize) -> u32 {
    let v = simd::_mm256_loadu_si256(b.as_ptr().add(p).cast());
    avx2_non_digit_bits(v)
}

/// AVX2: 返回 32 字节寄存器里的“非数字位图”。
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx2_non_digit_bits(v: simd::__m256i) -> u32 {
    let hi = simd::_mm256_cmpgt_epi8(v, simd::_mm256_set1_epi8(b'9' as i8));
    let lo = simd::_mm256_cmpgt_epi8(simd::_mm256_set1_epi8(b'0' as i8), v);
    simd::_mm256_movemask_epi8(simd::_mm256_or_si256(hi, lo)) as u32
}

/// AVX2: 取 256 位寄存器的低 128 位。
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx2_low(v: simd::__m256i) -> simd::__m128i {
    simd::_mm256_castsi256_si128(v)
}

// ---------------- 输入 ----------------

/// 输入: 构造时把 stdin 整个 mmap 进地址空间, 之后全部是内存解析
pub struct FastIn {
    buf: &'static [u8],
    ptr: *const u8, // 仅用于 Drop 时 munmap
    pos: usize,
    avx2: bool,
    sse41: bool,
}

/// 用宏生成不同整数类型的 next_*：核心解析只写一遍。
macro_rules! next_via_i64 {
    ($($name:ident: $ty:ty),* $(,)?) => {
        $(
            #[inline]
            pub fn $name(&mut self) -> $ty {
                self.next_i64() as $ty
            }
        )*
    };
}

impl FastIn {
    pub fn new() -> FastIn {
        // x86_64 struct stat 的 st_size 在偏移 48 (u64)
        let mut st = [0u8; 144];
        unsafe { sc3(SYS_FSTAT, 0, st.as_mut_ptr() as u64, 0) };
        let size = u64::from_le_bytes(st[48..56].try_into().unwrap()) as usize;

        // mmap(addr=0, len, PROT_READ, MAP_PRIVATE, fd=0, off=0); 失败返回 -errno
        let p = unsafe { sc6(SYS_MMAP, 0, size.max(1) as u64, PROT_READ, MAP_PRIVATE, 0, 0) };
        assert!(
            !(-4095..0).contains(&p),
            "mmap stdin failed (stdin 需是普通文件, 本地用 < file 重定向)"
        );
        let ptr = p as *const u8;
        FastIn {
            buf: unsafe { core::slice::from_raw_parts(ptr, size) },
            ptr,
            pos: 0,
            avx2: is_x86_feature_detected!("avx2"),
            sse41: is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("ssse3"),
        }
    }

    /// 连续空白判断 (ASCII 0x00..=0x20)
    #[inline(always)]
    fn is_ws(c: u8) -> bool {
        c <= b' '
    }

    /// 一次跳过一整块空白
    #[inline]
    fn skip_ws(&mut self) {
        let b = self.buf;
        let n = b.len();
        let mut p = self.pos;
        // 最常见的分隔符只有一个字符：直接标量跳过，避免 SIMD 开销
        if p >= n || b[p] > b' ' {
            self.pos = p;
            return;
        }
        if p + 1 >= n || b[p + 1] > b' ' {
            self.pos = p + 1;
            return;
        }
        // 连续多个空白才走 AVX2/SSE 批量路径
        if self.avx2 {
            while p + 32 <= n {
                let bits = unsafe { ws32(b, p) };
                let cnt = bits.trailing_ones() as usize;
                p += cnt;
                if cnt < 32 {
                    self.pos = p;
                    return;
                }
            }
        }
        while p + 16 <= n {
            let bits = unsafe { ws16(b, p) };
            let cnt = bits.trailing_ones() as usize;
            p += cnt;
            if cnt < 16 {
                self.pos = p;
                return;
            }
        }
        while p < n && b[p] <= b' ' {
            p += 1;
        }
        self.pos = p;
    }

    /// 从 start 开始找第一个空白
    #[inline]
    fn scan_ws(&self, start: usize) -> usize {
        let b = self.buf;
        let n = b.len();
        let mut p = start;
        if self.avx2 {
            while p + 32 <= n {
                let bits = unsafe { ws32(b, p) };
                let tz = bits.trailing_zeros() as usize;
                if tz < 32 {
                    return p + tz;
                }
                p += 32;
            }
        }
        while p + 16 <= n {
            let bits = unsafe { ws16(b, p) };
            let tz = bits.trailing_zeros() as usize;
            if tz < 16 {
                return p + tz;
            }
            p += 16;
        }
        while p < n && b[p] > b' ' {
            p += 1;
        }
        p
    }

    /// 从 start 开始连续数字的个数
    #[inline]
    fn digit_len(&self, start: usize) -> usize {
        let b = self.buf;
        let n = b.len();
        let mut p = start;
        let mut len = 0usize;
        if self.avx2 {
            while p + 32 <= n {
                let bits = unsafe { non_digit32(b, p) };
                let bad = bits.trailing_zeros() as usize;
                if bad < 32 {
                    return len + bad;
                }
                len += 32;
                p += 32;
            }
        }
        while p + 16 <= n {
            let bits = unsafe { non_digit16(b, p) };
            let bad = bits.trailing_zeros() as usize;
            if bad < 16 {
                return len + bad;
            }
            len += 16;
            p += 16;
        }
        while p < n && b[p].is_ascii_digit() {
            len += 1;
            p += 1;
        }
        len
    }

    /// 解析 start 开始的 len 个数字
    #[inline]
    fn parse_digits(&self, start: usize, len: usize) -> u64 {
        let b = self.buf;
        let end = start + len;
        let mut val = 0u64;
        let mut p = start;
        if self.sse41 {
            while end - p >= 16 {
                let mut buf = [0u8; 16];
                buf.copy_from_slice(&b[p..p + 16]);
                let part = unsafe { parse16_sse(&buf, 16) };
                val = val.wrapping_mul(POW10[16]).wrapping_add(part);
                p += 16;
            }
            let rem = end - p;
            if rem > 0 {
                let mut buf = [0u8; 16];
                buf[..rem].copy_from_slice(&b[p..end]);
                let part = unsafe { parse16_sse(&buf, rem) };
                val = val.wrapping_mul(POW10[rem]).wrapping_add(part);
            }
        } else {
            while end - p >= 8 {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&b[p..p + 8]);
                let part = parse8_swar(u64::from_le_bytes(bytes));
                val = val.wrapping_mul(100_000_000).wrapping_add(part);
                p += 8;
            }
            while p < end {
                val = val.wrapping_mul(10).wrapping_add((b[p] - b'0') as u64);
                p += 1;
            }
        }
        val
    }

    /// 单遍 SWAR 解析：8 位一组长数字，短数字标量收尾。
    /// 不先数长度再解析，避免对短整数做两次扫描。
    #[inline]
    fn parse_uint_at(&self, mut p: usize) -> (usize, u64) {
        let b = self.buf;
        let n = b.len();
        let start = p;
        let mut val = 0u64;
        while p + 8 <= n {
            let bytes = u64::from_le_bytes(b[p..p + 8].try_into().unwrap());
            if is_8digits(bytes) {
                val = val.wrapping_mul(100_000_000).wrapping_add(parse8_swar(bytes));
                p += 8;
                // 剩余至少 8 位才切到 mm256 冷路径；9~15 位仍走 SWAR，避免向量化开销
                if self.avx2
                    && p + 8 <= n
                    && is_8digits(u64::from_le_bytes(b[p..p + 8].try_into().unwrap()))
                {
                    let (used, v2) = self.parse_uint_avx_rest(p, val);
                    return (p + used - start, v2);
                }
            } else {
                break;
            }
        }
        while p < n && b[p].is_ascii_digit() {
            val = val.wrapping_mul(10).wrapping_add((b[p] - b'0') as u64);
            p += 1;
        }
        (p - start, val)
    }

    /// mm256 冷路径：前 8 位已由 SWAR 解析，这里继续用 AVX2 解析剩余长数字。
    #[inline(never)]
    fn parse_uint_avx_rest(&self, mut p: usize, mut val: u64) -> (usize, u64) {
        let b = self.buf;
        let n = b.len();
        let start = p;
        while p + 32 <= n && b[p + 15].is_ascii_digit() {
            let v = unsafe { simd::_mm256_loadu_si256(b.as_ptr().add(p).cast()) };
            let bits = unsafe { avx2_non_digit_bits(v) };
            if bits & 0xFFFF != 0 {
                break;
            }
            let v16 = unsafe { avx2_low(v) };
            let part = unsafe { parse16_avx2_high(v16, 16) };
            val = val.wrapping_mul(POW10[16]).wrapping_add(part);
            p += 16;
        }
        if p + 16 <= n && b[p + 7].is_ascii_digit() {
            let (len, part) = unsafe { load16_parse_avx(b, p) };
            val = val.wrapping_mul(POW10[len]).wrapping_add(part);
            p += len;
        }
        while p + 8 <= n {
            let bytes = u64::from_le_bytes(b[p..p + 8].try_into().unwrap());
            if is_8digits(bytes) {
                val = val.wrapping_mul(100_000_000).wrapping_add(parse8_swar(bytes));
                p += 8;
            } else {
                break;
            }
        }
        while p < n && b[p].is_ascii_digit() {
            val = val.wrapping_mul(10).wrapping_add((b[p] - b'0') as u64);
            p += 1;
        }
        (p - start, val)
    }

    /// 快速路径：AVX2/SSE 一次定位数字长度并解析。
    /// 短数字不再二次扫描，也不做 16 字节临时拷贝。
    #[inline]
    fn parse_uint_fast(&self, start: usize) -> (usize, u64) {
        let b = self.buf;
        let n = b.len();
        if self.avx2 && start + 32 <= n {
            let (len, val) = unsafe { load32_parse16(b, start) };
            if len <= 16 {
                let exact = len < 16 || start + 16 >= n || !b[start + 16].is_ascii_digit();
                if exact {
                    return (len, val);
                }
            }
        }
        if self.sse41 && start + 16 <= n {
            let (len, val) = unsafe { load16_parse(b, start) };
            let exact = len < 16 || start + 16 >= n || !b[start + 16].is_ascii_digit();
            if exact {
                return (len, val);
            }
        }
        let len = self.digit_len(start);
        let val = self.parse_digits(start, len);
        (len, val)
    }

    /// 下一个 token 的 `[start, end)`, 跳过 ASCII 空白
    #[inline]
    fn token(&mut self) -> (usize, usize) {
        self.skip_ws();
        let s = self.pos;
        let e = self.scan_ws(s);
        self.pos = e;
        (s, e)
    }

    /// 主读取路径：跳空白（单分隔符标量、多空白 AVX2）后解析。
    /// 解析时短数字走 SWAR，长数字走 mm256 向量化合并。
    #[inline]
    fn next_uint_inner(&mut self) -> (bool, u64) {
        self.skip_ws();
        let b = self.buf;
        let n = b.len();
        let mut p = self.pos;
        let neg = p < n && b[p] == b'-';
        if neg {
            p += 1;
        }
        let (len, val) = self.parse_uint_at(p);
        self.pos = p + len;
        (neg, val)
    }

    #[inline]
    pub fn next_i64(&mut self) -> i64 {
        let (neg, v) = self.next_uint_inner();
        if neg {
            (v as i64).wrapping_neg()
        } else {
            v as i64
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let (neg, v) = self.next_uint_inner();
        if neg {
            v.wrapping_neg()
        } else {
            v
        }
    }

    next_via_i64!(next_i32: i32, next_u32: u32, next_usize: usize, next_isize: isize);

    /// 浮点: 支持 `-`、小数点、指数 (CF 精度要求足够)
    pub fn next_f64(&mut self) -> f64 {
        let (s, e) = self.token();
        let d = &self.buf[s..e];
        let mut i = 0usize;
        let neg = i < d.len() && d[i] == b'-';
        if neg {
            i += 1;
        }
        let mut v = 0.0f64;
        while i < d.len() && d[i].is_ascii_digit() {
            v = v * 10.0 + (d[i] - b'0') as f64;
            i += 1;
        }
        if i < d.len() && d[i] == b'.' {
            i += 1;
            let mut frac = 0.1;
            while i < d.len() && d[i].is_ascii_digit() {
                v += (d[i] - b'0') as f64 * frac;
                frac *= 0.1;
                i += 1;
            }
        }
        if i < d.len() && (d[i] == b'e' || d[i] == b'E') {
            i += 1;
            let eneg = i < d.len() && d[i] == b'-';
            if eneg || (i < d.len() && d[i] == b'+') {
                i += 1;
            }
            let mut exp = 0i32;
            while i < d.len() && d[i].is_ascii_digit() {
                exp = exp * 10 + (d[i] - b'0') as i32;
                i += 1;
            }
            v *= 10f64.powi(if eneg { -exp } else { exp });
        }
        if neg { -v } else { v }
    }

    /// 下一个 token 的字节切片 (期间不能再调用其它 `next_*`)
    #[inline]
    pub fn next_bytes(&mut self) -> &[u8] {
        let (s, e) = self.token();
        &self.buf[s..e]
    }
}

impl Drop for FastIn {
    fn drop(&mut self) {
        unsafe { sc3(SYS_MUNMAP, self.ptr as u64, self.buf.len().max(1) as u64, 0) };
    }
}

// ---------------- 输出 ----------------

/// 10000 以内数字的 4 位 ASCII 表 (小端写入，与 oldyan 模板同思路)。
const fn build_out_pre() -> [u32; 10000] {
    let mut a = [0u32; 10000];
    let mut i = 0usize;
    while i < 10000 {
        let v = i as u32;
        a[i] = (b'0' as u32 + (v / 1000) % 10)
            + ((b'0' as u32 + (v / 100) % 10) << 8)
            + ((b'0' as u32 + (v / 10) % 10) << 16)
            + ((b'0' as u32 + v % 10) << 24);
        i += 1;
    }
    a
}
static OUT_PRE: [u32; 10000] = build_out_pre();

/// 输出缓冲: 写进内存, flush 时一次 write 刷出
pub struct FastOut {
    buf: Vec<u8>,
}

impl FastOut {
    pub fn new() -> FastOut {
        FastOut { buf: Vec::with_capacity(1 << 16) }
    }

    #[inline]
    pub fn push(&mut self, c: u8) {
        self.buf.push(c);
    }

    #[inline]
    pub fn write_bytes(&mut self, s: &[u8]) {
        self.buf.extend_from_slice(s);
    }

    #[inline]
    pub fn write_str(&mut self, s: &str) {
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// 4 位查表 itoa
    pub fn write_u64(&mut self, mut v: u64) {
        if v == 0 {
            self.buf.push(b'0');
            return;
        }
        let mut tmp = [0u8; 20];
        let mut len = tmp.len();
        while v >= 10000 {
            let g = OUT_PRE[(v % 10000) as usize].to_le_bytes();
            len -= 4;
            tmp[len..len + 4].copy_from_slice(&g);
            v /= 10000;
        }
        let g = OUT_PRE[v as usize].to_le_bytes();
        if v >= 1000 {
            len -= 4;
            tmp[len..len + 4].copy_from_slice(&g);
        } else if v >= 100 {
            len -= 3;
            tmp[len..len + 3].copy_from_slice(&g[1..]);
        } else if v >= 10 {
            len -= 2;
            tmp[len..len + 2].copy_from_slice(&g[2..]);
        } else {
            len -= 1;
            tmp[len] = g[3];
        }
        self.buf.extend_from_slice(&tmp[len..]);
    }

    /// 4 位查表 itoa (u32)
    pub fn write_u32(&mut self, mut v: u32) {
        if v == 0 {
            self.buf.push(b'0');
            return;
        }
        let mut tmp = [0u8; 10];
        let mut len = tmp.len();
        while v >= 10000 {
            let g = OUT_PRE[(v % 10000) as usize].to_le_bytes();
            len -= 4;
            tmp[len..len + 4].copy_from_slice(&g);
            v /= 10000;
        }
        let g = OUT_PRE[v as usize].to_le_bytes();
        if v >= 1000 {
            len -= 4;
            tmp[len..len + 4].copy_from_slice(&g);
        } else if v >= 100 {
            len -= 3;
            tmp[len..len + 3].copy_from_slice(&g[1..]);
        } else if v >= 10 {
            len -= 2;
            tmp[len..len + 2].copy_from_slice(&g[2..]);
        } else {
            len -= 1;
            tmp[len] = g[3];
        }
        self.buf.extend_from_slice(&tmp[len..]);
    }

    #[inline]
    pub fn write_i32(&mut self, v: i32) {
        if v < 0 {
            self.buf.push(b'-');
            self.write_u32(v.unsigned_abs());
        } else {
            self.write_u32(v as u32);
        }
    }

    #[inline]
    pub fn write_i64(&mut self, v: i64) {
        if v < 0 {
            self.buf.push(b'-');
            self.write_u64(v.unsigned_abs());
        } else {
            self.write_u64(v as u64);
        }
    }

    #[inline]
    pub fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }

    /// 定点小数, `prec` 位小数 (四舍五入)。超大值会饱和, CF 不会出现。
    pub fn write_f64(&mut self, v: f64, prec: usize) {
        if v.is_nan() {
            self.write_str("NaN");
            return;
        }
        if v.is_infinite() {
            if v < 0.0 {
                self.push(b'-');
            }
            self.write_str("inf");
            return;
        }
        if v < 0.0 {
            self.push(b'-');
        }
        let scale = 10f64.powi(prec as i32);
        let x = (v.abs() * scale).round() as u64;
        let scale = 10u64.pow(prec as u32);
        let ip = x / scale;
        let fp = x % scale;
        self.write_u64(ip);
        if prec > 0 {
            self.push(b'.');
            let mut digs = [0u8; 20];
            let mut f = fp;
            let mut n = 0usize;
            while f > 0 {
                digs[n] = (f % 10) as u8 + b'0';
                f /= 10;
                n += 1;
            }
            for _ in n..prec {
                self.push(b'0');
            }
            while n > 0 {
                n -= 1;
                self.push(digs[n]);
            }
        }
    }

    #[inline]
    pub fn ln(&mut self) {
        self.buf.push(b'\n');
    }

    /// 格式化输出 (慢于手写方法, 偶用): `w.fmt(format_args!("{} {}\n", a, b));`
    #[inline]
    pub fn fmt(&mut self, args: core::fmt::Arguments<'_>) {
        let _ = core::fmt::Write::write_fmt(self, args);
    }

    /// 一次 write 刷出
    pub fn flush(&mut self) {
        let mut p = 0usize;
        while p < self.buf.len() {
            let n = unsafe {
                sc3(SYS_WRITE, 1, self.buf.as_ptr().add(p) as u64, (self.buf.len() - p) as u64)
            };
            if n == -4 {
                continue; // EINTR
            }
            if n < 0 {
                panic!("write stdout failed: errno {}", -n);
            }
            p += n as usize;
        }
        self.buf.clear();
    }
}

impl Drop for FastOut {
    fn drop(&mut self) {
        self.flush();
    }
}

impl core::fmt::Write for FastOut {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.write_str(s);
        Ok(())
    }
}


// ================= P3391 解法 =================
// sqrt_bitset：懒标记位图化块状链表（reverse_bench 实测最优）
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

fn main() {
    let mut inp = FastIn::new();
    let mut out = FastOut::new();
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
    for (i, &v) in vals.iter().enumerate() {
        if i > 0 {
            out.write_bytes(b" ");
        }
        out.write_u32(v);
    }
    out.ln();
    // Drop 时自动 flush
}
