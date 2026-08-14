#!/usr/bin/env bash
# 检查区间反转各实现的 SIMD 指令：标量循环是否被自动向量化，AVX2 手写版用了什么。
# Usage: CXX=g++ RUSTC=rustc bash asm_check.sh
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
RUSTC=${RUSTC:-rustc}

cat > /tmp/rev_asm.cpp <<'EOF'
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>
using u32 = std::uint32_t;
using usize = std::size_t;
__attribute__((noinline)) void rev_scalar(u32* a, usize l, usize r) {
    while (l < r) { --r; u32 t = a[l]; a[l] = a[r]; a[r] = t; ++l; }
}
__attribute__((noinline)) void rev_std(u32* a, usize l, usize r) {
    std::reverse(a + l, a + r);
}
__attribute__((noinline)) void rev_avx2(u32* a, usize l, usize r) {
    const __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    while (l + 8 <= r) {
        __m256i L = _mm256_loadu_si256((const __m256i*)(a + l));
        __m256i R = _mm256_loadu_si256((const __m256i*)(a + r - 8));
        _mm256_storeu_si256((__m256i*)(a + l), _mm256_permutevar8x32_epi32(R, idx));
        _mm256_storeu_si256((__m256i*)(a + r - 8), _mm256_permutevar8x32_epi32(L, idx));
        l += 8; r -= 8;
    }
    while (l < r) { --r; u32 t = a[l]; a[l] = a[r]; a[r] = t; ++l; }
}
EOF

cat > /tmp/rev_asm.rs <<'EOF'
#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
use std::arch::x86_64::*;
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe fn rev_scalar(a: &mut [u32], mut l: usize, mut r: usize) {
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
#[unsafe(no_mangle)]
#[inline(never)]
pub fn rev_std(a: &mut [u32], l: usize, r: usize) {
    a[l..r].reverse();
}
#[cfg(target_feature = "avx2")]
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe fn rev_avx2(a: &mut [u32], mut l: usize, mut r: usize) {
    let idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    while l + 8 <= r {
        let base = a.as_mut_ptr();
        let lv = _mm256_loadu_si256(base.add(l) as *const __m256i);
        let rv = _mm256_loadu_si256(base.add(r - 8) as *const __m256i);
        _mm256_storeu_si256(base.add(l) as *mut __m256i, _mm256_permutevar8x32_epi32(rv, idx));
        _mm256_storeu_si256(base.add(r - 8) as *mut __m256i, _mm256_permutevar8x32_epi32(lv, idx));
        l += 8;
        r -= 8;
    }
    rev_scalar(a, l, r);
}
fn main() {}
EOF

"$CXX" -O3 -march=native -std=c++23 -S /tmp/rev_asm.cpp -o /tmp/rev_asm_cpp.s
"$RUSTC" --edition=2024 -O -C target-cpu=native --emit=asm /tmp/rev_asm.rs -o /tmp/rev_asm_rs.s

printf '%-12s %-30s %-8s %-8s\n' impl instr gcc# rust#
for impl in rev_scalar rev_std rev_avx2; do
    for pat in 'vpermd|vpermps' 'vpshufb' 'vpshufd' 'vpermq|vperm2i128'; do
        case "$pat" in
            'vpermd|vpermps') lab='vpermd | vpermps（32b 反转 shuffle）' ;;
            'vpshufb') lab='vpshufb（字节 shuffle）' ;;
            'vpshufd') lab='vpshufd（128b shuffle）' ;;
            *) lab='vpermq | vperm2i128' ;;
        esac
        c=$(grep -cE "$pat" /tmp/rev_asm_cpp.s || true)
        r=$(grep -cE "$pat" /tmp/rev_asm_rs.s || true)
        printf '%-12s %-30s %-8s %-8s\n' "$impl" "$lab" "$c" "$r"
    done
done

printf '\n说明：标量循环如果出现 vpermd/vpshuf 系指令，说明编译器自动向量化了；\n'
printf 'avx2 版本 vpermd 计数应明显 > 0（每 8 元素一对 load/store + 两次 vpermd）。\n'
