//! AVX-512 kernel. Besides the input-buffer read in `python.rs`, the only `unsafe` code. Gives
//! the same leaves as `leaf_scalar`.
//!
//! Each lane walks one row through the tree, 16 lanes per vector, without branches.
//! A lane whose row reaches a leaf records the leaf and takes the next row (lane refill),
//! so the work follows the average path length instead of the deepest of 16 rows.
//! Four groups of 16 lanes are interleaved to hide gather latency.

use crate::forest::{BLOCK, BLOCK_BITS, CAT, DEFAULT_LEFT, LEAF, MISSING, SET_WORDS_BITS, WIDE};
use std::arch::x86_64::*;

/// Proof that the CPU has AVX-512F and BMI2.
#[derive(Clone, Copy)]
pub(crate) struct Avx512(());

impl Avx512 {
    /// `None` without AVX-512F/BMI2, or if `FASTLEAVES_ISA=scalar`.
    pub fn detect() -> Option<Self> {
        let ok = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("bmi2")
            && std::env::var("FASTLEAVES_ISA").as_deref() != Ok("scalar");
        ok.then_some(Avx512(()))
    }

    /// Sets `leaf_of[r]` to the leaf that row `r` of the binned block `bins` reaches. `tree`
    /// must be a packed tree from `Forest` whose root is not a leaf, `sets` that forest's
    /// category sets, and `bins` must have `BLOCK` bytes for each slot it splits on and 3 bytes
    /// after them.
    pub fn leaves(self, tree: &[u64], sets: &[u32], bins: &[u8], leaf_of: &mut [u32]) {
        assert!(leaf_of.len() <= BLOCK);
        // SAFETY: CPU features are checked by `detect`; rows are below `BLOCK` (above); node
        // indices, slots and set indices are valid by construction of the packed tree, its sets
        // and its bins; a bin (< 256) picks one of the `SET_WORDS` words of a set.
        unsafe { leaves(tree, sets, bins, leaf_of) }
    }
}

/// 16 lanes, each walking one row: the current node's record halves, row, active lanes.
#[derive(Clone, Copy)]
struct Lanes {
    lo: __m512i,
    hi: __m512i,
    row: __m512i,
    active: u16,
}

#[target_feature(enable = "avx512f,bmi2")]
unsafe fn leaves(tree: &[u64], sets: &[u32], bins: &[u8], leaf_of: &mut [u32]) {
    let root = Lanes {
        lo: _mm512_set1_epi32(tree[0] as i32),
        hi: _mm512_set1_epi32((tree[0] >> 32) as i32),
        row: _mm512_setzero_si512(),
        active: 0,
    };
    let n = leaf_of.len() as i32;
    let mut next = 0;
    // four groups, stepped in turn to hide gather latency; separate variables keep them in registers
    let [mut a, mut b, mut c, mut d] = [root; 4];
    unsafe {
        for g in [&mut a, &mut b, &mut c, &mut d] {
            g.refill(&root, 0xFFFF, n, &mut next);
        }
        while (a.active | b.active | c.active | d.active) != 0 {
            a.step(&root, tree, sets, bins, leaf_of, n, &mut next);
            b.step(&root, tree, sets, bins, leaf_of, n, &mut next);
            c.step(&root, tree, sets, bins, leaf_of, n, &mut next);
            d.step(&root, tree, sets, bins, leaf_of, n, &mut next);
        }
    }
}

impl Lanes {
    /// Moves every active lane one level down; lanes that reach a leaf record it and refill.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn step(
        &mut self,
        root: &Lanes,
        tree: &[u64],
        sets: &[u32],
        bins: &[u8],
        leaf_of: &mut [u32],
        n: i32,
        next: &mut i32,
    ) {
        unsafe {
            let act = self.active;
            if act == 0 {
                return;
            }
            let set = |v: u32| _mm512_set1_epi32(v as i32);
            let zero = _mm512_setzero_si512();
            // left iff bin <= threshold (never for MISSING > any threshold bin), or the bin is
            // MISSING and the default is left; categorical: bit `bin` of the set (0 if bin >= 32).
            // `hi` is the threshold bin of a numerical split and the set of a categorical one;
            // WIDE: bit `bin % 32` of word `bin / 32` of set number `hi` in `sets`.
            let slot = _mm512_and_si512(self.lo, set(0xFF));
            let first = _mm512_and_si512(_mm512_srli_epi32::<8>(self.lo), set(0xFFFF));
            let at = _mm512_add_epi32(_mm512_slli_epi32::<BLOCK_BITS>(slot), self.row);
            let d = _mm512_mask_i32gather_epi32::<1>(zero, act, at, bins.as_ptr().cast());
            let d = _mm512_and_si512(d, set(0xFF));
            let cat = _mm512_test_epi32_mask(self.lo, set(CAT));
            let default_left = _mm512_mask_test_epi32_mask(
                _mm512_cmpeq_epi32_mask(d, set(MISSING as u32)),
                self.lo,
                set(DEFAULT_LEFT),
            );
            let num_left = _mm512_cmple_epu32_mask(d, self.hi) | default_left;
            let wide = _mm512_mask_test_epi32_mask(act, self.lo, set(WIDE));
            let (word, shift) = if wide == 0 {
                (self.hi, d)
            } else {
                let at =
                    _mm512_add_epi32(_mm512_slli_epi32::<SET_WORDS_BITS>(self.hi), _mm512_srli_epi32::<5>(d));
                let w = _mm512_mask_i32gather_epi32::<4>(self.hi, wide, at, sets.as_ptr().cast());
                (w, _mm512_mask_and_epi32(d, wide, d, set(31)))
            };
            let cat_left = _mm512_test_epi32_mask(_mm512_srlv_epi32(word, shift), set(1));
            let right = !((num_left & !cat) | (cat_left & cat));
            let child = _mm512_mask_add_epi32(first, right, first, set(1));

            // load the new nodes' records: two 8-lane gathers, split into halves
            let p = tree.as_ptr().cast();
            let r0 = _mm512_mask_i32gather_epi64::<8>(zero, act as u8, _mm512_castsi512_si256(child), p);
            let r1 = _mm512_mask_i32gather_epi64::<8>(
                zero,
                (act >> 8) as u8,
                _mm512_extracti64x4_epi64::<1>(child),
                p,
            );
            let iota2 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            self.lo = _mm512_permutex2var_epi32(r0, iota2, r1);
            self.hi = _mm512_permutex2var_epi32(r0, _mm512_add_epi32(iota2, set(1)), r1);

            let leaf = _mm512_mask_test_epi32_mask(act, self.lo, set(LEAF));
            if leaf != 0 {
                _mm512_mask_i32scatter_epi32::<4>(leaf_of.as_mut_ptr().cast(), leaf, self.row, child);
                self.active &= !leaf;
                self.refill(root, leaf, n, next);
            }
        }
    }

    /// Restarts the lanes in `free` at the root with the next rows, while rows remain.
    #[inline(always)]
    unsafe fn refill(&mut self, root: &Lanes, free: u16, n: i32, next: &mut i32) {
        let k = (free.count_ones() as i32).min(n - *next);
        unsafe {
            let m = _pdep_u32((1u32 << k) - 1, free as u32) as u16;
            let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            self.row =
                _mm512_mask_expand_epi32(self.row, m, _mm512_add_epi32(iota, _mm512_set1_epi32(*next)));
            self.lo = _mm512_mask_mov_epi32(self.lo, m, root.lo);
            self.hi = _mm512_mask_mov_epi32(self.hi, m, root.hi);
            self.active |= m;
        }
        *next += k;
    }
}
