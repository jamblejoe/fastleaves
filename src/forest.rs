//! Packed forest, input binning and the scalar kernel.
//!
//! Each input value is binned once per row to a byte, with LightGBM's rules:
//! - numerical: the number of the feature's thresholds below the value, so the split
//!   `x <= T[k]` is exactly `bin <= k`; or `MISSING`, which goes to the split's default side;
//! - categorical: the position of the category among the feature's categories (those in
//!   any of its splits' sets), or `NO_CAT` (goes right) if it is none of them.
//!
//! Each node is one `u64`. Low half: feature (bits 0..4), threshold bin (4..12), index of
//! the first child (12..28), flags `CAT`, `LEAF`, `DEFAULT_LEFT`. High half: bitset of the
//! category positions that go left, or LightGBM's index of a leaf. Children are adjacent, so the next node is `first + go_right`.

use crate::Error;
use crate::avx512::Avx512;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::ops::Range;

pub const MAX_FEATURES: usize = 12;
pub const MAX_THRESHOLDS: usize = 253;
pub const MAX_CATEGORIES: usize = 32; // per feature
pub(crate) const ROW: usize = 16; // bytes per binned row
pub(crate) const CAT: u32 = 1 << 28;
pub(crate) const LEAF: u32 = 1 << 29;
pub(crate) const DEFAULT_LEFT: u32 = 1 << 30;
pub(crate) const MISSING: u8 = 254;
const NO_CAT: u8 = 255;
const BLOCK: usize = 4096; // rows per block
const ZERO: f64 = 1e-35f32 as f64; // LightGBM reads inputs within ±ZERO as 0

/// Which numerical values count as missing (LightGBM's `MissingType`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Missing {
    /// None: NaN is read as 0.
    None,
    /// NaN and 0.
    Zero,
    /// NaN.
    NaN,
}

/// A decision tree, as given to [`Forest::new`].
pub enum Node {
    /// A leaf's value and its index in LightGBM's model (for `pred_leaf`).
    Leaf { value: f64, index: u32 },
    /// Goes left iff `x <= threshold`, or iff `default_left` if `x` is missing.
    Num {
        feature: usize,
        threshold: f64,
        missing: Missing,
        default_left: bool,
        left: Box<Node>,
        right: Box<Node>,
    },
    /// Goes left iff `x`, truncated to an integer, is one of `cats`.
    Cat { feature: usize, cats: Vec<u32>, left: Box<Node>, right: Box<Node> },
}

impl Node {
    fn split(&self) -> Option<(usize, &Node, &Node)> {
        match self {
            Node::Leaf { .. } => None,
            Node::Num { feature, left, right, .. } | Node::Cat { feature, left, right, .. } => {
                Some((*feature, left, right))
            }
        }
    }
}

#[derive(Clone)]
enum Feature {
    Unused,
    Num { thresholds: Vec<f64>, missing: Missing }, // thresholds sorted, unique
    Cat(Vec<u32>),                                  // categories in any split's set, sorted, unique
}

pub struct Forest {
    features: Vec<Feature>,
    roots: Vec<usize>,
    nodes: Vec<u64>,
    leaves: Vec<f64>, // value of each leaf node, 0 for splits
}

impl Forest {
    pub fn new(n_features: usize, trees: &[Node]) -> Result<Self, Error> {
        if n_features > MAX_FEATURES {
            return Err(Error::unsupported(format!("{n_features} features, at most {MAX_FEATURES}")));
        }
        let mut features = vec![Feature::Unused; n_features];
        let mut stack: Vec<&Node> = trees.iter().collect();
        while let Some(n) = stack.pop() {
            let Some((f, l, r)) = n.split() else { continue };
            let feature = features.get_mut(f).ok_or_else(|| {
                Error::Invalid(format!("split on feature {f}, but the model has {n_features} features"))
            })?;
            match (&mut *feature, n) {
                (Feature::Unused, Node::Num { missing, .. }) => {
                    *feature = Feature::Num { thresholds: vec![], missing: *missing }
                }
                (Feature::Unused, Node::Cat { .. }) => *feature = Feature::Cat(vec![]),
                _ => {}
            }
            match (&mut *feature, n) {
                (Feature::Num { thresholds, missing: m }, Node::Num { threshold, missing, .. })
                    if m == missing =>
                {
                    thresholds.push(*threshold)
                }
                (Feature::Cat(all), Node::Cat { cats, .. }) => all.extend(cats),
                _ => {
                    return Err(Error::unsupported(format!(
                        "feature {f} has splits of different kinds or missing-value types"
                    )));
                }
            }
            stack.extend([l, r]);
        }
        for (f, feature) in features.iter_mut().enumerate() {
            match feature {
                Feature::Num { thresholds: t, .. } => {
                    t.sort_by(f64::total_cmp);
                    t.dedup();
                    if t.len() > MAX_THRESHOLDS {
                        let n = t.len();
                        return Err(Error::unsupported(format!(
                            "feature {f} has {n} thresholds, at most {MAX_THRESHOLDS}"
                        )));
                    }
                }
                Feature::Cat(c) => {
                    c.sort_unstable();
                    c.dedup();
                    if c.len() > MAX_CATEGORIES {
                        let n = c.len();
                        return Err(Error::unsupported(format!(
                            "feature {f} has {n} categories in its splits, at most {MAX_CATEGORIES}"
                        )));
                    }
                }
                Feature::Unused => {}
            }
        }

        let mut forest = Forest { features, roots: vec![], nodes: vec![], leaves: vec![] };
        for tree in trees {
            forest.pack(tree)?;
        }
        Ok(forest)
    }

    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    pub fn n_trees(&self) -> usize {
        self.roots.len()
    }

    /// Appends `tree` in breadth-first order, so that siblings are adjacent.
    fn pack(&mut self, tree: &Node) -> Result<(), Error> {
        let root = self.nodes.len();
        self.roots.push(root);
        self.nodes.push(0);
        self.leaves.push(0.0);
        let mut queue = VecDeque::from([(tree, root)]);
        while let Some((n, i)) = queue.pop_front() {
            let (mut lo, hi) = match n {
                Node::Leaf { value, index } => {
                    self.leaves[i] = *value;
                    (LEAF, *index)
                }
                Node::Num { feature: f, threshold, default_left, .. } => {
                    let Feature::Num { thresholds, .. } = &self.features[*f] else { unreachable!() };
                    let k = thresholds.partition_point(|t| t < threshold) as u32;
                    (*f as u32 | k << 4 | if *default_left { DEFAULT_LEFT } else { 0 }, 0)
                }
                Node::Cat { feature: f, cats, .. } => {
                    let Feature::Cat(all) = &self.features[*f] else { unreachable!() };
                    let set = cats.iter().fold(0u32, |m, c| m | 1 << all.binary_search(c).unwrap());
                    (*f as u32 | CAT, set)
                }
            };
            if let Some((_, l, r)) = n.split() {
                let first = self.nodes.len() - root;
                if first + 1 > 0xFFFF {
                    return Err(Error::unsupported("trees with more than 65535 nodes".into()));
                }
                lo |= (first as u32) << 12;
                self.nodes.extend([0, 0]);
                self.leaves.extend([0.0, 0.0]);
                queue.extend([(l, root + first), (r, root + first + 1)]);
            }
            self.nodes[i] = lo as u64 | (hi as u64) << 32;
        }
        Ok(())
    }

    fn bin_row(&self, x: &[f64], out: &mut [u8]) {
        for (feature, (&v, o)) in self.features.iter().zip(x.iter().zip(out)) {
            let v = if v.abs() <= ZERO { 0.0 } else { v };
            *o = match feature {
                Feature::Unused => 0,
                // as LightGBM: truncate to int; NaN, negative and out of int range go right
                Feature::Cat(cats) if v > -1.0 && v < 2147483648.0 => {
                    cats.binary_search(&(v as u32)).map_or(NO_CAT, |k| k as u8)
                }
                Feature::Cat(_) => NO_CAT,
                Feature::Num { thresholds, missing } => {
                    let is_missing = match missing {
                        Missing::None => false,
                        Missing::Zero => v.is_nan() || v == 0.0,
                        Missing::NaN => v.is_nan(),
                    };
                    let v = if v.is_nan() { 0.0 } else { v };
                    if is_missing { MISSING } else { thresholds.partition_point(|&t| t < v) as u8 }
                }
            };
        }
    }

    /// Raw score of each row of the row-major matrix `x`: the sum of the leaf values of
    /// `trees`, added in tree order as LightGBM does. Runs on rayon's current thread pool.
    pub fn predict_raw(&self, x: &[f64], out: &mut [f64], trees: Range<usize>) {
        out.fill(0.0);
        self.for_each_tree(x, out, 1, trees, |acc, _, root, leaf_of| {
            let leaves = &self.leaves[root..];
            acc.iter_mut().zip(leaf_of).for_each(|(a, &l)| *a += leaves[l as usize]);
        });
    }

    /// LightGBM's index of the leaf each row of `x` reaches in each of `trees`: row-major,
    /// `trees.len()` per row. Runs on rayon's current thread pool.
    pub fn predict_leaf(&self, x: &[f64], out: &mut [i32], trees: Range<usize>) {
        let width = trees.len();
        if width == 0 {
            return;
        }
        self.for_each_tree(x, out, width, trees, |block, k, root, leaf_of| {
            for (row, &l) in block.chunks_mut(width).zip(leaf_of) {
                row[k] = (self.nodes[root + l as usize] >> 32) as i32;
            }
        });
    }

    /// For each block of rows of `x` and each of `trees` in order: `f(out, k, root, leaf_of)`
    /// with the block's part of `out` (`width` per row), the tree's position `k` in `trees`,
    /// its root and the node (counted from the root) of the leaf each row reaches.
    fn for_each_tree<T: Send>(
        &self,
        x: &[f64],
        out: &mut [T],
        width: usize,
        trees: Range<usize>,
        f: impl Fn(&mut [T], usize, usize, &[u32]) + Sync,
    ) {
        let nf = self.n_features();
        assert_eq!(
            x.len() * width,
            out.len() * nf,
            "x must have one row of {nf} features per {width} outputs"
        );
        let simd = Avx512::detect();
        out.par_chunks_mut(BLOCK * width).zip(x.par_chunks(BLOCK * nf.max(1))).for_each(|(out, x)| {
            let n = out.len() / width;
            let mut bins = vec![0u8; (n + 1) * ROW]; // +1 row: SIMD reads 4 bytes
            for (row, b) in x.chunks(nf).zip(bins.chunks_mut(ROW)) {
                self.bin_row(row, b);
            }
            let mut leaf_of = vec![0u32; n];
            for (k, &root) in self.roots[trees.clone()].iter().enumerate() {
                let tree = &self.nodes[root..];
                if tree[0] as u32 & LEAF != 0 {
                    leaf_of.fill(0);
                } else if let Some(s) = simd {
                    s.leaves(tree, &bins, &mut leaf_of);
                } else {
                    for (l, row) in leaf_of.iter_mut().zip(bins.chunks(ROW)) {
                        *l = leaf_scalar(tree, row);
                    }
                }
                f(out, k, root, &leaf_of);
            }
        });
    }
}

fn leaf_scalar(tree: &[u64], row: &[u8]) -> u32 {
    let mut n = 0;
    loop {
        let lo = tree[n] as u32;
        if lo & LEAF != 0 {
            return n as u32;
        }
        let d = row[(lo & 0xF) as usize];
        let left = if lo & CAT != 0 {
            ((tree[n] >> 32) as u32).checked_shr(d as u32).is_some_and(|m| m & 1 == 1)
        } else if d == MISSING {
            lo & DEFAULT_LEFT != 0
        } else {
            d as u32 <= (lo >> 4) & 0xFF
        };
        n = ((lo >> 12) & 0xFFFF) as usize + !left as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
        fn value(&mut self) -> f64 {
            const EDGE: [f64; 22] = [
                0.0,
                -0.0,
                ZERO,
                -ZERO,
                1e-36,
                -1e-36,
                -0.5,
                -0.99,
                -1.0,
                31.0,
                31.9,
                32.0,
                33.0,
                99.5,
                65536.0,
                2147483647.0,
                2147483648.0,
                4294967296.0,
                1e10,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ];
            match self.next(4) {
                0 => EDGE[self.next(EDGE.len() as u64) as usize],
                1 => self.next(110) as f64 - 4.0,
                _ => (self.next(4000) as f64 - 2000.0) / 100.0,
            }
        }
    }

    /// Feature kinds: categorical, or numerical with one missing-value type.
    fn kinds(rng: &mut Rng) -> Vec<Option<Missing>> {
        let all = [None, Some(Missing::None), Some(Missing::Zero), Some(Missing::NaN)];
        (0..MAX_FEATURES).map(|_| all[rng.next(4) as usize]).collect()
    }

    /// The categories of a categorical feature: small ones and some far apart.
    const CATS: [u32; MAX_CATEGORIES] = {
        let mut c = [0; MAX_CATEGORIES];
        let big = [32, 33, 99, 65536, 2147483647];
        let mut i = 0;
        while i < MAX_CATEGORIES {
            c[i] = if i < MAX_CATEGORIES - 5 { i as u32 } else { big[i + 5 - MAX_CATEGORIES] };
            i += 1;
        }
        c
    };

    fn tree(rng: &mut Rng, depth: u32, kinds: &[Option<Missing>]) -> Node {
        if depth == 0 || rng.next(5) == 0 {
            return Node::Leaf { value: rng.value().clamp(-5.0, 5.0) / 3.0, index: rng.next(1000) as u32 };
        }
        let feature = rng.next(MAX_FEATURES as u64) as usize;
        let (left, right) = (Box::new(tree(rng, depth - 1, kinds)), Box::new(tree(rng, depth - 1, kinds)));
        match kinds[feature] {
            None => {
                let cats = CATS.into_iter().filter(|_| rng.next(2) == 0).collect();
                Node::Cat { feature, cats, left, right }
            }
            Some(missing) => {
                let threshold = match rng.next(10) {
                    0 => ZERO,
                    1 => -ZERO,
                    _ => (rng.next(MAX_THRESHOLDS as u64 - 2) as f64 - 120.0) / 6.0,
                };
                Node::Num { feature, threshold, missing, default_left: rng.next(2) == 0, left, right }
            }
        }
    }

    /// LightGBM's `Tree::Decision` on the raw value, after its predictor's rounding to zero.
    /// Returns the leaf's value and index.
    fn eval(n: &Node, x: &[f64]) -> (f64, u32) {
        let value = |f: usize| if x[f].abs() <= ZERO { 0.0 } else { x[f] };
        match n {
            Node::Leaf { value, index } => (*value, *index),
            Node::Num { feature, threshold, missing, default_left, left, right } => {
                let mut v = value(*feature);
                if v.is_nan() && *missing != Missing::NaN {
                    v = 0.0;
                }
                let go_left =
                    if (*missing == Missing::Zero && v == 0.0) || (*missing == Missing::NaN && v.is_nan()) {
                        *default_left
                    } else {
                        v <= *threshold
                    };
                eval(if go_left { left } else { right }, x)
            }
            Node::Cat { feature, cats, left, right } => {
                let v = value(*feature);
                // static_cast<int> on x86: i32::MIN for NaN and out of range
                let c = if v > -2147483649.0 && v < 2147483648.0 { v as i32 } else { i32::MIN };
                eval(if c >= 0 && cats.contains(&(c as u32)) { left } else { right }, x)
            }
        }
    }

    #[test]
    fn matches_reference() {
        // CI sets this under the AVX-512 emulator so the SIMD comparison below cannot be skipped
        if std::env::var_os("FASTLEAVES_REQUIRE_AVX512").is_some() {
            assert!(Avx512::detect().is_some(), "FASTLEAVES_REQUIRE_AVX512 is set but AVX-512 is not used");
        }
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..30 {
            let kinds = kinds(&mut rng);
            let trees: Vec<Node> = (0..30).map(|_| tree(&mut rng, 12, &kinds)).collect();
            let forest = Forest::new(MAX_FEATURES, &trees).unwrap();
            for n in [0, 1, 15, 16, 17, 100, BLOCK + 5] {
                let x: Vec<f64> = (0..n * MAX_FEATURES).map(|_| rng.value()).collect();
                let mut out = vec![f64::NAN; n];
                forest.predict_raw(&x, &mut out, 3..trees.len());
                for (row, &got) in x.chunks(MAX_FEATURES).zip(&out) {
                    let want = trees[3..].iter().fold(0.0, |s, t| s + eval(t, row).0);
                    assert_eq!(got.to_bits(), want.to_bits());
                }
                let mut leaf = vec![-1; n * 7];
                forest.predict_leaf(&x, &mut leaf, 5..12);
                for (row, got) in x.chunks(MAX_FEATURES).zip(leaf.chunks(7)) {
                    let want: Vec<i32> = trees[5..12].iter().map(|t| eval(t, row).1 as i32).collect();
                    assert_eq!(got, want);
                }
                // both kernels pick the same leaves
                let Some(simd) = Avx512::detect() else { continue };
                let mut bins = vec![0u8; (n + 1) * ROW];
                for (row, b) in x.chunks(MAX_FEATURES).zip(bins.chunks_mut(ROW)) {
                    forest.bin_row(row, b);
                }
                for &root in &forest.roots {
                    let tree = &forest.nodes[root..];
                    if tree[0] as u32 & LEAF != 0 {
                        continue;
                    }
                    let mut got = vec![0; n];
                    simd.leaves(tree, &bins, &mut got);
                    let want: Vec<u32> = bins.chunks(ROW).take(n).map(|r| leaf_scalar(tree, r)).collect();
                    assert_eq!(got, want);
                }
            }
        }
    }

    #[test]
    fn rejects_unsupported_models() {
        let leaf = || Box::new(Node::Leaf { value: 1.0, index: 0 });
        let num = |feature, threshold, missing| Node::Num {
            feature,
            threshold,
            missing,
            default_left: false,
            left: leaf(),
            right: leaf(),
        };
        let cat = |feature, cats| Node::Cat { feature, cats, left: leaf(), right: leaf() };
        assert!(Forest::new(MAX_FEATURES + 1, &[]).is_err());
        assert!(Forest::new(2, &[num(2, 0.0, Missing::None)]).is_err());
        assert!(Forest::new(2, &[num(0, 0.0, Missing::None), cat(0, vec![1])]).is_err());
        assert!(Forest::new(2, &[num(0, 0.0, Missing::None), num(0, 1.0, Missing::NaN)]).is_err());
        let cats: Vec<u32> = (0..=MAX_CATEGORIES as u32).map(|c| c * 1000).collect();
        assert!(Forest::new(2, &[cat(1, cats[1..].to_vec())]).is_ok());
        assert!(Forest::new(2, &[cat(1, cats[1..].to_vec()), cat(1, vec![0])]).is_err());
        let many: Vec<Node> = (0..=MAX_THRESHOLDS).map(|i| num(0, i as f64, Missing::None)).collect();
        assert!(Forest::new(1, &many).is_err());
        assert!(Forest::new(1, &many[1..]).is_ok());
    }
}
