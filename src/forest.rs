//! Packed forest, input binning and the scalar kernel.
//!
//! Each input value of a column the model splits on is binned once to a byte, with LightGBM's rules:
//! - numerical: the number of the feature's thresholds below the value, so the split
//!   `x <= T[k]` is exactly `bin <= k`; or `MISSING`, which goes to the split's default side;
//! - categorical: the position of the category among the feature's categories (those in
//!   any of its splits' sets), or `NO_CAT` (goes right) if it is none of them.
//!
//! `MISSING` and `NO_CAT` are the same byte, 255: numerical bins are at most 254 (254
//! thresholds), category positions at most 254, and the kernels test for `MISSING` only on
//! numerical splits. A categorical split never has bit 255 set, so `NO_CAT` goes right.
//!
//! The columns the model splits on get slots 0, 1, ... in column order; columns it never
//! splits on are not read. Bins are stored per block of `BLOCK` rows, slot by slot: the bin of
//! slot `s` and row `r` is at `s * BLOCK + r`, so a tree reads only the slots it splits on.
//!
//! Each node is one `u64`. Low half: slot (bits 0..8), index of the first child (8..24), flags
//! `WIDE`, `CAT`, `LEAF`, `DEFAULT_LEFT`. High half: the threshold bin of a numerical split, the
//! bitset of the category positions that go left, or LightGBM's index of a leaf. Children are
//! adjacent, so the next node is `first + go_right`.
//!
//! A feature with more than 32 categories marks its splits `CAT | WIDE`: their high half is the
//! index of their 256-bit set in `Forest::sets` (8 words). Category positions are at most 254,
//! so bit 255 (`NO_CAT`) is never set and every bin indexes a word within the set.

use crate::Error;
use crate::avx512::Avx512;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::ops::Range;

pub const MAX_FEATURES: usize = 256; // columns used in splits
pub const MAX_THRESHOLDS: usize = 254; // per feature
pub const MAX_CATEGORIES: usize = 255; // per feature
const INLINE_CATEGORIES: usize = 32; // more: the split's set is in `Forest::sets`
pub(crate) const SET_WORDS: usize = 1 << SET_WORDS_BITS; // u32 words per set in `Forest::sets`
pub(crate) const SET_WORDS_BITS: u32 = 3;
pub(crate) const SLOT_MASK: u32 = 0xFF; // node bits 0..8
pub(crate) const FIRST_SHIFT: u32 = 8; // node bits 8..24: the first child
pub(crate) const FIRST_MASK: u32 = 0xFFFF;
pub(crate) const WIDE: u32 = 1 << 27;
pub(crate) const CAT: u32 = 1 << 28;
pub(crate) const LEAF: u32 = 1 << 29;
pub(crate) const DEFAULT_LEFT: u32 = 1 << 30;
pub(crate) const MISSING: u8 = 255; // numerical splits
const NO_CAT: u8 = 255; // categorical splits
pub(crate) const BLOCK: usize = 1 << BLOCK_BITS; // rows per block
pub(crate) const BLOCK_BITS: u32 = 12;
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

/// Frees a tree with a loop instead of recursion, so a chain 32,767 nodes deep fits on the stack:
/// each node's children are moved to `stack` (a leaf left in their place) before it is freed.
impl Drop for Node {
    fn drop(&mut self) {
        let mut stack = vec![];
        let detach = |n: &mut Node, stack: &mut Vec<Box<Node>>| {
            if let Node::Num { left, right, .. } | Node::Cat { left, right, .. } = n {
                for child in [left, right] {
                    stack.push(std::mem::replace(child, Box::new(Node::Leaf { value: 0.0, index: 0 })));
                }
            }
        };
        detach(self, &mut stack);
        while let Some(mut n) = stack.pop() {
            detach(&mut n, &mut stack);
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
    n_features: usize,
    columns: Vec<usize>, // the column of each slot
    slots: Vec<Feature>, // the feature of each slot
    roots: Vec<usize>,
    nodes: Vec<u64>,
    leaves: Vec<f64>, // value of each leaf node, 0 for splits
    sets: Vec<u32>,   // category sets of `WIDE` splits, `SET_WORDS` words each
}

impl Forest {
    pub fn new(n_features: usize, trees: &[Node]) -> Result<Self, Error> {
        let mut features = vec![Feature::Unused; n_features];
        let mut stack: Vec<&Node> = trees.iter().collect();
        while let Some(n) = stack.pop() {
            let Some((f, l, r)) = n.split() else { continue };
            let feature = features.get_mut(f).ok_or_else(|| {
                Error::Invalid(format!("split on feature {f}, but the model has {n_features} features"))
            })?;
            match (&mut *feature, n) {
                (Feature::Unused, Node::Num { threshold, missing, .. }) => {
                    *feature = Feature::Num { thresholds: vec![*threshold], missing: *missing }
                }
                (Feature::Unused, Node::Cat { cats, .. }) => *feature = Feature::Cat(cats.clone()),
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
        let columns: Vec<usize> =
            (0..n_features).filter(|&f| !matches!(features[f], Feature::Unused)).collect();
        if columns.len() > MAX_FEATURES {
            let n = columns.len();
            return Err(Error::unsupported(format!("{n} features used in splits, at most {MAX_FEATURES}")));
        }
        let slots = columns.iter().map(|&f| features[f].clone()).collect();
        let mut forest =
            Forest { n_features, columns, slots, roots: vec![], nodes: vec![], leaves: vec![], sets: vec![] };
        for tree in trees {
            forest.pack(tree)?;
        }
        Ok(forest)
    }

    pub fn n_features(&self) -> usize {
        self.n_features
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
                Node::Num { feature, threshold, default_left, .. } => {
                    let s = self.columns.binary_search(feature).unwrap();
                    let Feature::Num { thresholds, .. } = &self.slots[s] else { unreachable!() };
                    let k = thresholds.partition_point(|t| t < threshold) as u32;
                    (s as u32 | if *default_left { DEFAULT_LEFT } else { 0 }, k)
                }
                Node::Cat { feature, cats, .. } => {
                    let s = self.columns.binary_search(feature).unwrap();
                    let Feature::Cat(all) = &self.slots[s] else { unreachable!() };
                    let positions = cats.iter().map(|c| all.binary_search(c).unwrap());
                    if all.len() <= INLINE_CATEGORIES {
                        (s as u32 | CAT, positions.fold(0, |m, p| m | 1 << p))
                    } else {
                        let at = self.sets.len();
                        if at + SET_WORDS > i32::MAX as usize {
                            return Err(Error::unsupported("too many categorical splits"));
                        }
                        self.sets.resize(at + SET_WORDS, 0);
                        positions.for_each(|p| self.sets[at + p / 32] |= 1 << (p % 32));
                        (s as u32 | CAT | WIDE, (at / SET_WORDS) as u32)
                    }
                }
            };
            if let Some((_, l, r)) = n.split() {
                let first = self.nodes.len() - root;
                if first + 1 > FIRST_MASK as usize {
                    return Err(Error::unsupported("trees with more than 65535 nodes"));
                }
                lo |= (first as u32) << FIRST_SHIFT;
                self.nodes.extend([0, 0]);
                self.leaves.extend([0.0, 0.0]);
                queue.extend([(l, root + first), (r, root + first + 1)]);
            }
            self.nodes[i] = lo as u64 | (hi as u64) << 32;
        }
        Ok(())
    }

    /// The bins of the rows of `x` (at most `BLOCK`), slot by slot; 3 bytes of padding at the
    /// end, as SIMD reads 4 bytes.
    fn bin_block(&self, x: &[f64]) -> Vec<u8> {
        let mut bins = vec![0u8; self.slots.len() * BLOCK + 3];
        for (r, row) in x.chunks(self.n_features.max(1)).enumerate() {
            for (s, (feature, &c)) in self.slots.iter().zip(&self.columns).enumerate() {
                bins[s * BLOCK + r] = bin(feature, row[c]);
            }
        }
        bins
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
        let nf = self.n_features;
        assert_eq!(
            x.len() * width,
            out.len() * nf,
            "x must have one row of {nf} features per {width} outputs"
        );
        let simd = Avx512::detect();
        out.par_chunks_mut(BLOCK * width).zip(x.par_chunks(BLOCK * nf.max(1))).for_each(|(out, x)| {
            let n = out.len() / width;
            let bins = self.bin_block(x);
            let mut leaf_of = vec![0u32; n];
            for (k, &root) in self.roots[trees.clone()].iter().enumerate() {
                let tree = &self.nodes[root..];
                if is_leaf(tree[0]) {
                    leaf_of.fill(0);
                } else if let Some(s) = simd {
                    s.leaves(tree, &self.sets, &bins, &mut leaf_of);
                } else {
                    for (r, l) in leaf_of.iter_mut().enumerate() {
                        *l = leaf_scalar(tree, &self.sets, &bins, r);
                    }
                }
                f(out, k, root, &leaf_of);
            }
        });
    }
}

fn is_leaf(node: u64) -> bool {
    node as u32 & LEAF != 0
}

/// The bin of `v` for `feature`.
fn bin(feature: &Feature, v: f64) -> u8 {
    let v = if v.abs() <= ZERO { 0.0 } else { v };
    match feature {
        Feature::Unused => unreachable!(),
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
    }
}

/// The node (counted from the root of `tree`) of the leaf that row `r` of the binned block reaches;
/// `sets` holds the category sets of `WIDE` splits.
fn leaf_scalar(tree: &[u64], sets: &[u32], bins: &[u8], r: usize) -> u32 {
    let mut n = 0;
    loop {
        let (lo, hi) = (tree[n] as u32, (tree[n] >> 32) as u32);
        if lo & LEAF != 0 {
            return n as u32;
        }
        let d = bins[(lo & SLOT_MASK) as usize * BLOCK + r];
        let left = if lo & WIDE != 0 {
            sets[hi as usize * SET_WORDS + d as usize / 32] >> (d % 32) & 1 == 1
        } else if lo & CAT != 0 {
            hi.checked_shr(d as u32).is_some_and(|m| m & 1 == 1)
        } else if d == MISSING {
            lo & DEFAULT_LEFT != 0
        } else {
            d as u32 <= hi
        };
        n = ((lo >> FIRST_SHIFT) & FIRST_MASK) as usize + !left as usize;
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
            const EDGE: [f64; 26] = [
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
                249.0,
                254.0,
                255.0,
                256.0,
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
                1 => self.next(270) as f64 - 4.0,
                _ => (self.next(4000) as f64 - 2000.0) / 100.0,
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Kind {
        /// Categorical: with all of `CATS` (wide), or only every eighth of them (32, inline).
        Cat {
            wide: bool,
        },
        Num(Missing),
    }

    /// Feature kinds of `n` columns.
    fn kinds(rng: &mut Rng, n: usize) -> Vec<Kind> {
        let all = [
            Kind::Cat { wide: false },
            Kind::Cat { wide: true },
            Kind::Num(Missing::None),
            Kind::Num(Missing::Zero),
            Kind::Num(Missing::NaN),
        ];
        (0..n).map(|_| all[rng.next(5) as usize]).collect()
    }

    /// The categories of a categorical feature: small ones and some far apart.
    const CATS: [u32; MAX_CATEGORIES] = {
        let mut c = [0; MAX_CATEGORIES];
        let big = [255, 256, 1000, 65536, 2147483647];
        let mut i = 0;
        while i < MAX_CATEGORIES {
            c[i] = if i < MAX_CATEGORIES - 5 { i as u32 } else { big[i + 5 - MAX_CATEGORIES] };
            i += 1;
        }
        c
    };

    /// A random tree that splits on the columns `used`.
    fn tree(rng: &mut Rng, depth: u32, kinds: &[Kind], used: &[usize]) -> Node {
        if depth == 0 || rng.next(5) == 0 {
            return Node::Leaf { value: rng.value().clamp(-5.0, 5.0) / 3.0, index: rng.next(1000) as u32 };
        }
        let feature = used[rng.next(used.len() as u64) as usize];
        let (left, right) =
            (Box::new(tree(rng, depth - 1, kinds, used)), Box::new(tree(rng, depth - 1, kinds, used)));
        match kinds[feature] {
            Kind::Cat { wide } => {
                let cats = (0..MAX_CATEGORIES)
                    .filter(|&i| (wide || i % 8 == 6) && rng.next(2) == 0)
                    .map(|i| CATS[i])
                    .collect();
                Node::Cat { feature, cats, left, right }
            }
            Kind::Num(missing) => {
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
        for i in 0..30 {
            // 1 to 320 columns; a third of them unused, the last ones only if they fit the slots
            let nf = [1, 12, 17, 200, 320][i % 5];
            let kinds = kinds(&mut rng, nf);
            let mut used: Vec<usize> = (0..nf).filter(|_| rng.next(3) != 0).take(MAX_FEATURES).collect();
            used.push(nf - 1);
            used.dedup();
            used.truncate(MAX_FEATURES);
            let trees: Vec<Node> = (0..30).map(|_| tree(&mut rng, 12, &kinds, &used)).collect();
            let forest = Forest::new(nf, &trees).unwrap();
            for n in [0, 1, 15, 16, 17, 100, BLOCK + 5] {
                let x: Vec<f64> = (0..n * nf).map(|_| rng.value()).collect();
                let mut out = vec![f64::NAN; n];
                forest.predict_raw(&x, &mut out, 3..trees.len());
                for (row, &got) in x.chunks(nf).zip(&out) {
                    let want = trees[3..].iter().fold(0.0, |s, t| s + eval(t, row).0);
                    assert_eq!(got.to_bits(), want.to_bits());
                }
                let mut leaf = vec![-1; n * 7];
                forest.predict_leaf(&x, &mut leaf, 5..12);
                for (row, got) in x.chunks(nf).zip(leaf.chunks(7)) {
                    let want: Vec<i32> = trees[5..12].iter().map(|t| eval(t, row).1 as i32).collect();
                    assert_eq!(got, want);
                }
                // both kernels pick the same leaves
                let Some(simd) = Avx512::detect() else { continue };
                let rows = n.min(BLOCK);
                let bins = forest.bin_block(&x[..rows * nf]);
                for &root in &forest.roots {
                    let tree = &forest.nodes[root..];
                    if is_leaf(tree[0]) {
                        continue;
                    }
                    let mut got = vec![0; rows];
                    simd.leaves(tree, &forest.sets, &bins, &mut got);
                    let want: Vec<u32> =
                        (0..rows).map(|r| leaf_scalar(tree, &forest.sets, &bins, r)).collect();
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
        assert!(Forest::new(1000, &[num(999, 0.0, Missing::None)]).is_ok()); // unused columns are free
        let wide: Vec<Node> = (0..=MAX_FEATURES).map(|f| num(f, 0.0, Missing::None)).collect();
        assert!(Forest::new(MAX_FEATURES + 1, &wide).is_err());
        assert!(Forest::new(MAX_FEATURES + 1, &wide[1..]).is_ok());
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
