//! Reads LightGBM's text model format (`version=v4`), as written by `save_model`.
//!
//! The file is blocks separated by blank lines: a header of `key=value` lines, one block
//! per tree (`Tree=0`, `Tree=1`, ...), then `end of trees` and parts that prediction ignores.

use crate::{Error, Forest, Missing, Node};
use std::collections::HashMap;
use std::ops::Range;
use std::str::FromStr;

/// How a raw score becomes a prediction (LightGBM's `ConvertOutput`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Output {
    Identity,
    Exp,
}

pub struct Model {
    pub forest: Forest,
    pub feature_names: Vec<String>,
    /// The output transformation, or why it is not supported (then only raw scores are).
    pub output: Result<Output, Error>,
}

impl Model {
    pub fn parse(text: &str) -> Result<Model, Error> {
        let mut blocks = text.split("\n\n").map(str::trim).filter(|b| !b.is_empty());
        let header = key_values(blocks.next().unwrap_or(""));
        let get = |key| header.get(key).copied().ok_or_else(|| invalid(format!("model without {key}")));

        let version = get("version")?;
        if version != "v4" {
            return Err(Error::unsupported(format!("model version {version}, only v4 (LightGBM 4.x)")));
        }
        if get("num_class")? != "1" || get("num_tree_per_iteration")? != "1" {
            return Err(Error::unsupported("multiclass models"));
        }
        if header.contains_key("average_output") {
            return Err(Error::unsupported("random forest mode (average_output)"));
        }
        let n_features = number::<usize>(get("max_feature_idx")?)? + 1;
        let feature_names = get("feature_names")?.split(' ').map(String::from).collect();
        let objective = header.get("objective").copied().unwrap_or("none");

        let mut trees = vec![];
        for block in blocks.take_while(|b| *b != "end of trees") {
            if block.starts_with("Tree=") {
                trees.push(parse_tree(block)?);
            }
        }
        Ok(Model { forest: Forest::new(n_features, &trees)?, feature_names, output: output(objective) })
    }

    /// Predictions for the row-major matrix `x` from `trees`, or raw scores if `raw_score`.
    /// Fails before any work if the output transformation is not supported and `raw_score` is false.
    pub fn predict(
        &self,
        x: &[f64],
        out: &mut [f64],
        trees: Range<usize>,
        raw_score: bool,
    ) -> Result<(), Error> {
        let output = if raw_score { Output::Identity } else { self.output.clone()? };
        self.forest.predict_raw(x, out, trees);
        if output == Output::Exp {
            out.iter_mut().for_each(|v| *v = v.exp());
        }
        Ok(())
    }
}

fn output(objective: &str) -> Result<Output, Error> {
    let mut words = objective.split(' ');
    match words.next().unwrap_or("") {
        "regression" if words.any(|w| w == "sqrt") => Err(Error::unsupported("output of sqrt regression")),
        "regression" | "regression_l1" | "huber" | "fair" | "quantile" | "mape" => Ok(Output::Identity),
        "poisson" | "gamma" | "tweedie" => Ok(Output::Exp),
        name => Err(Error::unsupported(format!("output of objective {name}"))),
    }
}

fn parse_tree(block: &str) -> Result<Node, Error> {
    let kv = key_values(block);
    let get = |key| kv.get(key).copied().ok_or_else(|| invalid(format!("tree without {key}")));
    if kv.get("is_linear").is_some_and(|v| *v != "0") {
        return Err(Error::unsupported("linear trees"));
    }
    let num_leaves: usize = number(get("num_leaves")?)?;
    if num_leaves == 0 {
        return Err(invalid("tree without leaves"));
    }
    let leaf_value: Vec<f64> = numbers(get("leaf_value")?, num_leaves)?;
    if num_leaves == 1 {
        return Ok(Node::Leaf { value: leaf_value[0], index: 0 });
    }
    if num_leaves > 32768 {
        return Err(Error::unsupported("trees with more than 32768 leaves"));
    }
    let n = num_leaves - 1;
    let (cat_boundaries, cat_threshold) = match kv.get("cat_boundaries") {
        Some(b) => (numbers::<usize>(b, 0)?, numbers::<u32>(get("cat_threshold")?, 0)?),
        None => (vec![], vec![]),
    };
    let t = Tree {
        split_feature: numbers(get("split_feature")?, n)?,
        threshold: numbers(get("threshold")?, n)?,
        decision_type: numbers(get("decision_type")?, n)?,
        left_child: numbers(get("left_child")?, n)?,
        right_child: numbers(get("right_child")?, n)?,
        leaf_value,
        cat_boundaries,
        cat_threshold,
    };
    t.root()
}

struct Tree {
    split_feature: Vec<usize>,
    threshold: Vec<f64>,
    decision_type: Vec<u8>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
    cat_boundaries: Vec<usize>,
    cat_threshold: Vec<u32>,
}

impl Tree {
    /// The tree from its root. Children are leaves (`~leaf`) or later nodes, so building the
    /// nodes from last to first builds each node's children before it. A loop, not recursion:
    /// a tree may be a chain 32,767 nodes deep.
    fn root(&self) -> Result<Node, Error> {
        let mut built: Vec<Option<Node>> = self.left_child.iter().map(|_| None).collect();
        for i in (0..built.len()).rev() {
            let mut child = |c: i32| {
                let node = if c < 0 {
                    self.leaf_value.get(!c as usize).map(|&value| Node::Leaf { value, index: !c as u32 })
                } else if c as usize > i {
                    built.get_mut(c as usize).and_then(Option::take) // None if it has a parent already
                } else {
                    None
                };
                node.map(Box::new).ok_or_else(|| invalid(format!("node {i} has child {c}")))
            };
            let (left, right) = (child(self.left_child[i])?, child(self.right_child[i])?);
            built[i] = Some(self.split(i, left, right)?);
        }
        Ok(built[0].take().unwrap())
    }

    /// Split `u` with its children.
    fn split(&self, u: usize, left: Box<Node>, right: Box<Node>) -> Result<Node, Error> {
        let (feature, d) = (self.split_feature[u], self.decision_type[u]);
        if d & 1 == 0 {
            let missing = match (d >> 2) & 3 {
                0 => Missing::None,
                1 => Missing::Zero,
                2 => Missing::NaN,
                m => return Err(invalid(format!("missing type {m}"))),
            };
            let (threshold, default_left) = (self.threshold[u], d & 2 != 0);
            return Ok(Node::Num { feature, threshold, missing, default_left, left, right });
        }
        // categorical: the threshold indexes a bitset of the categories that go left
        let c = self.threshold[u] as usize;
        let words = self
            .cat_boundaries
            .get(c..c + 2)
            .and_then(|b| self.cat_threshold.get(b[0]..b[1]))
            .ok_or_else(|| invalid(format!("no category set {c}")))?;
        let cats =
            (0..32 * words.len() as u32).filter(|&k| words[k as usize / 32] >> (k % 32) & 1 == 1).collect();
        Ok(Node::Cat { feature, cats, left, right })
    }
}

fn invalid(what: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("invalid model: {what}"))
}

fn key_values(block: &str) -> HashMap<&str, &str> {
    block.lines().map(|line| line.split_once('=').unwrap_or((line, ""))).collect()
}

fn number<T: FromStr>(s: &str) -> Result<T, Error> {
    s.parse().map_err(|_| invalid(format!("cannot read number {s:?}")))
}

/// Space-separated numbers; exactly `n` of them unless `n` is 0.
fn numbers<T: FromStr>(s: &str, n: usize) -> Result<Vec<T>, Error> {
    let v: Vec<T> = s.split(' ').filter(|w| !w.is_empty()).map(number).collect::<Result<_, _>>()?;
    if n != 0 && v.len() != n {
        return Err(invalid(format!("{} numbers where {n} are expected", v.len())));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=1
objective=poisson
feature_names=a b
feature_infos=none none
tree_sizes=300
";

    /// `x0 <= 0.5 ? (x1 is category 1 ? 1.0 : 2.0) : 3.0`; NaN in x0 counts as 0.
    const TREE: &str = "Tree=0
num_leaves=3
num_cat=1
split_feature=0 1
threshold=0.5 0
decision_type=2 1
left_child=1 -1
right_child=-3 -2
leaf_value=1 2 3
cat_boundaries=0 1
cat_threshold=2
is_linear=0
shrinkage=1";

    /// A model file with the given tree blocks, as LightGBM lays it out.
    fn model(trees: &[&str]) -> String {
        format!("{HEADER}\n{}\n\nend of trees\n\nfeature_importances:\na=1\n", trees.join("\n\n"))
    }

    /// `text` with the line `key=...` replaced by `key=value`, or removed if `value` is None.
    fn set(text: &str, key: &str, value: Option<&str>) -> String {
        let line = |l: &str| match l.split_once('=') {
            Some((k, _)) if k == key => value.map(|v| format!("{key}={v}")),
            _ => Some(l.to_string()),
        };
        text.lines().filter_map(line).collect::<Vec<_>>().join("\n") + "\n"
    }

    fn predict(m: &Model, x: &[f64], raw_score: bool) -> Vec<f64> {
        let mut out = vec![0.0; x.len() / 2];
        m.predict(x, &mut out, 0..m.forest.n_trees(), raw_score).unwrap();
        out
    }

    #[test]
    fn reads_a_model_and_predicts() {
        let m = Model::parse(&model(&[TREE])).unwrap();
        assert_eq!((m.forest.n_features(), m.forest.n_trees()), (2, 1));
        assert_eq!(m.feature_names, ["a", "b"]);
        assert_eq!(m.output.clone(), Ok(Output::Exp));
        let x = [0.2, 1.0, 0.2, 2.0, 0.7, 1.0, f64::NAN, 1.9];
        assert_eq!(predict(&m, &x, true), [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(predict(&m, &x, false), [1f64.exp(), 2f64.exp(), 3f64.exp(), 1f64.exp()]);
        let mut leaf = vec![-1; 4];
        m.forest.predict_leaf(&x, &mut leaf, 0..1);
        assert_eq!(leaf, [0, 1, 2, 0]); // LightGBM's leaf numbers, not the packed order
    }

    #[test]
    fn reads_single_leaf_trees_and_no_trees() {
        let single = "Tree=0\nnum_leaves=1\nnum_cat=0\nsplit_feature=\nsplit_gain=\nthreshold=\n\
                      decision_type=\nleft_child=\nright_child=\nleaf_value=0.25\nis_linear=0\nshrinkage=1";
        let m = Model::parse(&model(&[single, TREE])).unwrap();
        assert_eq!(predict(&m, &[0.7, 0.0], true), [3.25]);
        let mut leaf = vec![-1; 2];
        m.forest.predict_leaf(&[0.7, 0.0], &mut leaf, 0..2);
        assert_eq!(leaf, [0, 2]);
        let empty = Model::parse(&format!("{HEADER}\nend of trees\n")).unwrap();
        assert_eq!(predict(&empty, &[0.7, 0.0], true), [0.0]);
    }

    #[test]
    fn decodes_category_sets_over_several_words() {
        let tree = set(&set(TREE, "cat_boundaries", Some("0 3")), "cat_threshold", Some("2 0 1"));
        let Node::Num { left, .. } = &parse_tree(&tree).unwrap() else { panic!() };
        let Node::Cat { cats, .. } = &**left else { panic!() };
        assert_eq!(cats, &[1, 64]);
    }

    #[test]
    fn output_transformations() {
        let cases = [
            ("regression", Some(Output::Identity)),
            ("regression_l1", Some(Output::Identity)),
            ("quantile", Some(Output::Identity)),
            ("poisson", Some(Output::Exp)),
            ("tweedie tweedie_variance_power:1.3", Some(Output::Exp)),
            ("regression sqrt", None),
            ("binary sigmoid:1", None),
            ("lambdarank", None),
            ("none", None),
        ];
        for (objective, expected) in cases {
            assert_eq!(output(objective).ok(), expected, "{objective}");
        }
    }

    #[test]
    fn refuses_unsupported_models() {
        let cases = [
            ("version 3", set(&model(&[TREE]), "version", Some("v3"))),
            ("multiclass", set(&model(&[TREE]), "num_class", Some("3"))),
            ("random forest", model(&[TREE]).replace("tree_sizes", "average_output\ntree_sizes")),
            ("linear tree", model(&[&set(TREE, "is_linear", Some("1"))])),
            (
                "256 categories on one feature",
                model(&[&set(TREE, "cat_threshold", Some(&["4294967295"; 8].join(" ")))
                    .replace("cat_boundaries=0 1", "cat_boundaries=0 8")]),
            ),
        ];
        for (what, text) in cases {
            assert!(matches!(Model::parse(&text), Err(Error::Unsupported(_))), "{what}");
        }
    }

    #[test]
    fn rejects_malformed_models_without_panicking() {
        let tree = |key, value| model(&[&set(TREE, key, value)]);
        let cases = [
            ("no max_feature_idx", set(&model(&[TREE]), "max_feature_idx", None)),
            ("no leaf_value", tree("leaf_value", None)),
            ("too few leaf values", tree("leaf_value", Some("1 2"))),
            ("no leaves", tree("num_leaves", Some("0"))),
            ("not a number", tree("threshold", Some("abc 0"))),
            ("child loops back", tree("left_child", Some("0 -1"))),
            ("child out of range", tree("left_child", Some("5 -1"))),
            ("leaf out of range", tree("right_child", Some("-9 -2"))),
            ("both children are node 1", tree("right_child", Some("1 -2"))),
            ("unknown missing type", tree("decision_type", Some("14 1"))),
            ("category set out of range", tree("threshold", Some("0.5 4"))),
            ("feature out of range", tree("split_feature", Some("0 7"))),
        ];
        for (what, text) in cases {
            assert!(matches!(Model::parse(&text), Err(Error::Invalid(_))), "{what}");
        }
    }
}
