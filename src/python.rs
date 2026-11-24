//! Python module `fastleaves._core`; the user-facing API is in `python/fastleaves`.

use crate::Error;
use crate::forest::BLOCK;
use pyo3::buffer::{Element, PyBuffer};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::ops::Range;
use std::sync::{Arc, Mutex};

create_exception!(
    fastleaves,
    UnsupportedError,
    PyValueError,
    "A model, input or option fastleaves does not support."
);

fn py_err(e: Error) -> PyErr {
    match e {
        Error::Unsupported(m) => UnsupportedError::new_err(m),
        Error::Invalid(m) => PyValueError::new_err(m),
    }
}

#[pyclass(frozen)]
struct Model(crate::Model);

#[pymethods]
impl Model {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        crate::Model::parse(text).map(Model).map_err(py_err)
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.0.forest.n_features()
    }

    #[getter]
    fn num_trees(&self) -> usize {
        self.0.forest.n_trees()
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.0.feature_names.clone()
    }

    /// Writes into `out` (float64, one per row) the predictions for the C-contiguous float64
    /// matrix `x` from trees `start..end`, on `num_threads` threads (0: all cores). Releases the GIL.
    #[allow(clippy::too_many_arguments)] // mirrors the Python call
    fn predict(
        &self,
        py: Python,
        x: PyBuffer<f64>,
        out: PyBuffer<f64>,
        start: usize,
        end: usize,
        raw_score: bool,
        num_threads: usize,
    ) -> PyResult<()> {
        let (x, rows) = self.rows(&x, start..end)?;
        let res = run(py, num_threads, rows, rows, |res| self.0.predict(x, res, start..end, raw_score))?;
        write(py, &out, &res)
    }

    /// Writes into `out` (int32, rows × `end - start`) LightGBM's leaf index for each row of `x`
    /// and each of trees `start..end`.
    fn predict_leaf(
        &self,
        py: Python,
        x: PyBuffer<f64>,
        out: PyBuffer<i32>,
        start: usize,
        end: usize,
        num_threads: usize,
    ) -> PyResult<()> {
        let (x, rows) = self.rows(&x, start..end)?;
        let forest = &self.0.forest;
        let res = run(py, num_threads, rows, rows * (end - start), |res| {
            forest.predict_leaf(x, res, start..end);
            Ok(())
        })?;
        write(py, &out, &res)
    }
}

impl Model {
    /// The values of `x` and its number of rows, after checking its shape and the tree range.
    fn rows<'a>(&self, x: &'a PyBuffer<f64>, trees: Range<usize>) -> PyResult<(&'a [f64], usize)> {
        let n = self.0.forest.n_features();
        let &[rows, cols] = x.shape() else {
            return Err(PyValueError::new_err("Input numpy.ndarray or list must be 2 dimensional"));
        };
        if cols != n {
            let msg =
                format!("The number of features in data ({cols}) is not the same as in the model ({n})");
            return Err(PyValueError::new_err(msg));
        }
        if trees.start > trees.end || trees.end > self.0.forest.n_trees() {
            return Err(PyValueError::new_err(format!("no trees {trees:?}")));
        }
        let ptr = x.as_slice_ptr().ok_or_else(|| PyValueError::new_err("x must be C-contiguous"))?;
        // SAFETY: `PyBuffer::<f64>::get` checked the format, item size and alignment, and
        // `as_slice_ptr` checked that the buffer is C-contiguous, so `ptr` covers exactly
        // rows × cols valid f64 values. The buffer keeps the array alive, and the array cannot be
        // resized while exported, until `x` is dropped; the slice borrows `x`, so it cannot outlive it.
        // Python code in another thread may still write to the array while the GIL is released.
        // fastleaves reads it only in `Forest::bin_block`, which maps every f64 to a valid bin, so such
        // a write can only change the predictions of the rows it touches, as in LightGBM.
        Ok((unsafe { ptr.as_ref() }, rows))
    }
}

/// Copies `res` into the C-contiguous, writable buffer `out` of the same length.
fn write<T: Element + Copy>(py: Python, out: &PyBuffer<T>, res: &[T]) -> PyResult<()> {
    if !out.is_c_contiguous() {
        return Err(PyValueError::new_err("out must be C-contiguous"));
    }
    out.copy_from_slice(py, res)
}

/// `len` outputs for `rows` rows, computed by `f` on `num_threads` threads (0: all cores)
/// without the GIL. Rows that fit in one block are computed on the calling thread.
fn run<T: Default + Clone + Send>(
    py: Python,
    num_threads: usize,
    rows: usize,
    len: usize,
    f: impl FnOnce(&mut [T]) -> Result<(), Error> + Send,
) -> PyResult<Vec<T>> {
    py.detach(|| {
        let mut out = vec![T::default(); len];
        if rows <= BLOCK {
            f(&mut out).map_err(py_err)?;
        } else {
            pool(num_threads)?.install(|| f(&mut out)).map_err(py_err)?;
        }
        Ok(out)
    })
}

/// The thread pool with `num_threads` threads (0: all cores), built on first use and kept, as
/// starting threads on every call costs more than predicting a few thousand rows.
fn pool(num_threads: usize) -> PyResult<Arc<ThreadPool>> {
    static POOLS: Mutex<Vec<(usize, Arc<ThreadPool>)>> = Mutex::new(Vec::new());
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p)) = pools.iter().find(|(n, _)| *n == num_threads) {
        return Ok(p.clone());
    }
    let p = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    pools.push((num_threads, Arc::new(p)));
    Ok(pools[pools.len() - 1].1.clone())
}

/// The SIMD path predictions use: "avx512" or "scalar".
#[pyfunction]
fn isa() -> &'static str {
    crate::isa()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(isa, m)?)?;
    m.add("UnsupportedError", m.py().get_type::<UnsupportedError>())
}
