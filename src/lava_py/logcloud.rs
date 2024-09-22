use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use parquet::file::page_index::index;
use pyo3::{pyfunction, types::PyString, PyAny};
use pyo3::{Py, PyResult};
use pyo3::{PyNativeType, Python};

use crate::lava;
use crate::lava::error::LavaError;

#[pyfunction]
pub fn index_logcloud(py: Python, index_name: String, num_groups: usize) -> () {
    py.allow_threads(|| lava::index_logcloud(&index_name, num_groups))
}

#[pyfunction]
pub fn search_logcloud(
    py: Python,
    split_index_prefix: String,
    query: String,
    limit: usize,
    reader_type: Option<&PyString>,
) -> Result<Vec<usize>, LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();
    py.allow_threads(|| lava::search_logcloud(split_index_prefix, query, limit, reader_type.into()))
}
