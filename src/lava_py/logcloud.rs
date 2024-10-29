use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use parquet::file::page_index::index;
use pyo3::{pyfunction, types::PyString, PyAny};
use pyo3::{Py, PyResult};
use pyo3::{PyNativeType, Python};

use crate::lava;
use crate::lava::error::LavaError;

#[pyfunction]
pub fn index_logcloud(py: Python, index_name: String, num_groups: usize, wavelet_tree: Option<bool>) -> () {
    py.allow_threads(|| lava::index_logcloud(&index_name, num_groups, wavelet_tree))
}

#[pyfunction]
pub fn index_analysis(py: Python, split_index_prefixes: Vec<String>, reader_type: Option<&PyString>) -> () {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();
    py.allow_threads(|| lava::index_analysis(split_index_prefixes, reader_type.into()))
}

#[pyfunction]
pub fn search_logcloud(
    py: Python,
    split_index_prefixes: Vec<String>,
    query: String,
    limit: usize,
    reader_type: Option<&PyString>,
    wavelet_tree: Option<bool>,
    exact: Option<bool>,
) -> Result<(u32, Vec<(usize, u32)>), LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();
    py.allow_threads(|| {
        lava::search_logcloud(
            split_index_prefixes,
            query,
            limit,
            reader_type.into(),
            wavelet_tree.unwrap_or(false),
            exact.unwrap_or(false),
        )
    })
}

#[pyfunction]
pub fn compress_logs(
    array: &PyAny,
    uid: &PyAny,
    index_name: String,
    group_number: usize,
    timestamp_bytes: Option<usize>,
    timestamp_format: Option<String>,
) -> Result<(), LavaError> {
    let array = ArrayData::from_pyarrow_bound(&array.as_borrowed())?;
    let uid = ArrayData::from_pyarrow_bound(&uid.as_borrowed())?;
    let _ = lava::compress_logs(array, uid, index_name, group_number, timestamp_bytes, timestamp_format);
    Ok(())
}
