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
    split_index_prefixes: Vec<String>,
    query: String,
    limit: usize,
    reader_type: Option<&PyString>,
) -> Result<(u32, Vec<(usize, u32)>), LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();
    py.allow_threads(|| lava::search_logcloud(split_index_prefixes, query, limit, reader_type.into()))
}

#[pyfunction]
pub fn compress_logs(
    array: &PyAny,
    uid: &PyAny,
    index_name: String,
    group_number: usize,
    timestamp_bytes: usize,
    timestamp_format: String,
) -> Result<(), LavaError> {
    let array = ArrayData::from_pyarrow_bound(&array.as_borrowed())?;
    let uid = ArrayData::from_pyarrow_bound(&uid.as_borrowed())?;
    let _ = lava::compress_logs(array, uid, index_name, group_number, timestamp_bytes, timestamp_format);
    Ok(())
}
