use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use pyo3::prelude::*;
use pyo3::{pyfunction, types::PyString, PyAny};

use crate::formats::{parquet, ParquetLayout, MatchResult};

#[pyclass]
struct ParquetLayoutWrapper {
    #[pyo3(get, set)]
    internal: ParquetLayout,
}

#[pyclass]
struct MatchResultWrapper {
    #[pyo3(get, set)]
    internal: MatchResult,
}

#[pyfunction]
pub fn get_parquet_layout( column_index: usize, file: &str) -> PyResult<ParquetLayoutWrapper> {
    let parquet_layout = parquet::get_parquet_layout(column_index, file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()));
    Ok(ParquetLayoutWrapper { internal: parquet_layout.unwrap() })
}

#[pyfunction]
pub fn search_indexed_pages(query: &PyString, column_index: usize, file_paths: Vec<&PyString>,
    row_groups: Vec<usize>, page_offsets: Vec<usize>, page_sizes: Vec<usize>, dict_page_sizes: Vec<usize>) -> PyResult<(MatchResultWrapper)> {
    let match_result = parquet::search_indexed_pages(
        query.to_string_lossy(),
        column_index,
        file_paths.iter().map(|x| x.to_string_lossy()).collect(),
        row_groups,
        page_offsets,
        page_sizes,
        dict_page_sizes, // 0 means no dict page
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()));
    Ok(MatchResultWrapper { internal: match_result.unwrap() })
}

