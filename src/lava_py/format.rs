use arrow::array::ArrayData;
use pyo3::prelude::*;
use pyo3::{pyfunction, types::PyString, PyAny, types::PyDict};
use arrow::pyarrow::{FromPyArrow, PyArrowException, PyArrowType, ToPyArrow};
use crate::formats::{parquet, MatchResult, ParquetLayout};

#[pyclass]
pub struct ParquetLayoutWrapper {
    #[pyo3(get, set)]
    pub num_row_groups: usize,
    #[pyo3(get, set)]
    pub dictionary_page_sizes: Vec<usize>, // 0 means no dict page
    #[pyo3(get, set)]
    pub data_page_sizes: Vec<usize>,
    #[pyo3(get, set)]
    pub data_page_offsets: Vec<usize>,
    #[pyo3(get, set)]
    pub data_page_num_rows: Vec<usize>,
    #[pyo3(get, set)]
    pub row_group_data_pages: Vec<usize>,
}

impl ParquetLayoutWrapper {
    fn from_parquet_layout(parquet_layout: ParquetLayout) -> Self {
        ParquetLayoutWrapper {
            num_row_groups: parquet_layout.num_row_groups,
            dictionary_page_sizes: parquet_layout.dictionary_page_sizes,
            data_page_sizes: parquet_layout.data_page_sizes,
            data_page_offsets: parquet_layout.data_page_offsets,
            data_page_num_rows: parquet_layout.data_page_num_rows,
            row_group_data_pages: parquet_layout.row_group_data_pages,
        }
    }
}

#[pyclass]
pub struct MatchResultWrapper {
    #[pyo3(get, set)]
    pub file_path: String,
    #[pyo3(get, set)]
    pub column_index: usize,
    #[pyo3(get, set)]
    pub row_group: usize,
    #[pyo3(get, set)]
    pub offset_in_row_group: usize,
    #[pyo3(get, set)]
    pub matched: String,
}

impl From<MatchResult> for MatchResultWrapper {
    fn from(match_result: MatchResult) -> Self {
        MatchResultWrapper {
            file_path: match_result.file_path,
            column_index: match_result.column_index,
            row_group: match_result.row_group,
            offset_in_row_group: match_result.offset_in_row_group,
            matched: match_result.matched,
        }

    }
}

#[pyfunction]
pub fn get_parquet_layout(column_index: usize, file: &str, py: Python) -> PyResult<(PyObject, ParquetLayoutWrapper)> {
    let (arr, parquet_layout) = parquet::get_parquet_layout(column_index, file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())).unwrap();
    Ok((arr.to_pyarrow(py).unwrap(), ParquetLayoutWrapper::from_parquet_layout(parquet_layout)))
}

#[pyfunction]
pub fn search_indexed_pages(
    query: &PyString,
    column_index: usize,
    file_paths: Vec<&PyString>,
    row_groups: Vec<usize>,
    page_offsets: Vec<usize>,
    page_sizes: Vec<usize>,
    dict_page_sizes: Vec<usize>,
) -> PyResult<Vec<MatchResultWrapper>> {

    let match_result = parquet::search_indexed_pages(
        query.to_string(),
        column_index,
        file_paths.iter().map(|x| x.to_string()).collect(),
        row_groups,
        page_offsets.iter().map(|x| *x as u64).collect(),
        page_sizes,
        dict_page_sizes, // 0 means no dict page
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()));
    Ok(match_result
        .unwrap()
        .into_iter()
        .map(|x| x.into())
        .collect())
}
