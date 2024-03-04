use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use crate::formats::{parquet, MatchResult, ParquetLayout};
use crate::lava::error::LavaError;
use arrow::pyarrow::{ToPyArrow, PyArrowType};
use arrow::array::ArrayData;
use pyo3::prelude::*;
use pyo3::{pyfunction, types::PyString};

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

#[pymethods]
impl MatchResultWrapper {
    fn __repr__(slf: &PyCell<Self>) -> PyResult<String> {
        Ok(format!(
            "MatchResult(file_path={}, column_index={}, row_group={}, offset_in_row_group={}, matched={})",
            slf.borrow().file_path, slf.borrow().column_index, slf.borrow().row_group, slf.borrow().offset_in_row_group, slf.borrow().matched
        ))
    }

    fn __str__(&self) -> String {
        format!(
            "MatchResult(file_path={}, column_index={}, row_group={}, offset_in_row_group={}, matched={})",
            self.file_path, self.column_index, self.row_group, self.offset_in_row_group, self.matched
        )
    }

    fn __hash__(&self) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.file_path.hash(&mut hasher);
        self.column_index.hash(&mut hasher);
        self.row_group.hash(&mut hasher);
        self.offset_in_row_group.hash(&mut hasher);
        self.matched.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }
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
pub fn get_parquet_layout(
    py: Python,
    column_name: &PyString,
    file: &PyString,
) -> Result<(PyObject, ParquetLayoutWrapper), LavaError> {
    let column_name = column_name.to_string();
    let file = file.to_string();
    let (arr, parquet_layout) =
        py.allow_threads(|| parquet::get_parquet_layout(&column_name, &file))?;
    Ok((
        arr.to_pyarrow(py).unwrap(),
        ParquetLayoutWrapper::from_parquet_layout(parquet_layout),
    ))
}

#[pyfunction]
pub fn read_indexed_pages(
    py: Python,
    column_name: &PyString,
    file_paths: Vec<&PyString>,
    row_groups: Vec<usize>,
    page_offsets: Vec<usize>,
    page_sizes: Vec<usize>,
    dict_page_sizes: Vec<usize>,
) -> Result<Vec<PyArrowType<ArrayData>>, LavaError> {
    let column_name = column_name.to_string();
    let file_paths: Vec<String> = file_paths.iter().map(|x| x.to_string()).collect();
    let page_offsets: Vec<u64> = page_offsets.iter().map(|x| *x as u64).collect();
    let match_result = py.allow_threads(|| {
        parquet::read_indexed_pages(
            column_name,
            file_paths,
            row_groups,
            page_offsets,
            page_sizes,
            dict_page_sizes, // 0 means no dict page
        )
    })?;
    Ok(match_result.into_iter().map(|x| PyArrowType(x)).collect())
}
