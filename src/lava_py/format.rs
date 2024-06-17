use crate::formats::{parquet, cache, MatchResult, ParquetLayout};
use crate::lava::error::LavaError;
use bytes::Bytes;
use arrow::array::ArrayData;
use arrow::pyarrow::{PyArrowType, ToPyArrow};
use pyo3::prelude::*;
use pyo3::{pyfunction, types::PyString};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use pyo3::types::{PyDict, PyBytes, PyTuple, PyList};
use std::collections::{BTreeMap, HashMap};

#[pyclass]
pub struct ParquetLayoutWrapper {
    #[pyo3(get, set)]
    pub num_row_groups: usize,
    #[pyo3(get, set)]
    pub metadata_bytes: PyObject,
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
    fn from_parquet_layout(py: Python, parquet_layout: ParquetLayout) -> Self {
        ParquetLayoutWrapper {
            num_row_groups: parquet_layout.num_row_groups,
            metadata_bytes: PyBytes::new(py, &parquet_layout.metadata_bytes.slice(..)).into_py(py), 
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
pub fn populate_cache(
    py: Python,
    filenames: Vec<&PyString>,
    ranges: Vec<Vec<(usize, usize)>>,
    reader_type: Option<&PyString>
)  -> Result<(), LavaError> {

    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();
   
    let mut range_dict: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for (i, filename) in filenames.iter().enumerate() {
        range_dict.insert(filename.to_string(), ranges[i].clone());
    }

    py.allow_threads(|| {
        cache::populate_cache(range_dict, reader_type.into())
    })
}

#[pyfunction]
pub fn get_parquet_layout(
    py: Python,
    column_name: &PyString,
    file: &PyString,
    reader_type: Option<&PyString>,
) -> Result<(Vec<PyArrowType<ArrayData>>, ParquetLayoutWrapper), LavaError> {
    let column_name = column_name.to_string();
    let file = file.to_string();
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();
    let (arrs, parquet_layout) =
        py.allow_threads(|| parquet::get_parquet_layout(&column_name, &file, reader_type.into()))?;
    Ok((
        arrs.into_iter().map(|x| PyArrowType(x)).collect(),
        ParquetLayoutWrapper::from_parquet_layout(py, parquet_layout),
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
    reader_type: Option<&PyString>,
    metadata_bytes: Option<&PyDict>,
) -> Result<Vec<PyArrowType<ArrayData>>, LavaError> {
    let column_name = column_name.to_string();
    let file_metadata: Option<HashMap<String, Bytes>>  = match metadata_bytes {
        Some(dict) => {
            let mut metadata_map: HashMap<String, Bytes> = HashMap::new();
            if let Some(dict) = metadata_bytes {
                for (key, value) in dict.iter() {
                    let key_str = key.extract::<&PyString>()?.to_string();
                    let value_bytes = Bytes::copy_from_slice(value.extract::<&PyBytes>()?.as_bytes());
                    metadata_map.insert(key_str, value_bytes);
                }
            }
            Some(metadata_map)
        }
        None => None
    };
        
    let file_paths: Vec<String> = file_paths.iter().map(|x| x.to_string()).collect();
    let page_offsets: Vec<u64> = page_offsets.iter().map(|x| *x as u64).collect();
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();
    let match_result = py.allow_threads(|| {
        parquet::read_indexed_pages(
            column_name,
            file_paths,
            row_groups,
            page_offsets,
            page_sizes,
            dict_page_sizes, // 0 means no dict page
            reader_type.into(),
            file_metadata,
        )
    })?;
    Ok(match_result.into_iter().map(|x| PyArrowType(x)).collect())
}
