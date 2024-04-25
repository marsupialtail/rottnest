use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use pyo3::Python;
use pyo3::{pyfunction, types::PyString, PyAny};

use crate::lava;
use crate::lava::error::LavaError;
use ndarray::{Array2, ArrayD, Ix2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::types::PyBytes;
use pyo3::IntoPy;
use pyo3::Py;

#[pyfunction]
pub fn search_lava_bm25(
    py: Python,
    files: Vec<String>,
    query_tokens: Vec<u32>,
    query_weights: Vec<f32>,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    py.allow_threads(|| lava::search_lava_bm25(files, query_tokens, query_weights, k))
}

#[pyfunction]
pub fn search_lava_substring(
    py: Python,
    files: Vec<String>,
    query: String,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    py.allow_threads(|| lava::search_lava_substring(files, query, k))
}

#[pyfunction]
pub fn search_lava_vector(
    py: Python,
    files: Vec<String>,
    column_name: &str,
    uid_nrows: Vec<Vec<usize>>,
    uid_to_metadatas: Vec<Vec<(String, usize, usize, usize, usize)>>,
    query: Vec<f32>,
    k: usize,
) -> Result<(Vec<(usize, usize)>, Py<PyArray2<f32>>), LavaError> {
    let (metadata, array) = py.allow_threads(|| {
        lava::search_lava_vector(files, column_name, &uid_nrows, &uid_to_metadatas, &query, k)
    })?;

    Ok((metadata, array.into_pyarray(py).to_owned()))
}

#[pyfunction]
pub fn get_tokenizer_vocab(py: Python, files: Vec<String>) -> Result<Vec<String>, LavaError> {
    py.allow_threads(|| lava::get_tokenizer_vocab(files))
}

#[pyfunction]
pub fn merge_lava_bm25(
    py: Python,
    condensed_lava_file: String,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> {
    py.allow_threads(|| {
        lava::parallel_merge_files(condensed_lava_file, lava_files, uid_offsets, 2, 0)
    })
}

#[pyfunction]
pub fn merge_lava_substring(
    py: Python,
    condensed_lava_file: String,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> {
    py.allow_threads(|| {
        lava::parallel_merge_files(condensed_lava_file, lava_files, uid_offsets, 2, 1)
    })
}

#[pyfunction]
pub fn merge_lava_vector(
    py: Python,
    condensed_lava_file: String,
    lava_files: Vec<String>,
    vectors: Vec<PyReadonlyArrayDyn<f32>>,
) -> Result<(), LavaError> {
    let vectors = vectors
        .iter()
        .map(|x| {
            let array = x.as_array();
            let owned_array: Array2<f32> = array.into_dimensionality::<Ix2>().unwrap().to_owned();
            owned_array
        })
        .collect();

    py.allow_threads(|| lava::parallel_merge_vector_files(condensed_lava_file, lava_files, vectors))
}

#[pyfunction]
pub fn build_lava_bm25(
    py: Python,
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    tokenizer_file: Option<&PyString>,
) -> Result<(), LavaError> {
    let output_file_name = output_file_name.to_string();
    let array = ArrayData::from_pyarrow(array)?;
    let uid = ArrayData::from_pyarrow(uid)?;
    let tokenizer_file = tokenizer_file.map(|x| x.to_string());

    py.allow_threads(|| {
        lava::build_lava_bm25(
            output_file_name,
            array,
            uid,
            tokenizer_file,
            Some(1.2),
            Some(0.75),
        )
    })
}

#[pyfunction]
pub fn build_lava_kmer(
    py: Python,
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    tokenizer_file: Option<&PyString>,
) -> Result<(), LavaError> {
    let output_file_name = output_file_name.to_string();
    let array = ArrayData::from_pyarrow(array)?;
    let uid = ArrayData::from_pyarrow(uid)?;
    let tokenizer_file = tokenizer_file.map(|x| x.to_string());

    py.allow_threads(|| lava::build_lava_kmer(output_file_name, array, uid, tokenizer_file))
}

#[pyfunction]
pub fn build_lava_substring(
    py: Python,
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    tokenizer_file: Option<&PyString>,
    token_skip_factor: Option<u32>,
) -> Result<(), LavaError> {
    let output_file_name = output_file_name.to_string();
    let array = ArrayData::from_pyarrow(array)?;
    let uid = ArrayData::from_pyarrow(uid)?;
    let tokenizer_file = tokenizer_file.map(|x| x.to_string());

    py.allow_threads(|| {
        lava::build_lava_substring(
            output_file_name,
            array,
            uid,
            tokenizer_file,
            token_skip_factor,
        )
    })
}

#[pyfunction]
pub fn build_lava_vector(
    py: Python,
    output_file_name: &PyString,
    array: PyReadonlyArrayDyn<f32>,
    uid: &PyAny,
) -> Result<(), LavaError> {
    let output_file_name = output_file_name.to_string();
    let array = array.as_array();
    let owned_array: Array2<f32> = array.into_dimensionality::<Ix2>().unwrap().to_owned();
    let uid = ArrayData::from_pyarrow(uid)?;

    py.allow_threads(|| lava::build_lava_vector(output_file_name, owned_array, uid))
}
