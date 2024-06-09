use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use pyo3::Python;
use pyo3::{pyfunction, types::PyString, PyAny};

use crate::lava;
use crate::lava::error::LavaError;
use ndarray::{Array1, Array2, Ix2};
use numpy::{IntoPyArray, PyArray2, PyArray1, PyReadonlyArrayDyn};
use pyo3::Py;


#[pyfunction]
pub fn search_lava_bm25(
    py: Python,
    files: Vec<String>,
    query_tokens: Vec<u32>,
    query_weights: Vec<f32>,
    k: usize,
    reader_type: Option<&PyString>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    py.allow_threads(|| {
        lava::search_lava_bm25(files, query_tokens, query_weights, k, reader_type.into())
    })
}

#[pyfunction]
pub fn search_lava_substring(
    py: Python,
    files: Vec<String>,
    query: String,
    k: usize,
    reader_type: Option<&PyString>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    py.allow_threads(|| lava::search_lava_substring(files, query, k, reader_type.into()))
}

#[pyfunction]
pub fn search_lava_uuid(
    py: Python,
    files: Vec<String>,
    query: String,
    k: usize,
    reader_type: Option<&PyString>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    py.allow_threads(|| lava::search_lava_uuid(files, query, k, reader_type.into()))
}

#[pyfunction]
pub fn search_lava_vector(
    py: Python,
    files: Vec<String>,
    query: Vec<f32>,
    nprobes: usize,
    reader_type: Option<&PyString>,
) -> Result<(Vec<usize>, Vec<Py<PyArray1<u8>>>, Vec<(usize, Py<PyArray1<u8>>)>), LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    let result: (Vec<usize>, Vec<Array1<u8>>, Vec<(usize, Array1<u8>)>) = py.allow_threads(|| {
        lava::search_lava_vector(
            files,
            &query,
            nprobes,
            reader_type.into(),
        )
    })?;

    let x = result.1
        .into_iter()
        .map(|x| x.into_pyarray(py).to_owned())
        .collect();

    let y = result.2
        .into_iter()
        .map(|(x, y)| (x, y.into_pyarray(py).to_owned()))
        .collect();
    

    Ok((result.0, x, y))
}


#[pyfunction]
pub fn get_tokenizer_vocab(
    py: Python,
    files: Vec<String>,
    reader_type: Option<&PyString>,
) -> Result<Vec<String>, LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    py.allow_threads(|| lava::get_tokenizer_vocab(files, reader_type.into()))
}

#[pyfunction]
pub fn merge_lava_bm25(
    py: Python,
    condensed_lava_file: String,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
    reader_type: Option<&PyString>,
) -> Result<(), LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    py.allow_threads(|| {
        lava::parallel_merge_files(
            condensed_lava_file,
            lava_files,
            uid_offsets,
            2,
            0,
            reader_type.into(),
        )
    })
}

#[pyfunction]
pub fn merge_lava_substring(
    py: Python,
    condensed_lava_file: String,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
    reader_type: Option<&PyString>,
) -> Result<(), LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    py.allow_threads(|| {
        lava::parallel_merge_files(
            condensed_lava_file,
            lava_files,
            uid_offsets,
            2,
            1,
            reader_type.into(),
        )
    })
}

#[pyfunction]
pub fn merge_lava_uuid(
    py: Python,
    condensed_lava_file: String,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
    reader_type: Option<&PyString>,
) -> Result<(), LavaError> {
    let reader_type = reader_type.map(|x| x.to_string()).unwrap_or_default();

    py.allow_threads(|| {
        lava::parallel_merge_files(
            condensed_lava_file,
            lava_files,
            uid_offsets,
            2,
            2,
            reader_type.into(),
        )
    })
}

#[pyfunction]
pub fn build_lava_bm25(
    py: Python,
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    tokenizer_file: Option<&PyString>,
) -> Result<Vec<(usize, usize)>, LavaError> {
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
pub fn build_lava_uuid(
    py: Python,
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
) -> Result<Vec<(usize, usize)>, LavaError> {
    let output_file_name = output_file_name.to_string();
    let array = ArrayData::from_pyarrow(array)?;
    let uid = ArrayData::from_pyarrow(uid)?;

    py.allow_threads(|| {
        lava::build_lava_uuid(
            output_file_name,
            array,
            uid,
        )
    })
}

// #[pyfunction]
// pub fn build_lava_kmer(
//     py: Python,
//     output_file_name: &PyString,
//     array: &PyAny,
//     uid: &PyAny,
//     tokenizer_file: Option<&PyString>,
// ) -> Result<(), LavaError> {
//     let output_file_name = output_file_name.to_string();
//     let array = ArrayData::from_pyarrow(array)?;
//     let uid = ArrayData::from_pyarrow(uid)?;
//     let tokenizer_file = tokenizer_file.map(|x| x.to_string());

//     py.allow_threads(|| lava::build_lava_kmer(output_file_name, array, uid, tokenizer_file))
// }

#[pyfunction]
pub fn build_lava_substring(
    py: Python,
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    tokenizer_file: Option<&PyString>,
    token_skip_factor: Option<u32>,
) -> Result<Vec<(usize, usize)>, LavaError> {
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