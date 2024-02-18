use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use pyo3::Python;
use pyo3::{pyfunction, types::PyString, PyAny};

use crate::lava;
use crate::lava::error::LavaError;

#[pyfunction]
pub fn search_lava(py: Python, file: String, query: String) -> Result<Vec<u64>, LavaError> {
    py.allow_threads(|| lava::search_lava(file, query))
}

#[pyfunction]
pub fn merge_lava(
    py: Python,
    condensed_lava_file: &PyString,
    lava_files: Vec<&PyString>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> {
    let condensed_lava_file = condensed_lava_file.to_string();
    let lava_files = lava_files.iter().map(|x| x.to_string()).collect();
    py.allow_threads(|| lava::merge_lava(condensed_lava_file, lava_files, uid_offsets))
}

#[pyfunction]
pub fn build_lava_natural_language(
    py: Python,
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    language: Option<&PyAny>,
) -> Result<(), LavaError> {
    let output_file_name = output_file_name.to_string();
    let array = ArrayData::from_pyarrow(array)?;
    let uid = ArrayData::from_pyarrow(uid)?;
    let language = language.map(|x| ArrayData::from_pyarrow(x)).transpose()?;

    py.allow_threads(|| lava::build_lava_natural_language(output_file_name, array, uid, language))
}
