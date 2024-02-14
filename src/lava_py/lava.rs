use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use pyo3::{pyfunction, types::PyString, PyAny};

use crate::lava;
use crate::lava::error::LavaError;

#[pyfunction]
pub fn search_lava(file: &str, query: &str) -> Result<Vec<u64>, LavaError> {
    lava::search_lava(file, query)
}

#[pyfunction]
pub fn merge_lava(
    condensed_lava_file: &PyString,
    lava_files: Vec<&PyString>,
    uid_offsets: Vec<u64>
) -> Result<(), LavaError> {
    println!("{:?}", uid_offsets);
    lava::merge_lava(
        condensed_lava_file.to_string_lossy(),
        lava_files.iter().map(|x| x.to_string_lossy()).collect(),
        uid_offsets
    )
}

#[pyfunction]
pub fn build_lava_natural_language(
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    language: Option<&PyAny>,
) -> Result<(), LavaError> {
    let array = ArrayData::from_pyarrow(array)?;
    let uid = ArrayData::from_pyarrow(uid)?;
    let language = language.map(|x| ArrayData::from_pyarrow(x)).transpose()?;

    lava::build_lava_natural_language(output_file_name.to_string_lossy(), array, uid, language)
}
