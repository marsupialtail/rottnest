use arrow::array::ArrayData;
use arrow::pyarrow::FromPyArrow;
use pyo3::prelude::*;
use pyo3::{pyfunction, types::PyString, PyAny};

use crate::lava;

#[pyfunction]
pub fn search_lava(file: &str, query: &str) -> PyResult<Vec<u64>> {
    lava::search_lava(file, query)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
pub fn merge_lava(condensed_lava_file: &PyString, lava_files: Vec<&PyString>) -> PyResult<()> {
    lava::merge_lava(
        condensed_lava_file.to_string_lossy(),
        lava_files.iter().map(|x| x.to_string_lossy()).collect(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
pub fn build_lava_natural_language(
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    language: Option<&PyAny>,
) -> PyResult<()> {
    let array = ArrayData::from_pyarrow(array)?;
    let uid = ArrayData::from_pyarrow(uid)?;
    let language = language.map(|x| ArrayData::from_pyarrow(x)).transpose()?;

    lava::build_lava_natural_language(output_file_name.to_string_lossy(), array, uid, language)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
