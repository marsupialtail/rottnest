use pyo3::prelude::*;

mod lava;

/// This module is a python module implemented in Rust.
#[pymodule]
fn rottnest_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lava::build_lava_natural_language, m)?)?;
    m.add_function(wrap_pyfunction!(lava::search_lava, m)?)?;
    m.add_function(wrap_pyfunction!(lava::merge_lava, m)?)?;
    Ok(())
}
