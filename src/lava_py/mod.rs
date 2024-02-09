use pyo3::prelude::*;

mod lava;
mod format;

#[pymodule]
fn rottnest_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lava::build_lava_natural_language, m)?)?;
    m.add_function(wrap_pyfunction!(lava::search_lava, m)?)?;
    m.add_function(wrap_pyfunction!(lava::merge_lava, m)?)?;
    m.add_function(wrap_pyfunction!(format::get_parquet_layout, m)?)?;
    m.add_function(wrap_pyfunction!(format::search_indexed_pages, m)?)?;
    Ok(())
}
