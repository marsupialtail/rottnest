use pyo3::prelude::*;

mod format;
mod lava;

#[pymodule]
fn rottnest(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(lava::build_lava_bm25, m)?)?;
    m.add_function(wrap_pyfunction!(lava::build_lava_substring, m)?)?;
    m.add_function(wrap_pyfunction!(lava::search_lava, m)?)?;
    m.add_function(wrap_pyfunction!(lava::merge_lava, m)?)?;
    m.add_function(wrap_pyfunction!(format::get_parquet_layout, m)?)?;
    m.add_function(wrap_pyfunction!(format::search_indexed_pages, m)?)?;
    Ok(())
}
