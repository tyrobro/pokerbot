use pyo3::prelude::*;

#[pyfunction]
fn ping() -> PyResult<i64> {
    Ok(core_engine::ping())
}

#[pymodule]
fn pokerbot_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    Ok(())
}
