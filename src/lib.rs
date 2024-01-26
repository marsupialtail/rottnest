use pyo3::prelude::*;
use tantivy::tokenizer::*;
use tantivy::schema::*;

/// Function that tokenizes the input text and returns a list of tokens.
#[pyfunction]
fn tokenize(text: String) -> PyResult<Vec<String>> {
    let tokenizer = SimpleTokenizer;
    let mut token_stream = tokenizer.token_stream(&text);
    let mut tokens = Vec::new();

    while let Some(token) = token_stream.next() {
        println!("{}", token.text.to_string());
        tokens.push(token.text.to_string());
    }

    Ok(tokens)
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn rottnest_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}
