use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use arrow::array::{make_array, Array, ArrayData, ArrayRef, Int64Array, StringArray};
use arrow::compute::kernels;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowException, PyArrowType, ToPyArrow};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::{RecordBatchIterator, RecordBatchReader};

use tantivy::schema::*;
use tantivy::tokenizer::Tokenizer;
// use tantivy_tokenizer_api::Tokenizer as TantivyApiTokenizerTrait;
use tantivy_jieba::JiebaTokenizer;
use tantivy::tokenizer::SimpleTokenizer;
use std::collections::HashMap;

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

enum TokenizerEnum {
    Simple(SimpleTokenizer),
    Jieba(JiebaTokenizer),
}

impl TokenizerEnum {
    fn token_stream(&mut self, text: &str) -> Box<dyn tantivy::tokenizer::TokenStream> {
        match self {
            TokenizerEnum::Simple(tokenizer) => {
                Box::new(tokenizer.token_stream(text)) 
            },
            TokenizerEnum::Jieba(tokenizer) => {
                Box::new(tokenizer.token_stream(text)) 
            },
        }
    }
}

/// Function that tokenizes the input text and returns a list of tokens.
#[pyfunction]
fn tokenize(array: &PyAny, py: Python) -> PyResult<Vec<Vec<String>>> {
    let array = make_array(ArrayData::from_pyarrow(array)?);

    let array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::ParseError("Expects string array".to_string()))
        .map_err(to_py_err)?;

    let mut tokenizers: HashMap<String, TokenizerEnum> = HashMap::new();
    tokenizers.insert("cn".to_string(), TokenizerEnum::Jieba(JiebaTokenizer {}));
    tokenizers.insert("en".to_string(), TokenizerEnum::Simple(SimpleTokenizer {}));
    
    for i in 0..array.len() {
        let text = array.value(i);
        if let Some(tokenizer) = tokenizers.get_mut("cn") {
            let mut token_stream = tokenizer.token_stream(text);
            let mut this_tokens = Vec::new();
            while let Some(token) = token_stream.next() {
                this_tokens.push(token.text.to_string());
            }
            tokens.push(this_tokens);
        } else {
            // Handle error: tokenizer not found
        }
    }
    
    Ok(tokens)
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn rottnest_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}
