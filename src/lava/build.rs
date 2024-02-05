use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use arrow::array::{make_array, Array, ArrayData, StringArray, UInt64Array};
use arrow::error::ArrowError;

use arrow::pyarrow::{FromPyArrow, PyArrowException};

use tantivy::tokenizer::*;
use tantivy_jieba::JiebaTokenizer;

use std::collections::BTreeMap;
use std::collections::HashMap;

use bincode;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use zstd::stream::encode_all;

use whatlang::{detect, Lang};

use lazy_static::lazy_static;

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

#[derive(Clone)]
enum TokenizerEnum {
    Simple(SimpleTokenizer),
    Jieba(JiebaTokenizer),
    English(TextAnalyzer),
}

lazy_static! {
    static ref DEFAULT_TOKENIZER: TokenizerEnum = TokenizerEnum::Simple(SimpleTokenizer::default());
    static ref TOKENIZERS: HashMap<Lang, TokenizerEnum> = {
        let mut tokenizers = HashMap::new();
        tokenizers.insert(Lang::Cmn, TokenizerEnum::Jieba(JiebaTokenizer {}));

        let en_stem = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(Stemmer::new(Language::English))
            .build();
        tokenizers.insert(Lang::Eng, TokenizerEnum::English(en_stem));

        tokenizers
    };
}

impl TokenizerEnum {
    fn token_stream<'a>(&'a mut self, text: &'a str) -> tantivy::tokenizer::BoxTokenStream<'a> {
        match self {
            TokenizerEnum::Simple(tokenizer) => BoxTokenStream::new(tokenizer.token_stream(text)),
            TokenizerEnum::Jieba(tokenizer) => BoxTokenStream::new(tokenizer.token_stream(text)),
            TokenizerEnum::English(tokenizer) => tokenizer.token_stream(text),
        }
    }
}

/*
Structure of the lava file
It is important to put the posting lists first. Just trust me bro.
| compressed posting lists line by line | compressed term dictionary | compressed posting list offsets|
8 bytes = offsets of compressed term dict | 8 bytes = offset of compressed posting list offsets
*/

/// Function that tokenizes the input text and returns a list of tokens.
#[pyfunction]
pub fn build_lava_natural_language(
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    language: Option<&PyAny>,
) -> PyResult<()> {
    let array = make_array(ArrayData::from_pyarrow(array)?);
    let uid = make_array(ArrayData::from_pyarrow(uid)?);

    let array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::ParseError("Expects string array as first argument".to_string()))
        .map_err(to_py_err)?;

    let uid = uid
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            ArrowError::ParseError("Expects uint64 array as second argument".to_string())
        })
        .map_err(to_py_err)?;

    if array.len() != uid.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "The length of the array and the uid array must be the same".to_string(),
        ));
    }

    let language = match language {
        Some(x) => {
            let array = make_array(ArrayData::from_pyarrow(x)?);

            let test = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    ArrowError::ParseError(
                        "Expects string array as optional third argument".to_string(),
                    )
                })
                .map_err(to_py_err)?;
            Some(test.clone())
        }
        None => None,
    };

    // let mut tokens: Vec<Vec<String>> = Vec::new();
    let mut inverted_index: BTreeMap<String, Vec<u64>> = BTreeMap::new();

    for i in 0..array.len() {
        let text = array.value(i);
        let lang = if let Some(ref language) = language {
            Lang::from_code(language.value(i))
        } else {
            detect(text).map(|info| info.lang())
        }
        .unwrap_or(Lang::Eng);

        let mut tokenizer = TOKENIZERS.get(&lang).unwrap_or(&DEFAULT_TOKENIZER).clone();
        // println!("text: {} {}", text, detect(text).unwrap_or(Info::new(Script::Latin, Lang::Eng, 0.0)).lang());

        // The following code can be optimized as multiple threads https://docs.rs/futures/0.3.30/futures/executor/struct.ThreadPool.html
        let mut token_stream = tokenizer.token_stream(text);
        // let mut this_tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            // this_tokens.push(token.text.to_string());
            inverted_index
                .entry(format!("{}\n", token.text))
                .or_insert_with(Vec::new)
                .push(uid.value(i));
        }
        // tokens.push(this_tokens);
    }

    let total_length: usize = inverted_index.keys().map(|k| k.len()).sum();
    let mut term_dictionary = String::with_capacity(total_length);
    for key in inverted_index.keys() {
        term_dictionary.push_str(key);
    }

    let mut file = File::create(output_file_name.to_str()?)?;

    let bytes = term_dictionary.as_bytes();
    let compressed_term_dictionary = encode_all(bytes, 0).expect("Compression failed");

    // Handle the compressed data (for example, saving to a file or sending over a network)
    println!(
        "Compressed term dictionary length: {}",
        compressed_term_dictionary.len()
    );

    let mut plist_offsets: Vec<u64> = Vec::with_capacity(inverted_index.len() + 1);
    plist_offsets.push(0);

    for (_, value) in inverted_index.iter() {
        let serialized = bincode::serialize(&value).unwrap();
        let compressed_plist = encode_all(&serialized[..], 0).expect("Compression failed");
        plist_offsets.push(plist_offsets[plist_offsets.len() - 1] + compressed_plist.len() as u64);
        file.write_all(&compressed_plist)?;
    }

    let compressed_term_dict_offset = file.seek(SeekFrom::Current(0))?;
    file.write_all(&compressed_term_dictionary)?;

    let compressed_plist_offsets_offset = file.seek(SeekFrom::Current(0))?;
    let serialized = bincode::serialize(&plist_offsets).unwrap();
    let compressed_plist_offsets =
        encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");
    file.write_all(&compressed_plist_offsets)?;

    file.write_all(&(compressed_term_dict_offset as u64).to_le_bytes())?;
    file.write_all(&(compressed_plist_offsets_offset as u64).to_le_bytes())?;

    Ok(())
}
