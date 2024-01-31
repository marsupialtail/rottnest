use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use arrow::array::{make_array, Array, ArrayData, UInt64Array, StringArray};
use arrow::error::ArrowError;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowException, PyArrowType, ToPyArrow};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::{RecordBatchIterator, RecordBatchReader};

use tantivy::schema::*;
use tantivy::tokenizer::*;
// use tantivy_tokenizer_api::Tokenizer as TantivyApiTokenizerTrait;
use tantivy_jieba::JiebaTokenizer;

use std::collections::HashMap;
use std::collections::BTreeMap;

use zstd::stream::read::Decoder;
use zstd::stream::write::Encoder;
use zstd::stream::encode_all;
use serde::{Serialize, Deserialize};
use bincode;
use std::io::{Read, Write, Cursor, BufReader, BufRead, Seek, SeekFrom};
use std::fs::{File, metadata};

use whatlang::{detect, Lang, Info, Script};
use regex::Regex;

use anyhow::Result;
use opendal::raw::oio::ReadExt;
use opendal::services::S3;
use opendal::services::Fs;
use opendal::Metadata;
use opendal::{Operator, Reader, Builder};
use std::env;

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

enum TokenizerEnum {
    Simple(SimpleTokenizer),
    Jieba(JiebaTokenizer),
    English(TextAnalyzer)
}

impl TokenizerEnum {
    fn token_stream<'a>(&'a mut self, text: &'a str) -> tantivy::tokenizer::BoxTokenStream<'a>  {
        match self {
            TokenizerEnum::Simple(tokenizer) => {
                tokenizer.token_stream(text).into()
            },
            TokenizerEnum::Jieba(tokenizer) => {
                tokenizer.token_stream(text).into()
            },
            TokenizerEnum::English(tokenizer) => {
                tokenizer.token_stream(text)
            }

        }
    }
}

/// Function that tokenizes the input text and returns a list of tokens.
#[pyfunction]
fn tokenize_natural_language(array: &PyAny, uid: &PyAny, language: Option<&PyAny>, py: Python) -> PyResult<()> {
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
        .ok_or_else(|| ArrowError::ParseError("Expects uint64 array as second argument".to_string()))
        .map_err(to_py_err)?;

    if array.len() != uid.len() {
        return Err(PyErr::new::<PyValueError, _>("The length of the array and the uid array must be the same".to_string()));
    }

    let language = match language {
        Some(x) => {
            let array = make_array(ArrayData::from_pyarrow(x)?);

            let test = array.as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| ArrowError::ParseError("Expects string array as optional third argument".to_string()))
                            .map_err(to_py_err)?;
            Some(test.clone())
        },
        None => None
    };

    let mut tokenizers: HashMap<String, TokenizerEnum> = HashMap::new();
    tokenizers.insert("cn".to_string(), TokenizerEnum::Jieba(JiebaTokenizer {}));
    tokenizers.insert("default".to_string(), TokenizerEnum::Simple(SimpleTokenizer::default()));
    
    let en_stem  = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(Stemmer::new(Language::English))
        .build();   

    tokenizers.insert("en".to_string(), TokenizerEnum::English(en_stem));

    // let mut tokens: Vec<Vec<String>> = Vec::new();
    let mut inverted_index: BTreeMap<String, Vec<u64>> = BTreeMap::new();

    for i in 0..array.len() {
        
        let text = array.value(i);
        // println!("text: {} {}", text, detect(text).unwrap_or(Info::new(Script::Latin, Lang::Eng, 0.0)).lang());

        let tokenizer = match language {
            // if the language is provided use the provided
            Some(ref x) => {
                let lang = x.value(i);
                match lang {
                    "cn" => tokenizers.get_mut("cn"),
                    "en" => tokenizers.get_mut("en"),
                    _ => tokenizers.get_mut("default")
                }.unwrap()
            }
            None  => {
                match detect(text).unwrap_or(Info::new(Script::Latin, Lang::Eng, 0.0)).lang() {
                    Lang::Cmn => tokenizers.get_mut("cn"),
                    Lang::Eng => tokenizers.get_mut("en"),
                    _ => tokenizers.get_mut("default")
                }.unwrap()
            }
        };


        let mut token_stream = tokenizer.token_stream(text);
        // let mut this_tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            // this_tokens.push(token.text.to_string());
            inverted_index.entry(format!("{}\n", token.text)).or_insert_with(Vec::new).push(uid.value(i));
        }
       // tokens.push(this_tokens);
    }

    let total_length: usize = inverted_index.keys().map(|k| k.len()).sum();
    let mut term_dictionary = String::with_capacity(total_length);
    for key in inverted_index.keys() {
        term_dictionary.push_str(key);
    }

    let mut file = File::create("output_file.bin")?;

    let bytes = term_dictionary.as_bytes();
    let compressed_term_dictionary = encode_all(bytes, 0).expect("Compression failed");

    // Handle the compressed data (for example, saving to a file or sending over a network)
    println!("Compressed term dictionary length: {}", compressed_term_dictionary.len());

    // first write the size of the compressed term dictionary, then write the dictionary itself
    file.write_all(&(compressed_term_dictionary.len() as u64).to_le_bytes())?;
    file.write_all(&compressed_term_dictionary)?;

    let mut plist_offsets: Vec<u64> = Vec::with_capacity(inverted_index.len() + 1);
    plist_offsets.push(0);
    
    for (_, value) in inverted_index.iter() {
        let serialized = bincode::serialize(&value).unwrap();
        let compressed_plist = encode_all(&serialized[..], 0).expect("Compression failed");
        plist_offsets.push(plist_offsets[plist_offsets.len() - 1] + compressed_plist.len() as u64);
        file.write_all(&compressed_plist)?;
    }
    let offset = file.seek(SeekFrom::Current(0))?;
    let serialized = bincode::serialize(&plist_offsets).unwrap();
    let compressed_plist_offsets = encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");
    file.write_all(&compressed_plist_offsets)?;
    file.write_all(&(offset as u64).to_le_bytes())?;
    
    Ok(())

}

#[tokio::main]
async fn search_lava_async(operator: &mut Operator, file: &str, query: &str) -> Result<Vec<u64>> {

    let file_size: u64 = operator.stat(file).await?.content_length();
    let mut reader: Reader = operator.clone().reader(file).await?;
    
    let mut buffer: [u8; 8] = [0u8; 8];
    reader.read(&mut buffer[..]).await?;
    let compressed_term_dictionary_size: u64 = u64::from_le_bytes(buffer);
    
    // now read the term dictionary
    let mut compressed_term_dictionary: Vec<u8> = vec![0u8; compressed_term_dictionary_size as usize];
    reader.read(&mut compressed_term_dictionary[..]).await?;
    
    let mut decompressed_term_dictionary: Vec<u8> = Vec::with_capacity(compressed_term_dictionary_size as usize);
    let mut decompressor: Decoder<'_, BufReader<&[u8]>> = Decoder::new(&compressed_term_dictionary[..])?;
    decompressor.read_to_end(&mut decompressed_term_dictionary)?;

    let cursor = Cursor::new(decompressed_term_dictionary);

    // Use a BufReader to read lines from the Cursor
    let buf_reader: BufReader<Cursor<Vec<u8>>> = BufReader::new(cursor);
    let mut matched: Vec<u64> = Vec::new();
    let re: Regex = Regex::new(query).unwrap();

    let mut counter: u64 = 0;
    for line in buf_reader.lines() {
        let line = line?; // Handle potential errors on each line
        if re.is_match(&line) {
            matched.push(counter);
        }
        counter += 1;
    }

    if matched.len() == 0 {
        return Ok(matched);
    }

    // read the last eight bytes
    reader.seek(SeekFrom::End(-8)).await?;
    let mut buffer1 = [0u8; 8];
    reader.read(&mut buffer1).await?;
    let offset = u64::from_le_bytes(buffer1);

    // seek to the offset
    reader.seek(SeekFrom::Start(offset)).await?;
    let mut buffer2: Vec<u8> = vec![0u8; (file_size - offset - 8) as usize];
    reader.read(&mut buffer2).await?;
    decompressor = Decoder::new(&buffer2[..])?;
    let mut decompressed_serialized_plist_offsets: Vec<u8> = Vec::with_capacity(buffer2.len() as usize);
    decompressor.read_to_end(&mut decompressed_serialized_plist_offsets)?;
    let plist_offsets: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist_offsets)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Bincode deserialization error: {}", e)))?;

    // now read the plist offsets that you need, whose indices are in matched

    let mut plist_result : Vec<u64> = Vec::new();
    for i in matched {
        reader.seek(SeekFrom::Start(8 + compressed_term_dictionary_size + plist_offsets[i as usize])).await?;
        let mut buffer3: Vec<u8> = vec![0u8; (plist_offsets[(i+1) as usize] - plist_offsets[i as usize]) as usize];
        reader.read(&mut buffer3).await?;
        decompressor = Decoder::new(&buffer3[..])?;
        let mut decompressed_serialized_plist: Vec<u8> = Vec::with_capacity(buffer3.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist)?;
        let mut plist: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Bincode deserialization error: {}", e)))?;
        plist_result.append(&mut plist);
    }

    Ok(plist_result)
}

#[pyfunction]
fn search_lava(file: &str, query: &str) -> PyResult<Vec<u64>> {

    let mut filename: String = file.to_string();
    let mut operator: Operator = match file.starts_with("s3://") {
        true => {
            let mut builder = S3::default();
            builder.bucket(file[5..].split("/").next().expect("malformed path"));
            filename = file[5..].split("/").skip(1).collect::<Vec<&str>>().join("/");
            // Set the region. This is required for some services, if you don't care about it, for example Minio service, just set it to "auto", it will be ignored.
            builder.region("us-west-2");
            builder.enable_virtual_host_style();
            builder.endpoint("");
            Operator::new(builder).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("S3 builder construction error: {}", e)))?.finish()
        },
        false => {
            let mut builder = Fs::default();
            let current_path = env::current_dir()?;
            builder.root(current_path.to_str().expect("no path"));
            Operator::new(builder).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Fs Builder construction error: {}", e)))?.finish()
        }
    };

    let result: Result<Vec<u64>, anyhow::Error> = search_lava_async(&mut operator, &filename, query);

    Ok(result.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("searching error: {}", e)))?)
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn rottnest_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_natural_language, m)?)?;
    m.add_function(wrap_pyfunction!(search_lava, m)?)?;
    Ok(())
}
