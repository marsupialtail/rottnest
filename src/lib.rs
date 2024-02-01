use pyo3::exceptions::PyValueError;
use pyo3::types::PyString;
use pyo3::prelude::*;

use arrow::array::{make_array, Array, ArrayAccessor, ArrayData, StringArray, UInt64Array};
use arrow::error::ArrowError;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowException, PyArrowType, ToPyArrow};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::{RecordBatchIterator, RecordBatchReader};

use tantivy::schema::*;
use tantivy::tokenizer::*;
// use tantivy_tokenizer_api::Tokenizer as TantivyApiTokenizerTrait;
use tantivy_jieba::JiebaTokenizer;

use std::collections::BTreeMap;
use std::collections::HashMap;

use bincode;
use serde::{Deserialize, Serialize};
use std::fs::{metadata, File};
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;
use zstd::stream::write::Encoder;

use regex::Regex;
use whatlang::{detect, Info, Lang, Script};

use anyhow::Result;
use lazy_static::lazy_static;
use opendal::raw::oio::ReadExt;
use opendal::services::Fs;
use opendal::services::S3;
use opendal::Metadata;
use opendal::{Builder, Operator, Reader};
use std::env;

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
            TokenizerEnum::Simple(tokenizer) => tokenizer.token_stream(text).into(),
            TokenizerEnum::Jieba(tokenizer) => tokenizer.token_stream(text).into(),
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
fn build_lava_natural_language(
    output_file_name: &PyString,
    array: &PyAny,
    uid: &PyAny,
    language: Option<&PyAny>,
    py: Python,
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

#[tokio::main]
async fn hoa(condensed_lava_file: &str, operator: &mut Operator, lava_files: Vec<&str>) -> Result<()> // hawaiian for lava condensation
{
    // instantiate a list of readers from lava_files
    let mut readers: Vec<Reader> = Vec::with_capacity(lava_files.len());
    let mut decompressed_term_dictionaries: Vec<BufReader<Cursor<Vec<u8>>>> = Vec::with_capacity(lava_files.len());
    let mut file_sizes: Vec<u64> = Vec::with_capacity(lava_files.len());
    let mut plist_offsets: Vec<Vec<u64>> = Vec::with_capacity(lava_files.len());

    // read in and decompress all the term dictionaries in memory. The term dictionaries corresponding to English language should be small.

    for file in lava_files {
        
        let file_size: u64 = operator.stat(file).await?.content_length();
        let mut reader: Reader = operator.clone().reader(file).await?;
        reader.seek(SeekFrom::End(-16)).await?;
        let mut buffer1 = [0u8; 8];
        reader.read(&mut buffer1).await?;
        let compressed_term_dict_offset = u64::from_le_bytes(buffer1);
        let mut buffer: [u8; 8] = [0u8; 8];
        reader.read(&mut buffer[..]).await?;
        let compressed_plist_offsets_offset = u64::from_le_bytes(buffer);

        reader.seek(SeekFrom::Start(compressed_term_dict_offset)).await?;
        let mut compressed_term_dictionary: Vec<u8> = vec![0u8; (compressed_plist_offsets_offset - compressed_term_dict_offset) as usize];
        reader.read(&mut compressed_term_dictionary[..]).await?;
        let mut decompressed_term_dictionary: Vec<u8> = Vec::new();
        let mut decompressor: Decoder<'_, BufReader<&[u8]>> = Decoder::new(&compressed_term_dictionary[..])?;
        decompressor.read_to_end(&mut decompressed_term_dictionary)?;
        let cursor = Cursor::new(decompressed_term_dictionary);
        let buf_reader: BufReader<Cursor<Vec<u8>>> = BufReader::new(cursor);

        reader.seek(SeekFrom::Start(compressed_plist_offsets_offset)).await?;
        let mut buffer2: Vec<u8> = vec![0u8; (file_size - compressed_plist_offsets_offset - 16) as usize];
        reader.read(&mut buffer2).await?;
        decompressor = Decoder::new(&buffer2[..])?;
        let mut decompressed_serialized_plist_offsets: Vec<u8> = Vec::with_capacity(buffer2.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist_offsets)?;
        let this_plist_offsets: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist_offsets)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Bincode deserialization error: {}", e)))?;
    
        plist_offsets.push(this_plist_offsets);
        decompressed_term_dictionaries.push(buf_reader);
        file_sizes.push(file_size);
        readers.push(reader);
    }

    // now do the merge sort

    let mut term_dictionary = String::new();
    let mut current_lines: Vec<Option<String>> = vec![None; decompressed_term_dictionaries.len()];
    let mut plist_cursor: Vec<u64> = vec![0; decompressed_term_dictionaries.len()];

    // Initialize the current line for each reader
    for (i, reader) in decompressed_term_dictionaries.iter_mut().enumerate() {
        if let Some(Ok(line)) = reader.lines().next() {
            current_lines[i] = Some(line);
        }
    }

    let mut output_file = File::create(condensed_lava_file)?;
    let mut new_plist_offsets = Vec::new();
    new_plist_offsets.push(0);

    while current_lines.iter().any(Option::is_some) {

        // Find the smallest current line
        let smallest_line = current_lines
            .iter()
            .filter_map(|line| line.as_ref()) // Extract only Some(&String) elements
            .min()
            .unwrap()
            .clone(); // Find the minimum line
    
        term_dictionary += &smallest_line;
        term_dictionary += "\n";

        // Progress the BufReaders whose last output was smallest line

        let mut plist: Vec<u64> = Vec::new();

        for i in 0..current_lines.len() {
            if current_lines[i].is_some() {
                let line = current_lines[i].as_ref().unwrap();
                if line.eq(&smallest_line) {

                    // we need to read and decompress the plists
                    let offset = plist_offsets[i][plist_cursor[i] as usize];
                    let size = plist_offsets[i][plist_cursor[i] as usize + 1] - plist_offsets[i][plist_cursor[i] as usize];
                    let file_reader = &mut readers[i];
                    file_reader.seek(SeekFrom::Start(offset as u64)).await?;
                    let mut compressed_plist = vec![0u8; size as usize];
                    file_reader.read(&mut compressed_plist).await?;
                    let mut decompressor = Decoder::new(&compressed_plist[..])?;
                    let mut decompressed_serialized_plist: Vec<u8> = Vec::with_capacity(compressed_plist.len() as usize);
                    decompressor.read_to_end(&mut decompressed_serialized_plist)?;
                    let mut this_plist: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Bincode deserialization error: {}", e)))?;
                   
                    plist.append(&mut this_plist);
        
                    plist_cursor[i] += 1;

                    let reader = &mut decompressed_term_dictionaries[i];
                    if let Some(Ok(line)) = reader.lines().next() {
                        current_lines[i] = Some(line);
                    } else {
                        current_lines[i] = None;
                    }
                }
            }
        }

        // print out plist
        let serialized = bincode::serialize(&plist).unwrap();
        let compressed_plist = encode_all(&serialized[..], 0).expect("Compression failed");
        new_plist_offsets.push(new_plist_offsets[new_plist_offsets.len() - 1] + compressed_plist.len() as u64);
        output_file.write_all(&compressed_plist)?;
        
    }

    println!("{:?}", term_dictionary);
    let bytes = term_dictionary.as_bytes();
    let compressed_term_dictionary = encode_all(bytes, 0).expect("Compression failed");
    let compressed_term_dict_offset = output_file.seek(SeekFrom::Current(0))?;

    output_file.write_all(&compressed_term_dictionary)?;

    let compressed_plist_offsets_offset = output_file.seek(SeekFrom::Current(0))?;
    let serialized = bincode::serialize(&new_plist_offsets).unwrap();
    let compressed_plist_offsets =
        encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");
    output_file.write_all(&compressed_plist_offsets)?;

    output_file.write_all(&(compressed_term_dict_offset as u64).to_le_bytes())?;
    output_file.write_all(&(compressed_plist_offsets_offset as u64).to_le_bytes())?;

    Ok(())
}

#[pyfunction]
fn merge_lava(condensed_lava_file: &PyString, lava_files: Vec<&PyString>) -> PyResult<()> 
{
    // you should only merge them on local disk. It's not worth random accessing S3 for this because of the request costs. 
    // worry about running out of disk later. Assume you have a fast SSD for now.
    let mut builder = Fs::default();
    let current_path = env::current_dir()?;
    builder.root(current_path.to_str().expect("no path"));
    let mut operator = Operator::new(builder).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Fs Builder construction error: {}", e)))?.finish();

    let result = hoa(condensed_lava_file.to_str()?, &mut operator, lava_files.into_iter().map(|x| x.to_str().unwrap()).collect::<Vec<&str>>());

    Ok(result.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("merging error: {}", e)))?)
}


#[tokio::main]
async fn search_lava_async(operator: &mut Operator, file: &str, query: &str) -> Result<Vec<u64>> {
    let file_size: u64 = operator.stat(file).await?.content_length();
    let mut reader: Reader = operator.clone().reader(file).await?;

    reader.seek(SeekFrom::End(-16)).await?;
    let mut buffer1 = [0u8; 8];
    reader.read(&mut buffer1).await?;
    let compressed_term_dictionary_offset = u64::from_le_bytes(buffer1);

    let mut buffer: [u8; 8] = [0u8; 8];
    reader.read(&mut buffer[..]).await?;
    let compressed_plist_offsets_offset: u64 = u64::from_le_bytes(buffer);

    // now read the term dictionary
    let mut compressed_term_dictionary: Vec<u8> =
        vec![0u8; (compressed_plist_offsets_offset - compressed_term_dictionary_offset) as usize];
    reader.seek(SeekFrom::Start(compressed_term_dictionary_offset)).await?;
    reader.read(&mut compressed_term_dictionary[..]).await?;

    let mut decompressed_term_dictionary: Vec<u8> = Vec::new();
    let mut decompressor: Decoder<'_, BufReader<&[u8]>> =
        Decoder::new(&compressed_term_dictionary[..])?;
    decompressor.read_to_end(&mut decompressed_term_dictionary)?;    

    let cursor = Cursor::new(decompressed_term_dictionary);
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

    // seek to the offset
    reader.seek(SeekFrom::Start(compressed_plist_offsets_offset)).await?;
    let mut buffer2: Vec<u8> = vec![0u8; (file_size - compressed_plist_offsets_offset - 16) as usize];
    reader.read(&mut buffer2).await?;
    decompressor = Decoder::new(&buffer2[..])?;
    let mut decompressed_serialized_plist_offsets: Vec<u8> =
        Vec::with_capacity(buffer2.len() as usize);
    decompressor.read_to_end(&mut decompressed_serialized_plist_offsets)?;
    let plist_offsets: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist_offsets)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Bincode deserialization error: {}", e))
        })?;
    
    // now read the plist offsets that you need, whose indices are in matched

    let mut plist_result: Vec<u64> = Vec::new();
    for i in matched {
        reader.seek(SeekFrom::Start(plist_offsets[i as usize]))
            .await?;
        let mut buffer3: Vec<u8> =
            vec![0u8; (plist_offsets[(i + 1) as usize] - plist_offsets[i as usize]) as usize];
        reader.read(&mut buffer3).await?;
        decompressor = Decoder::new(&buffer3[..])?;
        let mut decompressed_serialized_plist: Vec<u8> = Vec::with_capacity(buffer3.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist)?;
        let mut plist: Vec<u64> =
            bincode::deserialize(&decompressed_serialized_plist).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Bincode deserialization error: {}",
                    e
                ))
            })?;
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
            filename = file[5..]
                .split("/")
                .skip(1)
                .collect::<Vec<&str>>()
                .join("/");
            // Set the region. This is required for some services, if you don't care about it, for example Minio service, just set it to "auto", it will be ignored.
            builder.region("us-west-2");
            builder.enable_virtual_host_style();
            builder.endpoint("");
            Operator::new(builder)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "S3 builder construction error: {}",
                        e
                    ))
                })?
                .finish()
        }
        false => {
            let mut builder = Fs::default();
            let current_path = env::current_dir()?;
            builder.root(current_path.to_str().expect("no path"));
            Operator::new(builder)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Fs Builder construction error: {}",
                        e
                    ))
                })?
                .finish()
        }
    };

    println!("Searching {}", filename);
    let result: Result<Vec<u64>, anyhow::Error> =
        search_lava_async(&mut operator, &filename, query);

    Ok(result
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("searching error: {}", e)))?)
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn rottnest_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_lava_natural_language, m)?)?;
    m.add_function(wrap_pyfunction!(search_lava, m)?)?;
    m.add_function(wrap_pyfunction!(merge_lava, m)?)?;
    Ok(())
}
