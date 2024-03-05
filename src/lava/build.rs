use arrow::array::{make_array, Array, ArrayData, StringArray, UInt64Array};
use serde_json;
use tokenizers::tokenizer::Tokenizer;
use tokio::task::JoinHandle;

use bincode;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use zstd::stream::encode_all;

use std::io::Cursor;

use crate::lava::error::LavaError;
use crate::lava::plist::PListChunk;

use libdivsufsort_rs::divsufsort64;

/*
Structure of the lava file
It is important to put the posting lists first. Just trust me bro.
compressed_serialized_tokenizer | compressed posting lists line by line | compressed term dictionary | compressed posting list offsets|
8 bytes = offsets of compressed term dict | 8 bytes = offset of compressed posting list offsets
*/

/// Function that tokenizes the input text and returns a list of tokens.

#[tokio::main]
pub async fn build_lava_bm25(
    output_file_name: String,
    array: ArrayData,
    uid: ArrayData,
    tokenizer_file: Option<String>,
    k1: Option<f32>,
    b: Option<f32>,
) -> Result<(), LavaError> {
    // if k1 and b are not provided, set them to default value
    let k1: f32 = k1.unwrap_or(1.2);
    let b: f32 = b.unwrap_or(0.75);

    let array = make_array(array);
    // let uid = make_array(ArrayData::from_pyarrow(uid)?);
    let uid = make_array(uid);

    // if the tokenizer file is provided, check if the file exists. If it does not exist, raise an Error
    let tokenizer = if let Some(tokenizer_file) = tokenizer_file {
        if !std::path::Path::new(&tokenizer_file).exists() {
            return Err(LavaError::Parse(
                "Tokenizer file does not exist".to_string(),
            ));
        }
        println!("Tokenizer file: {}", tokenizer_file);
        Tokenizer::from_file(tokenizer_file).unwrap()
    } else {
        Tokenizer::from_pretrained("bert-base-uncased", None).unwrap()
    };

    let serialized_tokenizer = serde_json::to_string(&tokenizer).unwrap();
    let compressed_tokenizer =
        encode_all(serialized_tokenizer.as_bytes(), 0).expect("Compression failed");

    let array: &arrow_array::GenericByteArray<arrow_array::types::GenericStringType<i32>> = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(LavaError::Parse(
            "Expects string array as first argument".to_string(),
        ))?;

    let uid = uid
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(LavaError::Parse(
            "Expects uint64 array as second argument".to_string(),
        ))?;

    if array.len() != uid.len() {
        return Err(LavaError::Parse(
            "The length of the array and the uid array must be the same".to_string(),
        ));
    }

    let vocab_size: usize = tokenizer.get_vocab_size(false);
    let mut handles = Vec::new();
    for i in 0..array.len() {
        let text = array.value(i).to_string();
        let tokenizer = tokenizer.clone();
        let handle: JoinHandle<Result<Vec<u32>, LavaError>> = tokio::spawn(async move {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|_e| LavaError::Unknown)?;
            let ids = encoding.get_ids().to_vec();
            Ok(ids)
        });
        handles.push(handle);
    }
    let encodings = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|res| res.unwrap().unwrap())
        .collect::<Vec<Vec<u32>>>();

    let mut inverted_index: Vec<BTreeMap<usize, f32>> = vec![BTreeMap::new(); vocab_size];
    let mut token_counts: Vec<usize> = vec![0; vocab_size];

    let mut avg_len: f32 = 0.0;
    for encoding in encodings.iter() {
        avg_len += encoding.len() as f32;
    }
    avg_len /= encodings.len() as f32;

    for (i, encoding) in encodings.iter().enumerate() {
        let uid = uid.value(i) as usize;
        let mut local_token_counts: BTreeMap<u32, usize> = BTreeMap::new();
        for key in encoding {
            *local_token_counts.entry(*key).or_insert(0) += 1;
        }
        for key in local_token_counts.keys() {
            let local_count = local_token_counts[key];
            let local_factor: f32 = (local_count as f32) * (k1 + 1.0)
                / (local_count as f32 + k1 * (1.0 - b + b * encoding.len() as f32 / avg_len));

            inverted_index[*key as usize]
                .entry(uid)
                .and_modify(|e| *e = (*e).max(local_factor))
                .or_insert(local_factor);

            token_counts[*key as usize] += 1;
        }
    }

    let mut file = File::create(output_file_name)?;
    file.write_all(&compressed_tokenizer)?;

    let bytes = bincode::serialize(&token_counts)?;
    let compressed_token_counts: Vec<u8> = encode_all(&bytes[..], 0).expect("Compression failed");

    // Handle the compressed data (for example, saving to a file or sending over a network)
    println!(
        "Compressed token counts size: {} number of tokens: {}",
        compressed_token_counts.len(),
        inverted_index.len()
    );

    let mut plist_offsets: Vec<u64> = vec![file.seek(SeekFrom::Current(0))?];
    let mut plist_elems: Vec<u64> = vec![0];
    let mut plist_chunk = PListChunk::new()?;
    let mut counter: u64 = 0;

    for (_key, value) in inverted_index.iter().enumerate() {
        let plist = if value.len() == 0 {
            vec![]
        } else {
            let mut result = vec![];
            for (key, val) in value.iter() {
                result.push(*key as u64);
                // quantize the score to int.
                result.push((*val * 100 as f32) as u64);
            }
            result
        };

        counter += 1;

        let written = plist_chunk.add_plist(&plist)?;
        if written > 1024 * 1024 || counter == inverted_index.len() as u64 {
            let bytes = plist_chunk.finalize_compression()?;
            file.write_all(&bytes)?;
            plist_offsets.push(plist_offsets[plist_offsets.len() - 1] + bytes.len() as u64);
            plist_elems.push(counter);
            plist_chunk = PListChunk::new()?;
        }
    }

    println!("{}", counter);

    plist_offsets.append(&mut plist_elems);

    let compressed_term_dict_offset = file.seek(SeekFrom::Current(0))?;
    file.write_all(&compressed_token_counts)?;

    let compressed_plist_offsets_offset = file.seek(SeekFrom::Current(0))?;
    let serialized = bincode::serialize(&plist_offsets).unwrap();
    let compressed_plist_offsets =
        encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");
    file.write_all(&compressed_plist_offsets)?;

    file.write_all(&(compressed_term_dict_offset as u64).to_le_bytes())?;
    file.write_all(&(compressed_plist_offsets_offset as u64).to_le_bytes())?;
    file.write_all(&(encodings.len() as u64).to_le_bytes())?;

    Ok(())
}

#[tokio::main]
pub async fn build_lava_substring(
    output_file_name: String,
    array: ArrayData,
    uid: ArrayData,
) -> Result<(), LavaError> {
    let array = make_array(array);
    let array: &arrow_array::GenericByteArray<arrow_array::types::GenericStringType<i32>> = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(LavaError::Parse(
            "Expects string array as first argument".to_string(),
        ))?;

    let mut input = String::new();

    for i in 0..array.len() {
        let text = array.value(i).to_string();
        input.push_str(&text);
    }

    let input = input.as_bytes().to_vec();
    let sa = divsufsort64(&input).unwrap();

    let mut bwt = Vec::with_capacity(input.len());
    for i in 0..sa.len() {
        if sa[i] == 0 {
            bwt.push(input[input.len() - 1]);
        } else {
            bwt.push(input[(sa[i] - 1) as usize]);
        }
    }

    let compressed_term_dictionary = encode_all(Cursor::new(bwt), 0).expect("Compression failed");
    println!("{:?}", compressed_term_dictionary.len());
    Ok(())
}
