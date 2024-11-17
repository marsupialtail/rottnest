use crate::lava::constants::*;
use crate::lava::fm_chunk::FMChunk;
use crate::lava::plist::PListChunk;
use crate::{
    formats::readers::{
        get_file_size_and_reader, get_file_sizes_and_readers, get_reader, get_readers, AsyncReader,
        ClonableAsyncReader, ReaderType,
    },
    lava::error::LavaError,
};
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use itertools::Itertools;
use serde::de::DeserializeOwned;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;
use std::{
    collections::{HashMap, HashSet},
    io::Read,
};
use tokenizers::tokenizer::Tokenizer;
use tokio::task::JoinSet;

use futures::stream::{FuturesUnordered, StreamExt};
use std::cmp::Ordering;
use std::io::{self, Cursor};

use super::bm25::search_bm25_async;

enum QueryParam {
    SubstringCharWavelet(Vec<Vec<u8>>),
    SubstringChar(Vec<Vec<u8>>),
    Substring(Vec<Vec<u32>>),
    Uuid(String),
}
async fn get_tokenizer_async(
    mut readers: Vec<AsyncReader>,
) -> Result<(Tokenizer, Vec<String>), LavaError> {
    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    for i in 0..readers.len() {
        // now interpret this as a usize
        // readers[i].seek(SeekFrom::Start(0)).await?;
        let compressed_tokenizer_size = readers[i].read_usize_from_start(0, 1).await?[0];
        let this_compressed_tokenizer: bytes::Bytes = readers[i]
            .read_range(8, 8 + compressed_tokenizer_size)
            .await?;
        match &compressed_tokenizer {
            Some(value) => assert!(
                this_compressed_tokenizer == value,
                "detected different tokenizers between different lava files, can't search across them."
            ),
            None => compressed_tokenizer = Some(this_compressed_tokenizer.to_vec()),
        }
    }

    let slice = &compressed_tokenizer.unwrap()[..];
    let mut decompressor = Decoder::new(slice)?;
    let mut decompressed_serialized_tokenizer: Vec<u8> = Vec::with_capacity(slice.len() as usize);
    decompressor.read_to_end(&mut decompressed_serialized_tokenizer)?;

    let mut result: Vec<String> = Vec::new();
    let tokenizer = Tokenizer::from_bytes(decompressed_serialized_tokenizer).unwrap();

    for i in 0..tokenizer.get_vocab_size(false) {
        let tok = tokenizer.decode(&vec![i as u32], false).unwrap();
        result.push(tok);
    }

    Ok((tokenizer, result))
}

async fn search_uuid_one_file(
    file_id: u64,
    mut reader: AsyncReader,
    file_size: usize,
    query: String,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let mut result: Vec<(u64, u64)> = Vec::new();
    let mut start_time = Instant::now();

    let this_result: Vec<usize> =
        FastTrie::query_with_reader(file_size, &mut reader, &query).await?;
    result.extend(this_result.iter().map(|x| (file_id, *x as u64)));

    Ok(result)
}

async fn search_generic_async(
    mut file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query: QueryParam,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let mut join_set = JoinSet::new();

    let mut start_time = Instant::now();
    for file_id in 0..readers.len() {
        let reader = readers.remove(0);
        let file_size = file_sizes.remove(0);

        match query {
            QueryParam::Substring(ref value) => {
                join_set.spawn(search_substring_one_file::<u32>(
                    file_id as u64,
                    reader,
                    file_size,
                    value.clone(),
                ));
            }
            QueryParam::SubstringChar(ref value) => {
                join_set.spawn(search_substring_one_file::<u8>(
                    file_id as u64,
                    reader,
                    file_size,
                    value.clone(),
                ));
            }
            QueryParam::SubstringCharWavelet(ref value) => {
                join_set.spawn(search_substring_wavelet_one_file(
                    file_id as u64,
                    reader,
                    file_size,
                    value.clone(),
                ));
            }
            QueryParam::Uuid(ref value) => {
                join_set.spawn(search_uuid_one_file(
                    file_id as u64,
                    reader,
                    file_size,
                    value.clone(),
                ));
            }
            _ => panic!("invalid mode"),
        }
    }

    let mut result: BTreeSet<(u64, u64)> = BTreeSet::new();
    while let Some(res) = join_set.join_next().await {
        let res = res.unwrap().unwrap();
        result.extend(res);
        /*
        We cannot truncate to k anywhere, not even at the end, because of false positives
         */
        // if result.len() >= k {
        //     break;
        // }
    }

    join_set.shutdown().await;

    println!("Time stage 1 read: {:?}", start_time.elapsed());

    let result: Vec<(u64, u64)> = result.into_iter().collect_vec();
    Ok(result)
}

#[tokio::main]
pub async fn search_lava_bm25(
    files: Vec<String>,
    query_tokens: Vec<u32>,
    query_weights: Vec<f32>,
    k: usize,
    reader_type: ReaderType,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    search_bm25_async(file_sizes, readers, query_tokens, query_weights, k).await
}

#[tokio::main]
pub async fn search_lava_uuid(
    files: Vec<String>,
    query: String,
    k: usize,
    reader_type: ReaderType,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    search_generic_async(file_sizes, readers, QueryParam::Uuid(query), k).await
}

#[tokio::main]
pub async fn search_lava_substring(
    files: Vec<String>,
    query: String,
    k: usize,
    reader_type: ReaderType,
    token_viable_limit: Option<usize>,
    sample_factor: Option<usize>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let (_file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;
    let tokenizer = get_tokenizer_async(readers).await?.0;

    let mut skip_tokens: HashSet<u32> = HashSet::new();
    for char in SKIP.chars() {
        let char_str = char.to_string();
        skip_tokens.extend(
            tokenizer
                .encode(char_str.clone(), false)
                .unwrap()
                .get_ids()
                .to_vec(),
        );
        skip_tokens.extend(
            tokenizer
                .encode(format!(" {}", char_str), false)
                .unwrap()
                .get_ids()
                .to_vec(),
        );
        skip_tokens.extend(
            tokenizer
                .encode(format!("{} ", char_str), false)
                .unwrap()
                .get_ids()
                .to_vec(),
        );
    }

    let lower: String = query.chars().flat_map(|c| c.to_lowercase()).collect();
    let encoding = tokenizer.encode(lower, false).unwrap();
    let result: Vec<u32> = encoding
        .get_ids()
        .iter()
        .filter(|id| !skip_tokens.contains(id))
        .cloned()
        .collect();

    let mut query: Vec<Vec<u32>> = if let Some(sample_factor) = sample_factor {
        (0..sample_factor)
            .map(|offset| {
                result
                    .iter()
                    .skip(offset)
                    .step_by(sample_factor)
                    .cloned()
                    .collect::<Vec<u32>>()
            })
            .filter(|vec| !vec.is_empty())
            .collect()
    } else {
        vec![result]
    };

    // println!("query {:?}", query);

    // query = [i[-token_viable_limit:] for i in query]
    if let Some(token_viable_limit) = token_viable_limit {
        query.iter_mut().for_each(|vec| {
            if vec.len() > token_viable_limit {
                *vec = vec
                    .iter()
                    .rev()
                    .take(token_viable_limit)
                    .rev()
                    .cloned()
                    .collect();
            }
        });
    }

    // println!("query {:?}", query);

    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    search_generic_async(file_sizes, readers, QueryParam::Substring(query), k).await
}

#[tokio::main]
pub async fn search_lava_substring_char(
    files: Vec<String>,
    query: String,
    k: usize,
    reader_type: ReaderType,
    token_viable_limit: Option<usize>,
    sample_factor: Option<usize>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    _search_lava_substring_char(
        files,
        query,
        k,
        reader_type,
        token_viable_limit,
        sample_factor,
        false,
    )
    .await
}

#[tokio::main]
pub async fn get_tokenizer_vocab(
    files: Vec<String>,
    reader_type: ReaderType,
) -> Result<Vec<String>, LavaError> {
    let (_file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    Ok(get_tokenizer_async(readers).await?.1)
}

#[cfg(test)]
mod tests {
    use crate::formats::readers::ReaderType;

    use super::search_lava_bm25;
    use super::search_lava_substring;

    #[test]
    pub fn test_search_lava_one() {
        let file = "msmarco_index/1.lava";

        let res = search_lava_bm25(
            vec![file.to_string()],
            vec![6300, 15050],
            vec![0.1, 0.2],
            10,
            ReaderType::default(),
        )
        .unwrap();

        println!("{:?}", res);
    }

    #[test]
    pub fn test_search_lava_two() {
        let res = search_lava_bm25(
            vec!["bump1.lava".to_string(), "bump2.lava".to_string()],
            vec![6300, 15050],
            vec![0.1, 0.2],
            10,
            ReaderType::default(),
        )
        .unwrap();

        println!("{:?}", res);
    }

    #[test]
    pub fn test_search_substring() {
        let result = search_lava_substring(
            vec!["chinese_index/0.lava".to_string()],
            "Samsung Galaxy Note".to_string(),
            10,
            ReaderType::default(),
            Some(10),
            None,
        );
        println!("{:?}", result.unwrap());
    }
}
