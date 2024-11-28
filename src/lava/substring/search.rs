use super::constants::*;
use super::fm_chunk::FMChunk;
use crate::formats::readers::{get_file_sizes_and_readers, AsyncReader, ReaderType};
use crate::lava::error::LavaError;
use crate::lava::get_tokenizer_async;

use bincode;

use itertools::Itertools;

use std::collections::{BTreeSet, HashSet};
use std::fmt::Debug;
use std::fs::File;
use std::io::Read;
// You'll need the `byteorder` crate
use tokio::task::JoinSet;
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;

use num_traits::{AsPrimitive, PrimInt, Unsigned};
use serde::{Deserialize, Serialize};
use std::ops::Add;
use std::time::Instant;

enum QueryParam {
    SubstringCharWavelet(Vec<Vec<u8>>),
    SubstringChar(Vec<Vec<u8>>),
    Substring(Vec<Vec<u32>>),
    Uuid(String),
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

async fn process_substring_query<T>(
    query: Vec<T>,
    n: u64,
    fm_chunk_offsets: &[u64],
    cumulative_counts: &[u64],
    posting_list_offsets: &[u64],
    reader: &mut AsyncReader,
    file_id: u64,
) -> Vec<(u64, u64)>
where
    T: PrimInt
        + Unsigned
        + Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Eq
        + std::hash::Hash
        + AsPrimitive<usize>
        + 'static,
    usize: AsPrimitive<T>,
{
    let mut res: Vec<(u64, u64)> = vec![];
    let mut start: usize = 0;
    let mut end: usize = n as usize;

    for i in (0..query.len()).rev() {
        let current_token = query[i];

        let start_byte = fm_chunk_offsets[start / FM_CHUNK_TOKS];
        let end_byte = fm_chunk_offsets[start / FM_CHUNK_TOKS + 1];
        let start_chunk = reader.read_range(start_byte, end_byte).await.unwrap();

        let start_byte = fm_chunk_offsets[end / FM_CHUNK_TOKS];
        let end_byte = fm_chunk_offsets[end / FM_CHUNK_TOKS + 1];
        let end_chunk = reader.read_range(start_byte, end_byte).await.unwrap();

        start = cumulative_counts[current_token.as_()] as usize
            + FMChunk::<T>::new(start_chunk)
                .unwrap()
                .search(current_token, start % FM_CHUNK_TOKS)
                .unwrap() as usize;
        end = cumulative_counts[current_token.as_()] as usize
            + FMChunk::<T>::new(end_chunk)
                .unwrap()
                .search(current_token, end % FM_CHUNK_TOKS)
                .unwrap() as usize;

        if start >= end {
            return res;
        }

        if end <= start + 2 {
            break;
        }
    }

    let start_offset = posting_list_offsets[start / FM_CHUNK_TOKS];
    let end_offset = posting_list_offsets[end / FM_CHUNK_TOKS + 1];
    let total_chunks = end / FM_CHUNK_TOKS - start / FM_CHUNK_TOKS + 1;

    let plist_chunks = reader.read_range(start_offset, end_offset).await.unwrap();

    let mut chunk_set = JoinSet::new();

    for i in 0..total_chunks {
        let this_start = posting_list_offsets[start / FM_CHUNK_TOKS + i];
        let this_end = posting_list_offsets[start / FM_CHUNK_TOKS + i + 1];
        let this_chunk = plist_chunks
            [(this_start - start_offset) as usize..(this_end - start_offset) as usize]
            .to_vec();

        chunk_set.spawn(async move {
            let mut decompressor = Decoder::new(&this_chunk[..]).unwrap();
            let mut serialized_plist_chunk: Vec<u8> = Vec::with_capacity(this_chunk.len());
            decompressor
                .read_to_end(&mut serialized_plist_chunk)
                .unwrap();
            let plist_chunk: Vec<u64> = bincode::deserialize(&serialized_plist_chunk).unwrap();

            let chunk_res: Vec<(u64, u64)> = if i == 0 {
                if total_chunks == 1 {
                    plist_chunk[start % FM_CHUNK_TOKS..end % FM_CHUNK_TOKS]
                        .iter()
                        .map(|&uid| (file_id, uid))
                        .collect()
                } else {
                    plist_chunk[start % FM_CHUNK_TOKS..]
                        .iter()
                        .map(|&uid| (file_id, uid))
                        .collect()
                }
            } else if i == total_chunks - 1 {
                plist_chunk[..end % FM_CHUNK_TOKS]
                    .iter()
                    .map(|&uid| (file_id, uid))
                    .collect()
            } else {
                plist_chunk.iter().map(|&uid| (file_id, uid)).collect()
            };

            chunk_res
        });
    }

    while let Some(chunk_res) = chunk_set.join_next().await {
        res.extend(chunk_res.unwrap());
    }

    res
}

use super::wavelet_tree::search_wavelet_tree_from_reader;
use crate::formats::readers::read_and_decompress;

async fn search_substring_wavelet_one_file(
    file_id: u64,
    mut reader: AsyncReader,
    file_size: usize,
    queries: Vec<Vec<u8>>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    println!("{:?}", queries);

    let metadata_start = reader.read_usize_from_end(1).await?[0];

    let metadata: (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, usize) = read_and_decompress(
        &mut reader,
        metadata_start as u64,
        file_size as u64 - metadata_start - 8,
    )
    .await
    .unwrap();
    let (offsets, level_offsets, posting_list_offsets, cumulative_counts, n) = metadata;

    // let mut query_set = JoinSet::new();

    let mut res: Vec<(u64, u64)> = vec![];

    for query in queries {
        let mut reader = reader.clone();
        let (start, end) = search_wavelet_tree_from_reader(
            &mut reader,
            &query,
            n,
            &offsets,
            &level_offsets,
            &cumulative_counts,
        )
        .await?;

        println!("{} {}", start, end);

        if start == end {
            continue;
        }

        let start_offset = posting_list_offsets[start / FM_CHUNK_TOKS];
        let end_offset = posting_list_offsets[end / FM_CHUNK_TOKS + 1];
        let total_chunks = end / FM_CHUNK_TOKS - start / FM_CHUNK_TOKS + 1;

        let plist_chunks = reader
            .read_range(start_offset as u64, end_offset as u64)
            .await
            .unwrap();

        let mut chunk_set = JoinSet::new();

        for i in 0..total_chunks {
            let this_start = posting_list_offsets[start / FM_CHUNK_TOKS + i];
            let this_end = posting_list_offsets[start / FM_CHUNK_TOKS + i + 1];
            let this_chunk = plist_chunks
                [(this_start - start_offset) as usize..(this_end - start_offset) as usize]
                .to_vec();

            chunk_set.spawn(async move {
                let mut decompressor = Decoder::new(&this_chunk[..]).unwrap();
                let mut serialized_plist_chunk: Vec<u8> = Vec::with_capacity(this_chunk.len());
                decompressor
                    .read_to_end(&mut serialized_plist_chunk)
                    .unwrap();
                let plist_chunk: Vec<u64> = bincode::deserialize(&serialized_plist_chunk).unwrap();

                let chunk_res: Vec<(u64, u64)> = if i == 0 {
                    if total_chunks == 1 {
                        plist_chunk[start % FM_CHUNK_TOKS..end % FM_CHUNK_TOKS]
                            .iter()
                            .map(|&uid| (file_id, uid))
                            .collect()
                    } else {
                        plist_chunk[start % FM_CHUNK_TOKS..]
                            .iter()
                            .map(|&uid| (file_id, uid))
                            .collect()
                    }
                } else if i == total_chunks - 1 {
                    plist_chunk[..end % FM_CHUNK_TOKS]
                        .iter()
                        .map(|&uid| (file_id, uid))
                        .collect()
                } else {
                    plist_chunk.iter().map(|&uid| (file_id, uid)).collect()
                };

                chunk_res
            });
        }

        while let Some(chunk_res) = chunk_set.join_next().await {
            res.extend(chunk_res.unwrap());
        }
    }

    // let mut res = Vec::new();
    // while let Some(query_res) = query_set.join_next().await {
    //     res.extend(query_res.unwrap());
    // }

    Ok(res)
}

async fn search_substring_one_file<T>(
    file_id: u64,
    mut reader: AsyncReader,
    file_size: usize,
    queries: Vec<Vec<T>>,
) -> Result<Vec<(u64, u64)>, LavaError>
where
    T: PrimInt
        + Unsigned
        + Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Eq
        + std::hash::Hash
        + AsPrimitive<usize>
        + Debug
        + Send
        + 'static,
    usize: AsPrimitive<T>,
{
    println!("{:?}", queries);

    let results = reader.read_usize_from_end(4).await?;
    let fm_chunk_offsets_offset = results[0];
    let posting_list_offsets_offset = results[1];
    let total_counts_offset = results[2];
    let n = results[3];

    let fm_chunk_offsets: Vec<u64> = reader
        .read_range_and_decompress(fm_chunk_offsets_offset, posting_list_offsets_offset)
        .await?;
    let posting_list_offsets: Vec<u64> = reader
        .read_range_and_decompress(posting_list_offsets_offset, total_counts_offset)
        .await?;
    let cumulative_counts: Vec<u64> = reader
        .read_range_and_decompress(total_counts_offset, (file_size - 32) as u64)
        .await?;

    let mut query_set = JoinSet::new();

    for query in queries {
        let fm_chunk_offsets = fm_chunk_offsets.clone();
        let cumulative_counts = cumulative_counts.clone();
        let posting_list_offsets = posting_list_offsets.clone();
        let mut reader = reader.clone();

        query_set.spawn(async move {
            process_substring_query::<T>(
                query,
                n,
                &fm_chunk_offsets,
                &cumulative_counts,
                &posting_list_offsets,
                &mut reader,
                file_id,
            )
            .await
        });
    }

    let mut res = Vec::new();
    while let Some(query_res) = query_set.join_next().await {
        res.extend(query_res.unwrap());
    }
    Ok(res)
}

pub async fn _search_lava_substring_char(
    files: Vec<String>,
    query: String,
    k: usize,
    reader_type: ReaderType,
    token_viable_limit: Option<usize>,
    sample_factor: Option<usize>,
    wavelet_tree: bool,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let lower: String = query.chars().flat_map(|c| c.to_lowercase()).collect();
    let result: Vec<u8> = lower
        .chars()
        .filter(|id| !SKIP.chars().contains(id))
        .map(|c| c as u8)
        .collect();

    let mut query: Vec<Vec<u8>> = if let Some(sample_factor) = sample_factor {
        (0..sample_factor)
            .map(|offset| {
                result
                    .iter()
                    .skip(offset)
                    .step_by(sample_factor)
                    .cloned()
                    .collect::<Vec<u8>>()
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
    search_generic_async(
        file_sizes,
        readers,
        if wavelet_tree {
            QueryParam::SubstringCharWavelet(query)
        } else {
            QueryParam::SubstringChar(query)
        },
        k,
    )
    .await
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
