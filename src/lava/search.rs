use crate::lava::constants::*;
use crate::lava::fm_chunk::FMChunk;
use crate::lava::plist::PListChunk;
use crate::lava::wavelet_tree::search_wavelet_tree_from_reader;
use crate::{
    formats::readers::{
        get_file_size_and_reader, get_file_sizes_and_readers, get_reader, get_readers, AsyncReader,
        ClonableAsyncReader, ReaderType,
    },
    lava::error::LavaError,
};
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use itertools::Itertools;
use ndarray::{concatenate, stack, Array1, Array2, Axis};
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
use zstd::stream::read::Decoder;

use super::trie::FastTrie;
use super::wavelet_tree;
use futures::stream::{FuturesUnordered, StreamExt};
use std::cmp::Ordering;
use std::io::{self, Cursor};

enum QueryParam {
    SubstringCharWavelet(Vec<Vec<u8>>),
    SubstringChar(Vec<Vec<u8>>),
    Substring(Vec<Vec<u32>>),
    Uuid(String),
}
use std::fmt::Debug;
async fn get_tokenizer_async(mut readers: Vec<AsyncReader>) -> Result<(Tokenizer, Vec<String>), LavaError> {
    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    for i in 0..readers.len() {
        // now interpret this as a usize
        // readers[i].seek(SeekFrom::Start(0)).await?;
        let compressed_tokenizer_size = readers[i].read_usize_from_start(0, 1).await?[0];
        let this_compressed_tokenizer: bytes::Bytes = readers[i].read_range(8, 8 + compressed_tokenizer_size).await?;
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

use num_traits::{AsPrimitive, PrimInt, Unsigned};
use serde::{Deserialize, Serialize};
use std::ops::Add;

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
            + FMChunk::<T>::new(start_chunk).unwrap().search(current_token, start % FM_CHUNK_TOKS).unwrap() as usize;
        end = cumulative_counts[current_token.as_()] as usize
            + FMChunk::<T>::new(end_chunk).unwrap().search(current_token, end % FM_CHUNK_TOKS).unwrap() as usize;

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
        let this_chunk =
            plist_chunks[(this_start - start_offset) as usize..(this_end - start_offset) as usize].to_vec();

        chunk_set.spawn(async move {
            let mut decompressor = Decoder::new(&this_chunk[..]).unwrap();
            let mut serialized_plist_chunk: Vec<u8> = Vec::with_capacity(this_chunk.len());
            decompressor.read_to_end(&mut serialized_plist_chunk).unwrap();
            let plist_chunk: Vec<u64> = bincode::deserialize(&serialized_plist_chunk).unwrap();

            let chunk_res: Vec<(u64, u64)> = if i == 0 {
                if total_chunks == 1 {
                    plist_chunk[start % FM_CHUNK_TOKS..end % FM_CHUNK_TOKS].iter().map(|&uid| (file_id, uid)).collect()
                } else {
                    plist_chunk[start % FM_CHUNK_TOKS..].iter().map(|&uid| (file_id, uid)).collect()
                }
            } else if i == total_chunks - 1 {
                plist_chunk[..end % FM_CHUNK_TOKS].iter().map(|&uid| (file_id, uid)).collect()
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

async fn read_and_decompress<T>(reader: &mut AsyncReader, start: u64, size: u64) -> Result<T, LavaError>
where
    T: DeserializeOwned,
{
    let compressed = reader.read_range(start, start + size).await?;
    let mut decompressor = Decoder::new(&compressed[..]).unwrap();
    let mut decompressed = Vec::new();
    std::io::copy(&mut decompressor, &mut decompressed)?;
    let result: T = bincode::deserialize(&decompressed)?;
    Ok(result)
}

async fn search_substring_wavelet(
    file_id: u64,
    mut reader: AsyncReader,
    file_size: usize,
    queries: Vec<Vec<u8>>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    println!("{:?}", queries);

    let metadata_start = reader.read_usize_from_end(1).await?[0];

    let metadata: (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, usize) =
        read_and_decompress(&mut reader, metadata_start as u64, file_size as u64 - metadata_start - 8).await.unwrap();
    let (offsets, level_offsets, posting_list_offsets, cumulative_counts, n) = metadata;

    // let mut query_set = JoinSet::new();

    let mut res: Vec<(u64, u64)> = vec![];

    for query in queries {
        let mut reader = reader.clone();
        let (start, end) =
            search_wavelet_tree_from_reader(&mut reader, &query, n, &offsets, &level_offsets, &cumulative_counts)
                .await?;

        println!("{} {}", start, end);

        let start_offset = posting_list_offsets[start / FM_CHUNK_TOKS];
        let end_offset = posting_list_offsets[end / FM_CHUNK_TOKS + 1];
        let total_chunks = end / FM_CHUNK_TOKS - start / FM_CHUNK_TOKS + 1;

        let plist_chunks = reader.read_range(start_offset as u64, end_offset as u64).await.unwrap();

        let mut chunk_set = JoinSet::new();

        for i in 0..total_chunks {
            let this_start = posting_list_offsets[start / FM_CHUNK_TOKS + i];
            let this_end = posting_list_offsets[start / FM_CHUNK_TOKS + i + 1];
            let this_chunk =
                plist_chunks[(this_start - start_offset) as usize..(this_end - start_offset) as usize].to_vec();

            chunk_set.spawn(async move {
                let mut decompressor = Decoder::new(&this_chunk[..]).unwrap();
                let mut serialized_plist_chunk: Vec<u8> = Vec::with_capacity(this_chunk.len());
                decompressor.read_to_end(&mut serialized_plist_chunk).unwrap();
                let plist_chunk: Vec<u64> = bincode::deserialize(&serialized_plist_chunk).unwrap();

                let chunk_res: Vec<(u64, u64)> = if i == 0 {
                    if total_chunks == 1 {
                        plist_chunk[start % FM_CHUNK_TOKS..end % FM_CHUNK_TOKS]
                            .iter()
                            .map(|&uid| (file_id, uid))
                            .collect()
                    } else {
                        plist_chunk[start % FM_CHUNK_TOKS..].iter().map(|&uid| (file_id, uid)).collect()
                    }
                } else if i == total_chunks - 1 {
                    plist_chunk[..end % FM_CHUNK_TOKS].iter().map(|&uid| (file_id, uid)).collect()
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

    let fm_chunk_offsets: Vec<u64> =
        reader.read_range_and_decompress(fm_chunk_offsets_offset, posting_list_offsets_offset).await?;
    let posting_list_offsets: Vec<u64> =
        reader.read_range_and_decompress(posting_list_offsets_offset, total_counts_offset).await?;
    let cumulative_counts: Vec<u64> =
        reader.read_range_and_decompress(total_counts_offset, (file_size - 32) as u64).await?;

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

async fn search_uuid_one_file(
    file_id: u64,
    mut reader: AsyncReader,
    file_size: usize,
    query: String,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let mut result: Vec<(u64, u64)> = Vec::new();
    let mut start_time = Instant::now();

    let this_result: Vec<usize> = FastTrie::query_with_reader(file_size, &mut reader, &query).await?;
    result.extend(this_result.iter().map(|x| (file_id, *x as u64)));

    // println!(
    //     "search_uuid_one_file: {}ms",
    //     start_time.elapsed().as_millis()
    // );

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
                join_set.spawn(search_substring_one_file::<u32>(file_id as u64, reader, file_size, value.clone()));
            }
            QueryParam::SubstringChar(ref value) => {
                join_set.spawn(search_substring_one_file::<u8>(file_id as u64, reader, file_size, value.clone()));
            }
            QueryParam::SubstringCharWavelet(ref value) => {
                join_set.spawn(search_substring_wavelet(file_id as u64, reader, file_size, value.clone()));
            }
            QueryParam::Uuid(ref value) => {
                join_set.spawn(search_uuid_one_file(file_id as u64, reader, file_size, value.clone()));
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

async fn search_bm25_async(
    file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query_tokens: Vec<u32>,
    query_weights: Vec<f32>,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let mut idf: HashMap<u32, f32> = HashMap::new();
    let mut total_token_counts: HashMap<u32, usize> = HashMap::new();
    for token in query_tokens.iter() {
        total_token_counts.insert(*token, 0);
    }
    let mut total_documents: usize = 0;
    let mut all_plist_offsets: Vec<Vec<u64>> = Vec::new();
    let mut chunks_to_search: HashMap<(usize, usize), Vec<(u32, u64)>> = HashMap::new();

    for i in 0..readers.len() {
        let results = readers[i].read_usize_from_end(3).await?;
        let compressed_term_dictionary_offset = results[0];
        let compressed_plist_offsets_offset = results[1];
        let num_documents = results[2];

        // now read the term dictionary
        let token_counts = readers[i]
            .read_range_and_decompress(compressed_term_dictionary_offset, compressed_plist_offsets_offset)
            .await?;

        for query_token in query_tokens.iter() {
            total_token_counts
                .insert(*query_token, total_token_counts[query_token] + token_counts[*query_token as usize] as usize);
        }
        total_documents += num_documents as usize;

        let plist_offsets =
            readers[i].read_range_and_decompress(compressed_plist_offsets_offset, file_sizes[i] as u64 - 24).await?;

        if plist_offsets.len() % 2 != 0 {
            let err = LavaError::Parse("data corruption".to_string());
            return Err(err);
        }

        let num_chunks: usize = plist_offsets.len() / 2;
        let term_dict_len: &[u64] = &plist_offsets[num_chunks..];

        for token in query_tokens.iter() {
            let tok = *token as u64;
            let (idx, offset) = match term_dict_len.binary_search(&tok) {
                Ok(idx) => (idx, 0),
                Err(idx) => (idx - 1, tok - term_dict_len[idx - 1]),
            };

            chunks_to_search.entry((i as usize, idx)).or_insert_with(Vec::new).push((*token, offset as u64));
        }

        all_plist_offsets.push(plist_offsets);
    }

    // compute the weighted IDF for each query token
    for (i, query_token) in query_tokens.iter().enumerate() {
        let query_weight = query_weights[i];
        let query_token = *query_token;
        let token_count = total_token_counts[&query_token];
        idf.insert(
            query_token,
            query_weight
                * ((total_documents as f32 - token_count as f32 + 0.5) / (token_count as f32 + 0.5) + 1.0).ln(),
        );
    }

    let mut plist_result: Vec<(u64, u64)> = Vec::new();
    let mut page_scores: HashMap<(u64, u64), f32> = HashMap::new();

    let mut join_set: JoinSet<Result<Vec<(usize, u64, u32, u64)>, LavaError>> = JoinSet::new();
    // need to parallelize this @Rain.
    for (file_id, chunk_id, tokens, offsets) in
        chunks_to_search.into_iter().map(|((file_id, chunk_id), token_offsets)| {
            let (tokens, offsets): (Vec<u32>, Vec<u64>) = token_offsets.into_iter().unzip();
            (file_id, chunk_id, Arc::new(tokens), Arc::new(offsets))
        })
    {
        let reader_type = match readers[file_id].reader {
            ClonableAsyncReader::AwsSdk(_) => ReaderType::AwsSdk,
            ClonableAsyncReader::Http(_) => ReaderType::Http,
            ClonableAsyncReader::Local(_) => ReaderType::Local,
        };

        let mut reader = match reader_type {
            ReaderType::AwsSdk | ReaderType::Http => readers[file_id].clone(),
            ReaderType::Local => {
                get_file_size_and_reader(readers[file_id].filename.clone(), reader_type).await.unwrap().1
            }
        };
        let start = all_plist_offsets[file_id][chunk_id];
        let end = all_plist_offsets[file_id][chunk_id + 1];
        let tokens = tokens.clone();
        let offsets = offsets.clone();

        join_set.spawn(async move {
            // println!("file_id: {}, chunk_id: {}", file_id, chunk_id);
            let buffer3 = reader.read_range(start, end).await?;

            // get all the second item in the offsets into its own vector

            let results: Vec<Vec<u64>> = PListChunk::search_compressed(buffer3.to_vec(), offsets.as_ref())?;

            let mut res = vec![];
            for (i, result) in results.iter().enumerate() {
                let token = &tokens[i];
                assert_eq!(result.len() % 2, 0);
                for i in (0..result.len()).step_by(2) {
                    let uid = result[i];
                    let page_score = result[i + 1];
                    res.push((file_id, uid, *token, page_score));
                }
            }
            Ok(res)
        });
    }

    while let Some(res) = join_set.join_next().await {
        let res = res.map_err(|e| LavaError::Parse(format!("join error: {:?}", e)))??;
        for (file_id, uid, token, page_score) in res {
            page_scores
                .entry((file_id as u64, uid))
                .and_modify(|e| *e += idf[&token] * page_score as f32)
                .or_insert(idf[&token] * page_score as f32);
        }
    }

    // sort the page scores by descending order
    let mut page_scores_vec: Vec<((u64, u64), f32)> = page_scores.into_iter().collect();
    page_scores_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // get the top k results
    for (uid, _score) in page_scores_vec.iter().take(k) {
        // println!("{}", score);
        plist_result.push(*uid);
    }

    Ok(plist_result)
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
        skip_tokens.extend(tokenizer.encode(char_str.clone(), false).unwrap().get_ids().to_vec());
        skip_tokens.extend(tokenizer.encode(format!(" {}", char_str), false).unwrap().get_ids().to_vec());
        skip_tokens.extend(tokenizer.encode(format!("{} ", char_str), false).unwrap().get_ids().to_vec());
    }

    let lower: String = query.chars().flat_map(|c| c.to_lowercase()).collect();
    let encoding = tokenizer.encode(lower, false).unwrap();
    let result: Vec<u32> = encoding.get_ids().iter().filter(|id| !skip_tokens.contains(id)).cloned().collect();

    let mut query: Vec<Vec<u32>> = if let Some(sample_factor) = sample_factor {
        (0..sample_factor)
            .map(|offset| result.iter().skip(offset).step_by(sample_factor).cloned().collect::<Vec<u32>>())
            .filter(|vec| !vec.is_empty())
            .collect()
    } else {
        vec![result]
    };

    println!("query {:?}", query);

    // query = [i[-token_viable_limit:] for i in query]
    if let Some(token_viable_limit) = token_viable_limit {
        query.iter_mut().for_each(|vec| {
            if vec.len() > token_viable_limit {
                *vec = vec.iter().rev().take(token_viable_limit).rev().cloned().collect();
            }
        });
    }

    println!("query {:?}", query);

    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    search_generic_async(file_sizes, readers, QueryParam::Substring(query), k).await
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
    let result: Vec<u8> = lower.chars().filter(|id| !SKIP.chars().contains(id)).map(|c| c as u8).collect();

    let mut query: Vec<Vec<u8>> = if let Some(sample_factor) = sample_factor {
        (0..sample_factor)
            .map(|offset| result.iter().skip(offset).step_by(sample_factor).cloned().collect::<Vec<u8>>())
            .filter(|vec| !vec.is_empty())
            .collect()
    } else {
        vec![result]
    };

    println!("query {:?}", query);

    // query = [i[-token_viable_limit:] for i in query]
    if let Some(token_viable_limit) = token_viable_limit {
        query.iter_mut().for_each(|vec| {
            if vec.len() > token_viable_limit {
                *vec = vec.iter().rev().take(token_viable_limit).rev().cloned().collect();
            }
        });
    }

    println!("query {:?}", query);

    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    search_generic_async(
        file_sizes,
        readers,
        if wavelet_tree { QueryParam::SubstringCharWavelet(query) } else { QueryParam::SubstringChar(query) },
        k,
    )
    .await
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
    _search_lava_substring_char(files, query, k, reader_type, token_viable_limit, sample_factor, false).await
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let mut vec = Vec::with_capacity(bytes.len() / 4);
    let mut i = 0;
    while i < bytes.len() {
        let value = LittleEndian::read_f32(&bytes[i..i + 4]);
        vec.push(value);
        i += 4;
    }
    vec
}

pub fn search_lava_vector(
    files: Vec<String>,
    query: Vec<f32>,
    nprobes: usize,
    reader_type: ReaderType,
) -> Result<(Vec<usize>, Vec<Array1<u8>>, Vec<(usize, Array1<u8>)>), LavaError> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();

    let res = rt.block_on(search_lava_vector_async(files, query, nprobes, reader_type));
    rt.shutdown_background();
    res
}

pub async fn search_lava_vector_async(
    files: Vec<String>,
    query: Vec<f32>,
    nprobes: usize,
    reader_type: ReaderType,
) -> Result<(Vec<usize>, Vec<Array1<u8>>, Vec<(usize, Array1<u8>)>), LavaError> {
    let start = Instant::now();

    let (_, mut readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;

    let mut futures = Vec::new();

    for _ in 0..readers.len() {
        let mut reader = readers.remove(0);

        futures.push(tokio::spawn(async move {
            let results = reader.read_usize_from_end(4).await.unwrap();

            let centroid_vectors_compressed_bytes = reader.read_range(results[2], results[3]).await.unwrap();

            // decompress them
            let mut decompressor = Decoder::new(centroid_vectors_compressed_bytes.as_ref()).unwrap();
            let mut centroid_vectors: Vec<u8> = Vec::with_capacity(centroid_vectors_compressed_bytes.len() as usize);
            decompressor.read_to_end(&mut centroid_vectors).unwrap();

            let centroid_vectors = bytes_to_f32_vec(&centroid_vectors);
            let num_vectors = centroid_vectors.len() / 128;
            let array2 = Array2::<f32>::from_shape_vec((num_vectors, 128), centroid_vectors).unwrap();

            (num_vectors, array2)
        }));
    }

    let result: Vec<Result<(usize, Array2<f32>), tokio::task::JoinError>> = futures::future::join_all(futures).await;

    let end = Instant::now();
    println!("Time stage 1 read: {:?}", end - start);

    let start = Instant::now();

    let arr_lens = result.iter().map(|x| x.as_ref().unwrap().0).collect::<Vec<_>>();
    // get cumulative arr len starting from 0
    let cumsum = arr_lens
        .iter()
        .scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    let arrays: Vec<Array2<f32>> = result.into_iter().map(|x| x.unwrap().1).collect();
    let centroids =
        concatenate(Axis(0), arrays.iter().map(|array| array.view()).collect::<Vec<_>>().as_slice()).unwrap();
    let query = Array1::<f32>::from_vec(query);
    let query_broadcast = query.broadcast(centroids.dim()).unwrap();

    let difference = &centroids - &query_broadcast;
    let norms = difference.map_axis(Axis(1), |row| row.dot(&row).sqrt());
    let mut indices_and_values: Vec<(usize, f32)> = norms.iter().enumerate().map(|(idx, &val)| (idx, val)).collect();

    indices_and_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let smallest_indices: Vec<usize> = indices_and_values.iter().map(|&(idx, _)| idx).take(nprobes).collect();

    let mut file_indices: Vec<Vec<usize>> = vec![vec![]; files.len()];
    for idx in smallest_indices.iter() {
        // figure out which file idx based on cumsum. need to find the index of the thing that is just bigger than idx

        let file_idx = cumsum.iter().enumerate().find(|(_, &val)| val > *idx).unwrap().0;
        let last_cumsum = if file_idx == 0 { 0 } else { cumsum[file_idx - 1] };
        let remainder = idx - last_cumsum;
        file_indices[file_idx].push(remainder);
    }

    let end = Instant::now();
    println!("Time math: {:?}", end - start);

    let start = Instant::now();

    let (_, mut readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;

    let mut file_ids = vec![];
    let mut futures = Vec::new();

    for file_id in 0..readers.len() {
        let mut reader = readers.remove(0);
        if file_indices[file_id].len() == 0 {
            continue;
        }
        let my_idx: Vec<usize> = file_indices[file_id].clone();
        file_ids.push(file_id);

        futures.push(tokio::spawn(async move {
            let results = reader.read_usize_from_end(4).await.unwrap();

            let pq_bytes = reader.read_range(results[0], results[1]).await.unwrap();

            let compressed_centroid_offset_bytes = reader.read_range(results[1], results[2]).await.unwrap();
            let mut decompressor = Decoder::new(compressed_centroid_offset_bytes.as_ref()).unwrap();
            let mut centroid_offsets_bytes: Vec<u8> =
                Vec::with_capacity(compressed_centroid_offset_bytes.len() as usize);
            decompressor.read_to_end(&mut centroid_offsets_bytes).unwrap();

            // now reinterpret centroid_offsets_bytes as a Vec<u64>

            let mut centroid_offsets = Vec::with_capacity(centroid_offsets_bytes.len() / 8);
            let mut cursor = Cursor::new(centroid_offsets_bytes);

            while cursor.position() < cursor.get_ref().len() as u64 {
                let value = cursor.read_u64::<LittleEndian>().unwrap();
                centroid_offsets.push(value);
            }

            let mut this_result: Vec<(usize, u64, u64)> = vec![];

            for idx in my_idx.iter() {
                this_result.push((file_id, centroid_offsets[*idx], centroid_offsets[*idx + 1]));
            }
            (this_result, Array1::<u8>::from_vec(pq_bytes.to_vec()))
        }));
    }

    let result: Vec<Result<(Vec<(usize, u64, u64)>, Array1<u8>), tokio::task::JoinError>> =
        futures::future::join_all(futures).await;
    let result: Vec<(Vec<(usize, u64, u64)>, Array1<u8>)> = result.into_iter().map(|x| x.unwrap()).collect();

    let pq_bytes: Vec<Array1<u8>> = result.iter().map(|x| x.1.clone()).collect::<Vec<_>>();

    let end = Instant::now();
    println!("Time stage 2 read: {:?}", end - start);

    let start = Instant::now();
    let reader = get_reader(files[file_ids[0]].clone(), reader_type.clone()).await.unwrap();

    let mut futures = FuturesUnordered::new();
    for i in 0..result.len() {
        let to_read = result[i].0.clone();
        for (file_id, start, end) in to_read.into_iter() {
            let mut reader_c = reader.clone();
            reader_c.update_filename(files[file_id].clone()).unwrap();

            futures.push(tokio::spawn(async move {
                let start_time = Instant::now();
                let codes_and_plist = reader_c.read_range(start, end).await.unwrap();
                // println!(
                //     "Time to read {:?}, {:?}",
                //     Instant::now() - start_time,
                //     codes_and_plist.len()
                // );
                (file_id, Array1::<u8>::from_vec(codes_and_plist.to_vec()))
            }));
        }
    }

    let mut ranges: Vec<(usize, Array1<u8>)> = vec![];

    while let Some(x) = futures.next().await {
        ranges.push(x.unwrap());
    }

    let end = Instant::now();
    println!("Time stage 3 read: {:?}", end - start);

    Ok((file_ids, pq_bytes, ranges))
}

#[tokio::main]
pub async fn get_tokenizer_vocab(files: Vec<String>, reader_type: ReaderType) -> Result<Vec<String>, LavaError> {
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

        let res =
            search_lava_bm25(vec![file.to_string()], vec![6300, 15050], vec![0.1, 0.2], 10, ReaderType::default())
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
