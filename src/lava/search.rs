use itertools::Itertools;
use ndarray::{concatenate, stack, Array1, Array2, Axis};
use rayon::result;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::{
    collections::{HashMap, HashSet},
    io::Read,
};
use tokio::task::JoinSet;
use zstd::stream::read::Decoder;
use crate::lava::trie::BinaryTrieNode;
use crate::lava::constants::*;
use crate::lava::fm_chunk::FMChunk;
use crate::lava::plist::PListChunk;
use crate::vamana::vamana::VectorAccessMethod;
use crate::vamana::{access::ReaderAccessMethodF32, access::InMemoryAccessMethodF32, EuclideanF32, IndexParams, VamanaIndex};
use crate::{
    formats::readers::{get_file_size_and_reader, get_file_sizes_and_readers, get_readers, get_reader, AsyncReader, ClonableAsyncReader, ReaderType},
    lava::error::LavaError,
};
use std::time::Instant;
use tokenizers::tokenizer::Tokenizer;
use std::collections::BTreeMap;
use ordered_float::OrderedFloat;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};

use super::trie::FastTrie;
use std::cmp::Ordering;
use std::io::{self, Cursor};

enum QueryParam {
    Substring(Vec<u32>),
    Uuid(String),
    Vector(Vec<f32>)
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
            Some(value) => assert!(this_compressed_tokenizer == value, "detected different tokenizers between different lava files, can't search across them."), 
            None => compressed_tokenizer = Some(this_compressed_tokenizer.to_vec())
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

async fn search_substring_one_file(
    file_id: u64,
    mut reader: AsyncReader,
    file_size: usize,
    query: Vec<u32>,
) -> Result<Vec<(u64, u64)>, LavaError> {
    // println!("executing on thread {:?}", std::thread::current().id());

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

    let mut start: usize = 0;
    let mut end: usize = n as usize;
    // let previous_range: u64 = u64::MAX;

    for i in (0..query.len()).rev() {
        let current_token = query[i];

        let start_byte = fm_chunk_offsets[start / FM_CHUNK_TOKS];
        let end_byte = fm_chunk_offsets[start / FM_CHUNK_TOKS + 1];
        let start_chunk = reader.read_range(start_byte, end_byte).await?;

        let start_byte = fm_chunk_offsets[end / FM_CHUNK_TOKS];
        let end_byte = fm_chunk_offsets[end / FM_CHUNK_TOKS + 1];
        let end_chunk = reader.read_range(start_byte, end_byte).await?;

        // read the first four bytes
        start = cumulative_counts[current_token as usize] as usize
            + FMChunk::new(start_chunk)?
                .search(current_token, start % FM_CHUNK_TOKS)
                .unwrap() as usize;
        end = cumulative_counts[current_token as usize] as usize
            + FMChunk::new(end_chunk)?
                .search(current_token, end % FM_CHUNK_TOKS)
                .unwrap() as usize;

        if start >= end {
            break;
        }
    }

    if start >= end {
        return Ok(Vec::new());
    }

    let start_offset = posting_list_offsets[start / FM_CHUNK_TOKS];
    let end_offset = posting_list_offsets[end / FM_CHUNK_TOKS + 1];
    let total_chunks = end / FM_CHUNK_TOKS - start / FM_CHUNK_TOKS + 1;

    // println!("total chunks: {}", total_chunks);

    let plist_chunks = reader.read_range(start_offset, end_offset).await?;
    let mut res = vec![];
    for i in 0..total_chunks {
        let this_start = posting_list_offsets[start / FM_CHUNK_TOKS + i];
        let this_end = posting_list_offsets[start / FM_CHUNK_TOKS + i + 1];
        let this_chunk =
            &plist_chunks[(this_start - start_offset) as usize..(this_end - start_offset) as usize];

        // decompress this chunk
        let mut decompressor = Decoder::new(&this_chunk[..])?;
        let mut serialized_plist_chunk: Vec<u8> = Vec::with_capacity(this_chunk.len() as usize);
        decompressor.read_to_end(&mut serialized_plist_chunk)?;
        let plist_chunk: Vec<u64> = bincode::deserialize(&serialized_plist_chunk)?;

        if i == 0 {
            if total_chunks == 1 {
                for uid in &plist_chunk[start % FM_CHUNK_TOKS..end % FM_CHUNK_TOKS] {
                    // println!("push file_id {}", file_id);
                    res.push((file_id as u64, *uid));
                }
            } else {
                for uid in &plist_chunk[start % FM_CHUNK_TOKS..] {
                    // println!("push file_id {}", file_id);
                    res.push((file_id as u64, *uid));
                }
            }
        } else if i == total_chunks - 1 {
            for uid in &plist_chunk[..end % FM_CHUNK_TOKS] {
                // println!("push file_id {}", file_id);
                res.push((file_id as u64, *uid));
            }
        } else {
            for uid in &plist_chunk[..] {
                // println!("push file_id {}", file_id);
                res.push((file_id as u64, *uid));
            }
        }
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
    let mut end_time = Instant::now();

    let this_result: Vec<usize> = FastTrie::query_with_reader(file_size, &mut reader, &query).await?;
    result.extend(this_result.iter().map(|x| (file_id , *x as u64)));
    
    Ok(result)
}

async fn search_generic_async(
    mut file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query: QueryParam,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {

    let mut join_set = JoinSet::new();

    for file_id in 0..readers.len() {
        let reader = readers.remove(0);
        let file_size = file_sizes.remove(0);

        match query {
            QueryParam::Substring(ref value) => {
                join_set.spawn(search_substring_one_file(
                    file_id as u64,
                    reader,
                    file_size,
                    value.clone(),
                ));
            },
            QueryParam::Uuid(ref value) => {
                join_set.spawn(search_uuid_one_file(
                    file_id as u64,
                    reader,
                    file_size,
                    value.clone(),
                ));
            },
            _ => panic!("invalid mode"),
        }
            
    }

    let mut result: BTreeSet<(u64, u64)> = BTreeSet::new();
    while let Some(res) = join_set.join_next().await {
        let res = res.unwrap().unwrap();
        result.extend(res);
        if result.len() >= k {
            break;
        }
    }

    join_set.shutdown().await;

    // keep only k elements in the result
    let mut result: Vec<(u64, u64)> = result.into_iter().collect_vec();
    result.truncate(k);
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
            .read_range_and_decompress(
                compressed_term_dictionary_offset,
                compressed_plist_offsets_offset,
            )
            .await?;

        for query_token in query_tokens.iter() {
            total_token_counts.insert(
                *query_token,
                total_token_counts[query_token] + token_counts[*query_token as usize] as usize,
            );
        }
        total_documents += num_documents as usize;

        let plist_offsets = readers[i]
            .read_range_and_decompress(compressed_plist_offsets_offset, file_sizes[i] as u64 - 24)
            .await?;

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

            chunks_to_search
                .entry((i as usize, idx))
                .or_insert_with(Vec::new)
                .push((*token, offset as u64));
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
                * ((total_documents as f32 - token_count as f32 + 0.5)
                    / (token_count as f32 + 0.5)
                    + 1.0)
                    .ln(),
        );
    }

    let mut plist_result: Vec<(u64, u64)> = Vec::new();
    let mut page_scores: HashMap<(u64, u64), f32> = HashMap::new();

    let mut join_set: JoinSet<Result<Vec<(usize, u64, u32, u64)>, LavaError>> = JoinSet::new();
    // need to parallelize this @Rain.
    for (file_id, chunk_id, tokens, offsets) in
        chunks_to_search
            .into_iter()
            .map(|((file_id, chunk_id), token_offsets)| {
                let (tokens, offsets): (Vec<u32>, Vec<u64>) = token_offsets.into_iter().unzip();
                (file_id, chunk_id, Arc::new(tokens), Arc::new(offsets))
            })
    {
        let reader_type = match readers[file_id].reader {
            ClonableAsyncReader::AwsSdk(_) => ReaderType::AwsSdk,
            ClonableAsyncReader::Http(_) => ReaderType::Http,
            ClonableAsyncReader::Opendal(_) => ReaderType::Opendal,
        };

        let mut reader = match reader_type {
            ReaderType::AwsSdk | ReaderType::Http => readers[file_id].clone(),
            ReaderType::Opendal => {
                get_file_size_and_reader(readers[file_id].filename.clone(), reader_type)
                    .await
                    .unwrap()
                    .1
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

            let results: Vec<Vec<u64>> =
                PListChunk::search_compressed(buffer3.to_vec(), offsets.as_ref())?;

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

    // println!("{:?}", result);

    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    search_generic_async(file_sizes, readers, QueryParam::Substring(result), k).await
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

#[tokio::main]
pub async fn search_lava_vector(
    files: Vec<String>,
    query: &Vec<f32>,
    nprobes: usize,
    reader_type: ReaderType,
) -> Result<(Vec<usize>, Vec<Array1<u8>>, Vec<(usize, Array1<u8>)>), LavaError> {

    let start = Instant::now();

    let (_,  mut readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;

    let mut futures = Vec::new();

    for _ in 0..readers.len() {
        let mut reader = readers.remove(0);

        futures.push(tokio::spawn(async move {
            let results = reader.read_usize_from_end(4).await.unwrap();

            let centroid_vectors_compressed_bytes = reader
                .read_range(results[2], results[3])
                .await
                .unwrap();

            // decompress them
            let mut decompressor = Decoder::new(centroid_vectors_compressed_bytes.as_ref()).unwrap();
            let mut centroid_vectors: Vec<u8> = Vec::with_capacity(centroid_vectors_compressed_bytes.len() as usize);
            decompressor.read_to_end(&mut centroid_vectors).unwrap();

            let centroid_vectors = bytes_to_f32_vec(&centroid_vectors);
            let array2 = Array2::<f32>::from_shape_vec((1000,128), centroid_vectors).unwrap();

            (results, array2)
        }));
    }

    let result: Vec<Result<(Vec<u64>, Array2<f32>), tokio::task::JoinError>> = futures::future::join_all(futures).await;

    let end = Instant::now();
    println!("Time stage 1 read: {:?}", end - start);

    let start = Instant::now();

    let arrays: Vec<Array2<f32>> = result.into_iter().map(|x| x.unwrap().1).collect();
    let centroids = concatenate(Axis(0), arrays.iter().map(|array| array.view()).collect::<Vec<_>>().as_slice()).unwrap();
    let query = Array1::<f32>::from_vec(query.clone());
    let query_broadcast = query.broadcast(centroids.dim()).unwrap();

    let difference = &centroids - &query_broadcast;
    let norms = difference.map_axis(Axis(1), |row| row.dot(&row).sqrt());
    let mut indices_and_values: Vec<(usize, f32)> = norms.iter().enumerate().map(|(idx, &val)| (idx, val)).collect();

    indices_and_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let smallest_indices: Vec<usize> = indices_and_values.iter().map(|&(idx, _)| idx).take(nprobes).collect();

    let mut file_indices: Vec<Vec<usize>> = vec![vec![]; files.len()];
    for idx in smallest_indices.iter() {
        file_indices[*idx / 1000].push(*idx % 1000 as usize);
    }

    let end = Instant::now();
    println!("Time math: {:?}", end - start);


    let start = Instant::now();

    let (_,  mut readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;

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

            let pq_bytes = reader
                .read_range(results[0], results[1])
                .await
                .unwrap();
        
            let compressed_centroid_offset_bytes = reader
                .read_range(results[1], results[2])
                .await
                .unwrap();
            let mut decompressor = Decoder::new(compressed_centroid_offset_bytes.as_ref()).unwrap();
            let mut centroid_offsets_bytes: Vec<u8> = Vec::with_capacity(compressed_centroid_offset_bytes.len() as usize);
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

    let result: Vec<Result<(Vec<(usize, u64, u64)>, Array1<u8>),tokio::task::JoinError>> = futures::future::join_all(futures).await;
    let result: Vec<(Vec<(usize, u64, u64)>, Array1<u8>)> = result.into_iter().map(|x| x.unwrap()).collect();

    let pq_bytes: Vec<Array1<u8>> = result
        .iter()
        .map(|x| x.1.clone())
        .collect::<Vec<_>>();

    let end = Instant::now();
    println!("Time stage 2 read: {:?}", end - start);   

    let start = Instant::now();

    let mut readers_map: BTreeMap<usize, AsyncReader> = BTreeMap::new();
    for file_id in file_ids.iter() {
        let reader = get_reader(files[*file_id].clone(), reader_type.clone()).await.unwrap();
        readers_map.insert(*file_id, reader);
    }

    let mut futures = Vec::new();
    for i in 0 .. result.len() {
        let to_read = result[i].0.clone();
        for (file_id, start, end) in to_read.into_iter() {
            // let file_name = files[file_id].clone();
            // let my_reader_type = reader_type.clone();
            // let mut reader = get_reader(file_name, my_reader_type).await.unwrap();
            let start_time = Instant::now();
            let mut reader = readers_map.get_mut(&file_id).unwrap().clone();
            println!("Time to get reader {:?}", Instant::now() - start_time);
            futures.push(tokio::spawn(async move {
                
                let start_time = Instant::now();
                let codes_and_plist = reader.read_range(start, end).await.unwrap();
                println!("Time to read {:?}", Instant::now() - start_time);
                (file_id, Array1::<u8>::from_vec(codes_and_plist.to_vec()))
            }));
        }
    }

    let ranges: Vec<Result<(usize, Array1<u8>), tokio::task::JoinError>> = futures::future::join_all(futures).await;
    let ranges: Vec<(usize, Array1<u8>)> = ranges.into_iter().map(|x| x.unwrap()).collect();


    let end = Instant::now();
    println!("Time stage 3 read: {:?}", end - start);

    Ok((file_ids, pq_bytes, ranges))
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
        );
        println!("{:?}", result.unwrap());
    }
}
