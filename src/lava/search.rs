use itertools::Itertools;
use ndarray::Array2;
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
    formats::readers::{get_file_size_and_reader, get_file_sizes_and_readers, AsyncReader, ClonableAsyncReader, ReaderType},
    lava::error::LavaError,
};
use std::time::Instant;
use tokenizers::tokenizer::Tokenizer;

use ordered_float::OrderedFloat;

use super::trie::FastTrie;

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

async fn search_substring_async(
    mut file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query: Vec<u32>,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let mut join_set = JoinSet::new();

    for file_id in 0..readers.len() {
        let reader = readers.remove(0);
        let file_size = file_sizes.remove(0);
        let query_clone = query.clone();

        join_set.spawn(search_substring_one_file(
            file_id as u64,
            reader,
            file_size,
            query_clone,
        ));
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

async fn search_uuid_async(
    file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query: &str,
    k: usize
) -> Result<Vec<(u64, u64)>, LavaError> {

    let mut result: Vec<(u64, u64)> = Vec::new();
    let mut start_time = Instant::now();
    let mut end_time = Instant::now();
    for i in 0..readers.len() {

        let this_result: Vec<usize> = FastTrie::query_with_reader(file_sizes[i], &mut readers[i],  query).await?;

        result.extend(this_result.iter().map(|x| (i as u64, *x as u64)));
        if result.len() >= k {
            break;
        }

    }
    
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



async fn search_vector_mem_async(
    file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    array: Array2<f32>,
    queries: &Vec<Vec<f32>>,
    k: usize,
    reader_type: ReaderType,
) -> Result<Vec<Vec<usize>>, LavaError> {
    
    let mut reader_access_methods: Vec<InMemoryAccessMethodF32> = vec![];

    let mut indices: Vec<VamanaIndex<f32, EuclideanF32, _>> = vec![];

    for i in 0..readers.len() {
        let bytes = readers[i].read_range(0, file_sizes[i] as u64).await?;
        // let num_points = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        // let dim = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let start = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let compressed_nlist = &bytes[24..];

        let mut decompressor = Decoder::new(&compressed_nlist[..])?;
        let mut serialized_nlist: Vec<u8> = Vec::with_capacity(compressed_nlist.len() as usize);
        decompressor.read_to_end(&mut serialized_nlist)?;
        let nlist: Array2<usize> = bincode::deserialize(&serialized_nlist).unwrap();

        let reader_access_method = InMemoryAccessMethodF32 { data: array.clone() };
        reader_access_methods.push(InMemoryAccessMethodF32 { data: array.clone() });

        // we probably want to serialize and deserialize the indexparams too
        // upon merging if they are not the same throw an error

        let index: VamanaIndex<f32, EuclideanF32, _> = VamanaIndex::hydrate(
            reader_access_method,
            IndexParams {
                num_neighbors: 32,
                search_frontier_size: 32,
                pruning_threshold: 2.0,
            },
            nlist,
            start as usize,
        );

        indices.push(index);
    }


    let mut all_results: Vec<Vec<usize>> = vec![];
        

    for query in queries.iter() {

        let mut results: BTreeSet<(OrderedFloat<f32>, usize, usize)> = BTreeSet::new();

        for (i, index) in indices.iter().enumerate() {
            let mut ctx: crate::vamana::vamana::SearchContext = index.get_search_context();
            let _ = index
                .search(&mut ctx, query.as_slice(), reader_type.clone())
                .await;
            let local_results: Vec<(OrderedFloat<f32>, usize, usize)> = ctx
                .frontier
                .iter()
                .map(|(v, d)| (OrderedFloat(*d as f32), i, *v))
                .sorted_by_key(|k| k.0)
                .collect();
            results.extend(local_results);
        }
        let results: Vec<usize> = results
            .iter()
            .take(k)
            .cloned()
            .map(|(_v, _i, d)| d)
            .collect();

        all_results.push(results);
            
    }

    Ok(all_results)
}

async fn search_vector_async(
    column_name: &str,
    file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    uid_nrows: &Vec<Vec<usize>>,
    uid_to_metadatas: &Vec<Vec<(String, usize, usize, usize, usize)>>,
    query: &Vec<f32>,
    k: usize,
    reader_type: ReaderType,
) -> Result<(Vec<(usize, usize)>, Array2<f32>), LavaError> {
    let mut results: BTreeSet<(OrderedFloat<f32>, usize, usize)> = BTreeSet::new();
    let mut reader_access_methods: Vec<ReaderAccessMethodF32> = vec![];

    for i in 0..readers.len() {
        let bytes = readers[i].read_range(0, file_sizes[i] as u64).await?;
        let num_points = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let dim = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let start = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let compressed_nlist = &bytes[24..];

        let mut decompressor = Decoder::new(&compressed_nlist[..])?;
        let mut serialized_nlist: Vec<u8> = Vec::with_capacity(compressed_nlist.len() as usize);
        decompressor.read_to_end(&mut serialized_nlist)?;
        let nlist: Array2<usize> = bincode::deserialize(&serialized_nlist).unwrap();

        let reader_access_method = ReaderAccessMethodF32 {
            dim: dim as usize,
            num_points: num_points as usize,
            column_name: column_name.to_string(),
            uid_nrows: &uid_nrows[i],
            // uid to (file_path, row_group, page_offset, page_size, dict_page_size)
            uid_to_metadata: &uid_to_metadatas[i],
        };
        reader_access_methods.push(reader_access_method.clone());

        // we probably want to serialize and deserialize the indexparams too
        // upon merging if they are not the same throw an error

        let index: VamanaIndex<f32, EuclideanF32, _> = VamanaIndex::hydrate(
            reader_access_method,
            IndexParams {
                num_neighbors: 32,
                search_frontier_size: 32,
                pruning_threshold: 2.0,
            },
            nlist,
            start as usize,
        );

        let mut ctx = index.get_search_context();
        let _ = index
            .search(&mut ctx, query.as_slice(), reader_type.clone())
            .await;
        let local_results: Vec<(OrderedFloat<f32>, usize, usize)> = ctx
            .frontier
            .iter()
            .map(|(v, d)| (OrderedFloat(*d as f32), i, *v))
            .collect();

        results.extend(local_results);
    }

    let results: Vec<(usize, usize)> = results
        .iter()
        .take(k)
        .cloned()
        .map(|(_v, i, d)| (i, d))
        .collect();

    let futures: Vec<_> = results
        .iter()
        .map(|(file_id, n)| reader_access_methods[*file_id].get_vec(*n, reader_type.clone()))
        .collect();

    let vectors: Vec<Vec<f32>> = futures::future::join_all(futures).await;
    let rows = vectors.len();
    let cols = vectors[0].len();
    let vectors: Vec<f32> = vectors.into_iter().flatten().collect();
    let vectors = Array2::from_shape_vec((rows, cols), vectors).unwrap();

    Ok((results, vectors))
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
    search_uuid_async(file_sizes, readers, &query, k).await
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
    search_substring_async(file_sizes, readers, result, k).await
}

#[tokio::main]
pub async fn search_lava_vector(
    files: Vec<String>,
    column_name: &str,
    uid_nrows: &Vec<Vec<usize>>,
    uid_to_metadatas: &Vec<Vec<(String, usize, usize, usize, usize)>>,
    query: &Vec<f32>,
    k: usize,
    reader_type: ReaderType,
) -> Result<(Vec<(usize, usize)>, Array2<f32>), LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;
    search_vector_async(
        column_name,
        file_sizes,
        readers,
        uid_nrows,
        uid_to_metadatas,
        query,
        k,
        reader_type,
    )
    .await
}

#[tokio::main]
pub async fn search_lava_vector_mem(
    files: Vec<String>,
    array: Array2<f32>,
    queries: &Vec<Vec<f32>>,
    k: usize,
    reader_type: ReaderType,
) -> Result<Vec<Vec<usize>>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;

    search_vector_mem_async(
        file_sizes,
        readers,
        array,
        queries,
        k,
        reader_type
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
        );
        println!("{:?}", result.unwrap());
    }
}
