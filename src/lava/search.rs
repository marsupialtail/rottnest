use bytes::Bytes;
use futures::{FutureExt, SinkExt};
use core::num;
use itertools::Itertools;
use std::env;
use std::{
    collections::{HashMap, HashSet},
    io::{BufRead, BufReader, Cursor, Read, SeekFrom},
};
use zstd::stream::read::Decoder;

use crate::formats::parquet::read_indexed_pages;
use crate::lava::constants::*;
use crate::lava::fm_chunk::FMChunk;
use crate::{formats::io::READER_BUFFER_SIZE, lava::plist::PListChunk};
use crate::{
    formats::io::{AsyncReader, FsBuilder, Operators, S3Builder},
    lava::error::LavaError,
};
use futures::future::{AbortHandle, Abortable, Aborted};
use std::sync::Arc;
use std::time::Duration;
use tokenizers::tokenizer::Tokenizer;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::{self, mpsc, Mutex};

async fn get_tokenizer_async(
    mut readers: Vec<AsyncReader>,
) -> Result<(Tokenizer, Vec<String>), LavaError> {
    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    for i in 0..readers.len() {
        // now interpret this as a usize
        readers[i].seek(SeekFrom::Start(0)).await?;
        let compressed_tokenizer_size = readers[i].read_u64_le().await?;
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
    //all_uids: Arc<Mutex<Vec<(u64, u64)>>>,
    mut tx: mpsc::UnboundedSender<Flow>,
) -> Result<(), LavaError> {
    println!("executing");

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
    let previous_range = u64::MAX;

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
        return Ok(());
    }

    let start_offset = posting_list_offsets[start / FM_CHUNK_TOKS];
    let end_offset = posting_list_offsets[end / FM_CHUNK_TOKS + 1];
    let total_chunks = end / FM_CHUNK_TOKS - start / FM_CHUNK_TOKS + 1;

    println!("{}", total_chunks);

    let plist_chunks = reader.read_range(start_offset, end_offset).await?;
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
                    // all_uids.lock().await.push((file_id as u64, *uid));
                    let _ = tx.send(Flow::Data((file_id as u64, *uid)));
                }
            } else {
                for uid in &plist_chunk[start % FM_CHUNK_TOKS..] {
                    // all_uids.lock().await.push((file_id as u64, *uid));
                    let _ = tx.send(Flow::Data((file_id as u64, *uid)));
                }
            }
        } else if i == total_chunks - 1 {
            println!("Warning");
            for uid in &plist_chunk[..end % FM_CHUNK_TOKS] {
                // all_uids.lock().await.push((file_id as u64, *uid));
                let _ = tx.send(Flow::Data((file_id as u64, *uid)));
            }
        } else {
            println!("Warning");
            for uid in &plist_chunk[..] {
                // all_uids.lock().await.push((file_id as u64, *uid));
                let _ = tx.send(Flow::Data((file_id as u64, *uid)));
            }
        }
    }
    Ok(())
}

enum Flow {
    End,
    Data((u64, u64)),
}

async fn search_substring_async(
    mut file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query: Vec<u32>,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let shared_list: Arc<Mutex<Vec<(u64, u64)>>> = Arc::new(Mutex::new(Vec::new()));
    // let mut abort_handles = Vec::new();
    let mut handles = Vec::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<Flow>();
    let (mut abort_tx, mut abort_rx) = sync::oneshot::channel::<()>();

    for file_id in 0..readers.len() {
        // let list_clone = shared_list.clone();

        // let (abort_handle, abort_registration) = AbortHandle::new_pair();

        let reader = readers.remove(0);
        let file_size = file_sizes.remove(0);
        let query_clone = query.clone();

        let tx = tx.clone();
        let handle = tokio::spawn(async move {
            search_substring_one_file(
                file_id as u64,
                reader,
                file_size,
                query_clone,
                tx,
            )
        });

        handles.push(handle);
    }

    let search_substring_async = async move {
        for handle in handles {
            if let Ok(()) = abort_rx.try_recv() {
                break;
            }
            let _ = handle.await.unwrap();
        }
        tx.send(Flow::End).unwrap();
    };

    let collect_result_async = async move {
        let mut res = Vec::new();
        while let Some(Flow::Data(uid)) = rx.recv().await {
            res.push(uid);
            if res.len() >= k {
                abort_tx.send(()).unwrap();
                break;
            }
        }
        res
    };

    let (_, result) = tokio::join!(search_substring_async, collect_result_async);

    // Monitor the shared list and abort tasks when target_length is reached
    // loop {
    //     let list = shared_list.lock().await;
    //     println!("{:?}", list);
    //     if list.len() >= k {
    //         println!("Target length reached, aborting tasks...");
    //         // Abort all tasks
    //         for abort_handle in abort_handles {
    //             abort_handle.abort();
    //         }
    //         break;
    //     }
    //     drop(list);
    //     tokio::time::sleep(Duration::from_millis(100)).await;
    // }

    // let result = shared_list.lock().await.clone();
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

    // need to parallelize this @Rain.
    for ((file_id, chunk_id), token_offsets) in chunks_to_search.into_iter() {
        // println!("file_id: {}, chunk_id: {}", file_id, chunk_id);
        let buffer3 = readers[file_id]
            .read_range(
                all_plist_offsets[file_id][chunk_id],
                all_plist_offsets[file_id][chunk_id + 1],
            )
            .await?;

        // get all the second item in the offsets into its own vector
        let (tokens, offsets): (Vec<u32>, Vec<u64>) = token_offsets.into_iter().unzip();

        let results: Vec<Vec<u64>> =
            PListChunk::search_compressed(buffer3.to_vec(), offsets).unwrap();

        for (i, result) in results.iter().enumerate() {
            let token = &tokens[i];
            assert_eq!(result.len() % 2, 0);
            for i in (0..result.len()).step_by(2) {
                let uid = result[i];
                let page_score = result[i + 1];

                // page_scores[uid] += idf[token] * page_score;
                page_scores
                    .entry((file_id as u64, uid))
                    .and_modify(|e| *e += idf[token] * page_score as f32)
                    .or_insert(idf[token] * page_score as f32);
            }
        }
    }

    // sort the page scores by descending order
    let mut page_scores_vec: Vec<((u64, u64), f32)> = page_scores.into_iter().collect();
    page_scores_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // get the top k results
    for (uid, score) in page_scores_vec.iter().take(k) {
        // println!("{}", score);
        plist_result.push(*uid);
    }

    Ok(plist_result)
}

async fn get_file_sizes_and_readers(
    files: &[String],
) -> Result<(Vec<usize>, Vec<AsyncReader>), LavaError> {
    let tasks: Vec<_> = files
        .iter()
        .map(|file| {
            let file = file.clone(); // Clone file name to move into the async block
            tokio::spawn(async move {
                // Determine the operator based on the file scheme
                let operator = if file.starts_with("s3://") {
                    Operators::from(S3Builder::from(file.as_str())).into_inner()
                } else {
                    let current_path = env::current_dir()?;
                    Operators::from(FsBuilder::from(current_path.to_str().expect("no path")))
                        .into_inner()
                };

                // Extract filename
                let filename = if file.starts_with("s3://") {
                    file[5..].split('/').collect::<Vec<_>>()[1..].join("/")
                } else {
                    file.clone()
                };

                // Create the reader
                let reader: AsyncReader = operator
                    .clone()
                    .reader_with(&filename)
                    .buffer(READER_BUFFER_SIZE)
                    .await?
                    .into();

                // Get the file size
                let file_size: u64 = operator.stat(&filename).await?.content_length();

                Ok::<_, LavaError>((file_size as usize, reader))
            })
        })
        .collect();

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;

    // Process results, separating out file sizes and readers
    let mut file_sizes = Vec::new();
    let mut readers = Vec::new();

    for result in results {
        match result {
            Ok(Ok((size, reader))) => {
                file_sizes.push(size);
                readers.push(reader);
            }
            Ok(Err(e)) => return Err(e), // Handle error from inner task
            Err(e) => return Err(LavaError::Parse("Task join error: {}".to_string())), // Handle join error
        }
    }

    Ok((file_sizes, readers))
}

#[tokio::main]
pub async fn search_lava_bm25(
    files: Vec<String>,
    query_tokens: Vec<u32>,
    query_weights: Vec<f32>,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    search_bm25_async(file_sizes, readers, query_tokens, query_weights, k).await
}

#[tokio::main]
pub async fn search_lava_substring(
    files: Vec<String>,
    query: String,
    k: usize,
) -> Result<Vec<(u64, u64)>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
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

    println!("{:?}", result);

    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    search_substring_async(file_sizes, readers, result, k).await
}

#[tokio::main]
pub async fn get_tokenizer_vocab(files: Vec<String>) -> Result<Vec<String>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    Ok(get_tokenizer_async(readers).await?.1)
}

#[cfg(test)]
mod tests {
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
        );
        println!("{:?}", result.unwrap());
    }
}
