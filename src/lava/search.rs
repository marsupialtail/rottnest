use std::{
    collections::{HashMap, HashSet},
    io::{BufRead, BufReader, Cursor, Read, SeekFrom},
};
use itertools::Itertools;
use zstd::stream::read::Decoder;
use bytes::Bytes;
use std::env;

use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokenizers::tokenizer::Tokenizer;
use crate::lava::constants::*;
use crate::{formats::io::READER_BUFFER_SIZE, lava::plist::PListChunk};
use crate::formats::parquet::read_indexed_pages;
use crate::{
    formats::io::{AsyncReader, FsBuilder, Operators, S3Builder},
    lava::error::LavaError,
};

async fn get_tokenizer_async(
    file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
) -> Result<(Tokenizer,Vec<String>), LavaError>
{
    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    for i in 0 .. readers.len() {
        let results = readers[i].read_usize_from_end(3).await?;
        let compressed_plist_offsets_offset = results[1];
        // now read the term dictionary
        readers[i].seek(SeekFrom::Start(compressed_plist_offsets_offset)).await?;
        let mut buffer2: Vec<u8> = vec![0u8; file_sizes[i] - compressed_plist_offsets_offset as usize - 24];
        readers[i].read(&mut buffer2).await?;
        let mut decompressor = Decoder::new(&buffer2[..])?;
        let mut decompressed_serialized_plist_offsets: Vec<u8> =
            Vec::with_capacity(buffer2.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist_offsets)?;
        let plist_offsets: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist_offsets)?;

        let this_compressed_tokenizer: bytes::Bytes = readers[i].read_range(0, plist_offsets[0]).await?;

        match &compressed_tokenizer {
            Some(value) => assert!(this_compressed_tokenizer == value, "detected different tokenizers between different lava files, can't search across them."), 
            None => compressed_tokenizer = Some(this_compressed_tokenizer.to_vec())
        }
    }

    let slice = &compressed_tokenizer.unwrap()[..];
    let mut decompressor = Decoder::new(slice)?;
    let mut decompressed_serialized_tokenizer: Vec<u8> =
        Vec::with_capacity(slice.len() as usize);
    decompressor.read_to_end(&mut decompressed_serialized_tokenizer)?;

    let mut result: Vec<String> = Vec::new();
    let tokenizer = Tokenizer::from_bytes(decompressed_serialized_tokenizer).unwrap();

    for i in 0 .. tokenizer.get_vocab_size(false) {
        let tok = tokenizer.decode(&vec![i as u32], false).unwrap();
        result.push(tok);
    }

    Ok((tokenizer,result))
        
}

async fn search_substring_async(
    file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query: Vec<u32>,
    k: usize
) -> Result<Vec<(u64,u64)>, LavaError> 
{

    fn search_chunk(chunk: Bytes, token: u32, pos: usize) -> Result<u64, LavaError> {
        let compressed_counts_size = u64::from_le_bytes(chunk[0 .. 8].try_into().unwrap());
        let compressed_counts = &chunk[8 .. (compressed_counts_size + 8) as usize];
        let mut decompressor = Decoder::new(compressed_counts)?;
        let mut serialized_counts: Vec<u8> = Vec::with_capacity(compressed_counts_size as usize);
        decompressor.read_to_end(&mut serialized_counts)?;
        let counts: HashMap<u32, u64> = bincode::deserialize(&serialized_counts)?;
        let compressed_fm_chunk = &chunk[(compressed_counts_size + 8) as usize ..];
        let mut decompressor = Decoder::new(compressed_fm_chunk)?;
        let mut serialized_fm_chunk: Vec<u8> = Vec::with_capacity(compressed_fm_chunk.len() as usize);
        decompressor.read_to_end(&mut serialized_fm_chunk)?;
        let fm_chunk: Vec<u32> = bincode::deserialize(&serialized_fm_chunk)?;

        let mut result = *counts.get(&token).unwrap_or(&0);
        for j in 0 .. pos {
            if fm_chunk[j] == token {
                result += 1;
            }
        }
        Ok(result)
    }

    let mut all_uids: HashSet<(u64,u64)> = HashSet::new();

    // @Rain can you please parallelize this.
    for file_id in 0 .. readers.len() {
        let results = readers[file_id].read_usize_from_end(4).await?;
        let fm_chunk_offsets_offset = results[0];
        let posting_list_offsets_offset = results[1];
        let total_counts_offset = results[2];
        let n = results[3];

        // now read the term dictionary
        readers[file_id].seek(SeekFrom::Start(fm_chunk_offsets_offset)).await?;
        let compressed_fm_chunk_offsets = readers[file_id].read_range(fm_chunk_offsets_offset, posting_list_offsets_offset).await?;
        let mut decompressor = Decoder::new(&compressed_fm_chunk_offsets[..])?;
        let mut serialized_fm_chunk_offsets: Vec<u8> = Vec::with_capacity(compressed_fm_chunk_offsets.len() as usize);
        decompressor.read_to_end(&mut serialized_fm_chunk_offsets)?;
        let fm_chunk_offsets: Vec<u64> = bincode::deserialize(&serialized_fm_chunk_offsets)?;

        // now read the posting list offsets
        readers[file_id].seek(SeekFrom::Start(posting_list_offsets_offset)).await?;
        let compressed_posting_list_offsets = readers[file_id].read_range(posting_list_offsets_offset, total_counts_offset).await?;
        let mut decompressor = Decoder::new(&compressed_posting_list_offsets[..])?;
        let mut serialized_posting_list_offsets: Vec<u8> = Vec::with_capacity(compressed_posting_list_offsets.len() as usize);
        decompressor.read_to_end(&mut serialized_posting_list_offsets)?;
        let posting_list_offsets: Vec<u64> = bincode::deserialize(&serialized_posting_list_offsets)?;

        // now read the total counts
        readers[file_id].seek(SeekFrom::Start(total_counts_offset)).await?;
        let compressed_total_counts = readers[file_id].read_range(total_counts_offset, (file_sizes[file_id] - 32) as u64).await?;
        let mut decompressor = Decoder::new(&compressed_total_counts[..])?;
        let mut serialized_total_counts: Vec<u8> = Vec::with_capacity(compressed_total_counts.len() as usize);
        decompressor.read_to_end(&mut serialized_total_counts)?;
        let total_counts: HashMap<u32, u64> = bincode::deserialize(&serialized_total_counts)?;

        let start: usize = 0;
        let end: usize = (n + 1) as usize;
        let previous_range = u64::MAX;

        for i in 1 .. query.len() {
            let current_token = query[query.len() - i];

            let start_byte = fm_chunk_offsets[start / FM_CHUNK_TOKS];
            let end_byte = fm_chunk_offsets[start / FM_CHUNK_TOKS + 1];
            let start_chunk = readers[file_id].read_range(start_byte, end_byte).await?;

            let start_byte = fm_chunk_offsets[end / FM_CHUNK_TOKS];
            let end_byte = fm_chunk_offsets[end / FM_CHUNK_TOKS + 1];
            let end_chunk = readers[file_id].read_range(start_byte, end_byte).await?;

            // read the first four bytes
            let start = total_counts[&current_token] + search_chunk(start_chunk, current_token, start % FM_CHUNK_TOKS)?;
            let end = total_counts[&current_token] + search_chunk(end_chunk, current_token, end % FM_CHUNK_TOKS)?;
            
            if start >= end {
                break;
            }
        }

        if start >= end {
            continue;
        } 

        let start_offset = posting_list_offsets[start / FM_CHUNK_TOKS];
        let end_offset = posting_list_offsets[end / FM_CHUNK_TOKS + 1];
        let total_chunks = end / FM_CHUNK_TOKS - start / FM_CHUNK_TOKS + 1;

        let plist_chunks = readers[file_id].read_range(start_offset, end_offset).await?;
        for i in 0 .. total_chunks {
            let this_start = posting_list_offsets[start / FM_CHUNK_TOKS + i];
            let this_end = posting_list_offsets[start / FM_CHUNK_TOKS + i + 1];
            let this_chunk = &plist_chunks[(this_start - start_offset) as usize .. (this_end - start_offset) as usize];

            // decompress this chunk
            let mut decompressor = Decoder::new(&this_chunk[..])?;
            let mut serialized_plist_chunk: Vec<u8> = Vec::with_capacity(this_chunk.len() as usize);
            decompressor.read_to_end(&mut serialized_plist_chunk)?;
            let plist_chunk: Vec<u64> = bincode::deserialize(&serialized_plist_chunk)?;
            
            if i == 0 {
                if total_chunks == 1 {
                    for uid in &plist_chunk[start % FM_CHUNK_TOKS .. end % FM_CHUNK_TOKS] {
                        all_uids.insert((file_id as u64,*uid));
                    }
                } else {
                    for uid in &plist_chunk[start % FM_CHUNK_TOKS .. ] {
                        all_uids.insert((file_id as u64,*uid));
                    }
                }
            } else if i == total_chunks - 1 { 
                println!("Warning");
                for uid in &plist_chunk[ .. end % FM_CHUNK_TOKS] {
                    all_uids.insert((file_id as u64,*uid));
                }
            } else {
                println!("Warning");
                for uid in &plist_chunk[..] {
                    all_uids.insert((file_id as u64,*uid));
                }
            }

            if all_uids.len() > k {
                break;
            }
        }
        if all_uids.len() > k {
            break;
        }
        
    }
    Ok(all_uids.into_iter().collect())
}


async fn search_bm25_async(
    file_sizes: Vec<usize>,
    mut readers: Vec<AsyncReader>,
    query_tokens: Vec<u32>,
    query_weights: Vec<f32>,
    k: usize,
) -> Result<Vec<(u64,u64)>, LavaError> 
{
    let mut idf: HashMap<u32, f32> = HashMap::new();
    let mut total_token_counts: HashMap<u32, usize> = HashMap::new();
    for token in query_tokens.iter() {
        total_token_counts.insert(*token, 0);
    }
    let mut total_documents: usize = 0;
    let mut all_plist_offsets: Vec<Vec<u64>> = Vec::new();
    let mut chunks_to_search: HashMap<(usize, usize), Vec<(u32,u64)>> = HashMap::new();
    
    for i in 0 .. readers.len() {
        let results = readers[i].read_usize_from_end(3).await?;
        let compressed_term_dictionary_offset = results[0];
        let compressed_plist_offsets_offset = results[1];
        let num_documents = results[2];

        // now read the term dictionary
        let compressed_term_dictionary = readers[i]
            .read_range(
                compressed_term_dictionary_offset,
                compressed_plist_offsets_offset,
            )
            .await?;

        let mut decompressed_token_counts: Vec<u8> = Vec::new();
        let mut decompressor: Decoder<'_, BufReader<&[u8]>> =
            Decoder::new(&compressed_term_dictionary[..])?;
        decompressor.read_to_end(&mut decompressed_token_counts)?;
        let token_counts: Vec<usize> = bincode::deserialize(&decompressed_token_counts)?;

        for query_token in query_tokens.iter() { 
            total_token_counts.insert(
                *query_token,
                total_token_counts[query_token] + token_counts[*query_token as usize],
            );
        }
        total_documents += num_documents as usize;

        readers[i].seek(SeekFrom::Start(compressed_plist_offsets_offset)).await?;
        let mut buffer2: Vec<u8> = vec![0u8; file_sizes[i] - compressed_plist_offsets_offset as usize - 24];
        readers[i].read(&mut buffer2).await?;
        decompressor = Decoder::new(&buffer2[..])?;
        let mut decompressed_serialized_plist_offsets: Vec<u8> =
            Vec::with_capacity(buffer2.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist_offsets)?;
        let plist_offsets: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist_offsets)?;

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
                .entry((i as usize,idx))
                .or_insert_with(Vec::new)
                .push((*token, offset as u64));
        }

        all_plist_offsets.push(plist_offsets);
    }

    // compute the weighted IDF for each query token
    for (i, query_token) in query_tokens.iter().enumerate() {
        let query_weight = query_weights[i];
        let query_token= *query_token;
        let token_count = total_token_counts[&query_token];
        idf.insert(
            query_token,
            query_weight * ( (total_documents as f32 - token_count as f32 + 0.5) / (token_count as f32 + 0.5) + 1.0).ln(),
        );
    }
    
    let mut plist_result: Vec<(u64,u64)> = Vec::new();
    let mut page_scores: HashMap<(u64,u64), f32> = HashMap::new();

    // need to parallelize this @Rain.
    for ((file_id, chunk_id), token_offsets) in chunks_to_search.into_iter() {
        // println!("file_id: {}, chunk_id: {}", file_id, chunk_id);
        let buffer3 = readers[file_id].read_range(
            all_plist_offsets[file_id][chunk_id],
            all_plist_offsets[file_id][chunk_id + 1],
            ).await?;
        
        // get all the second item in the offsets into its own vector
        let (tokens, offsets): (Vec<u32>, Vec<u64>) = token_offsets.into_iter().unzip();

        let results: Vec<Vec<u64>> = PListChunk::search_compressed(buffer3.to_vec(), offsets).unwrap();

        for (i, result) in results.iter().enumerate() {
            let token = &tokens[i];
            assert_eq!(result.len() % 2, 0);
            for i in (0..result.len()).step_by(2) {
                let uid = result[i];
                let page_score = result[i + 1];

                // page_scores[uid] += idf[token] * page_score;
                page_scores.entry((file_id as u64, uid))
                    .and_modify(|e| *e += idf[token] * page_score as f32)
                    .or_insert(idf[token] * page_score as f32);
                
            }
        }
    }

    // sort the page scores by descending order
    let mut page_scores_vec: Vec<((u64,u64), f32)> = page_scores.into_iter().collect();
    page_scores_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // get the top k results
    for (uid, score) in page_scores_vec.iter().take(k) {
        // println!("{}", score);
        plist_result.push(*uid);
    }

    Ok(plist_result)
}

async fn get_file_sizes_and_readers(files: &Vec<String>) -> Result<(Vec<usize>, Vec<AsyncReader>), LavaError> {
    let mut readers: Vec<AsyncReader> = Vec::new();
    let mut file_sizes: Vec<usize> = Vec::new();
    for file in files {
        let operator = if file.starts_with("s3://") {
            Operators::from(S3Builder::from(file.as_str())).into_inner()
        } else {
            let current_path = env::current_dir()?;
            Operators::from(FsBuilder::from(current_path.to_str().expect("no path"))).into_inner()
        };

        let filename = if file.starts_with("s3://") {
            file[5..].split("/").collect::<Vec<&str>>().join("/")
        } else {
            file.to_string()
        };
        let reader: AsyncReader = operator
            .clone()
            .reader_with(&file)
            .buffer(READER_BUFFER_SIZE)
            .await?
            .into();
        readers.push(reader);

        let file_size: u64 = operator.stat(&filename).await?.content_length();
        file_sizes.push(file_size as usize);
    }

    Ok((file_sizes, readers))
}

#[tokio::main]
pub async fn search_lava(files: Vec<String>, query_tokens: Vec<u32>, query_weights: Vec<f32>, k: usize ) -> Result<Vec<(u64,u64)>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    search_bm25_async(file_sizes, readers, query_tokens, query_weights, k).await
}

#[tokio::main]
pub async fn search_substring(files: Vec<String>, query: String, k: usize) -> Result<Vec<(u64,u64)>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    let tokenizer = get_tokenizer_async(file_sizes, readers).await?.0;

    let mut skip_tokens: HashSet<u32> = HashSet::new();
    for char in SKIP.chars() {
        let char_str = char.to_string(); 
        skip_tokens.extend(tokenizer.encode(char_str.clone(), false).unwrap().get_ids().to_vec());
        skip_tokens.extend(tokenizer.encode(format!(" {}", char_str), false).unwrap().get_ids().to_vec());
        skip_tokens.extend(tokenizer.encode(format!("{} ", char_str), false).unwrap().get_ids().to_vec());
    }


    let encoding = tokenizer.encode(query, false).unwrap();
    let result: Vec<u32> = encoding.get_ids().iter()
        .filter(|id| !skip_tokens.contains(id))
        .cloned()
        .collect();

    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    search_substring_async(file_sizes, readers, result, k).await
}


#[tokio::main]
pub async fn get_tokenizer_vocab(files: Vec<String>) -> Result<Vec<String>, LavaError> {
    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    Ok(get_tokenizer_async(file_sizes, readers).await?.1)
}

#[cfg(test)]
mod tests {
    use super::search_lava;

    #[test]
    pub fn test_search_lava_one() {
        let file = "condensed.lava";

        let res = search_lava(vec![file.to_string()], vec![6300,15050], vec![0.1,0.2], 10).unwrap();

        println!("{:?}", res);
    }

    #[test]
    pub fn test_search_lava_two() {

        let res = search_lava(vec!["bump1.lava".to_string(), "bump2.lava".to_string()], vec![6300,15050], vec![0.1,0.2], 10).unwrap();

        println!("{:?}", res);
    }
}
