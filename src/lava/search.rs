use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Cursor, Read, SeekFrom},
};
use zstd::stream::read::Decoder;

use std::env;

use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::{formats::io::READER_BUFFER_SIZE, lava::plist::PListChunk};
use crate::{
    formats::io::{AsyncReader, FsBuilder, Operators, S3Builder},
    lava::error::LavaError,
};

async fn search_lava_async(
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
        let (compressed_term_dictionary_offset, compressed_plist_offsets_offset, num_documents) =
            readers[i].read_offsets().await?;
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

    for ((file_id, chunk_id), token_offsets) in chunks_to_search.into_iter() {
        println!("file_id: {}, chunk_id: {}", file_id, chunk_id);
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
        println!("{}", score);
        plist_result.push(*uid);
    }

    Ok(plist_result)
}

#[tokio::main]
pub async fn search_lava(files: Vec<String>, query_tokens: Vec<u32>, query_weights: Vec<f32>, k: usize ) -> Result<Vec<(u64,u64)>, LavaError> {
    
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
    search_lava_async(file_sizes, readers, query_tokens, query_weights, k).await
}

#[cfg(test)]
mod tests {
    use super::search_lava;

    #[test]
    pub fn test_search_lava() {
        let file = "bump2.lava";

        let res = search_lava(vec![file.to_string()], vec![6300,15050], vec![0.1,0.2], 10).unwrap();

        println!("{:?}", res);
    }
}
