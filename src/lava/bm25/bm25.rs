use crate::formats::readers::{
    get_file_size_and_reader, get_file_sizes_and_readers, AsyncReader, ClonableAsyncReader,
    ReaderType,
};
use crate::lava::error::LavaError;
use crate::lava::plist::PListChunk;
use arrow::array::{make_array, Array, ArrayData, LargeStringArray, UInt64Array};
use bincode;
use tokio::task::JoinSet;

use std::collections::{BTreeMap, HashMap};

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use tokenizers::parallelism::MaybeParallelIterator;
use zstd::stream::encode_all;
use zstd::stream::Decoder;

use super::super::get_tokenizer;

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
) -> Result<Vec<(usize, usize)>, LavaError> {
    // if k1 and b are not provided, set them to default value
    let k1: f32 = k1.unwrap_or(1.2);
    let b: f32 = b.unwrap_or(0.75);

    let array = make_array(array);
    // let uid = make_array(ArrayData::from_pyarrow(uid)?);
    let uid = make_array(uid);
    let array: &arrow_array::GenericByteArray<arrow_array::types::GenericStringType<i64>> = array
        .as_any()
        .downcast_ref::<LargeStringArray>()
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

    let (tokenizer, compressed_tokenizer) = get_tokenizer(tokenizer_file)?;
    let vocab_size: usize = tokenizer.get_vocab_size(false);

    let mut texts = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        let text = array.value(i);
        texts.push(text);
    }

    let encodings = texts
        .into_maybe_par_iter()
        .map(|text| {
            let encoding = tokenizer.encode(text, false).unwrap();
            encoding.get_ids().to_vec()
        })
        .collect::<Vec<Vec<u32>>>();

    let mut inverted_index: Vec<BTreeMap<usize, f32>> = vec![BTreeMap::new(); vocab_size];
    let mut token_counts: Vec<usize> = vec![0; vocab_size];

    let mut avg_len: f32 = 0.0;
    for encoding in encodings.iter() {
        avg_len += encoding.len() as f32;
    }
    avg_len /= encodings.len() as f32;

    for (i, encoding) in encodings.iter().enumerate() {
        let this_uid = uid.value(i) as usize;
        let mut local_token_counts: BTreeMap<u32, usize> = BTreeMap::new();
        for key in encoding {
            *local_token_counts.entry(*key).or_insert(0) += 1;
        }
        for key in local_token_counts.keys() {
            let local_count = local_token_counts[key];
            let local_factor: f32 = (local_count as f32) * (k1 + 1.0)
                / (local_count as f32 + k1 * (1.0 - b + b * encoding.len() as f32 / avg_len));

            inverted_index[*key as usize]
                .entry(this_uid)
                .and_modify(|e| *e = (*e).max(local_factor))
                .or_insert(local_factor);

            token_counts[*key as usize] += 1;
        }
    }

    let mut file = File::create(output_file_name)?;
    file.write_all(&(compressed_tokenizer.len() as u64).to_le_bytes())?;
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

    let cache_end = file.seek(SeekFrom::Current(0))? as usize;

    Ok(vec![(compressed_term_dict_offset as usize, cache_end)])
}

struct PListChunkIterator {
    reader: AsyncReader,
    current_offset_in_chunk: usize,
    current_chunk_offset: usize,
    current_chunk: Vec<Vec<u64>>,
    plist_offsets: Vec<u64>,
    plist_elems: Vec<u64>,
}

impl PListChunkIterator {
    // take ownership of the data structures
    pub async fn new(
        mut reader: AsyncReader,
        plist_offsets: Vec<u64>,
        plist_elems: Vec<u64>,
    ) -> Result<Self, LavaError> {
        // read the first chunk

        let buffer3 = reader
            .read_range(plist_offsets[0], plist_offsets[1])
            .await?;
        let result: Vec<Vec<u64>> =
            PListChunk::search_compressed(buffer3.to_vec(), &(0..plist_elems[1]).collect())
                .unwrap();

        Ok(Self {
            reader: reader,
            current_offset_in_chunk: 0,
            current_chunk_offset: 0,
            current_chunk: result,
            plist_offsets: plist_offsets,
            plist_elems: plist_elems,
        })
    }

    pub fn get(&mut self) -> Vec<u64> {
        self.current_chunk[self.current_offset_in_chunk as usize].clone()
    }

    pub async fn advance(&mut self) -> Result<(), LavaError> {
        self.current_offset_in_chunk += 1;
        if self.current_offset_in_chunk == self.current_chunk.len() {
            // read the next chunk
            self.current_offset_in_chunk = 0;
            self.current_chunk_offset += 1;
            if self.current_chunk_offset + 2 > self.plist_offsets.len() {
                return Err(LavaError::Parse("out of chunks".to_string()));
            }

            let buffer3 = self
                .reader
                .read_range(
                    self.plist_offsets[self.current_chunk_offset],
                    self.plist_offsets[self.current_chunk_offset + 1],
                )
                .await?;

            self.current_chunk = PListChunk::search_compressed(
                buffer3.to_vec(),
                &(0..(self.plist_elems[self.current_chunk_offset + 1]
                    - self.plist_elems[self.current_chunk_offset]))
                    .collect(),
            )
            .unwrap();
        }

        Ok(())
    }
}

pub(crate) async fn merge_lava_bm25(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
    reader_type: ReaderType,
) -> Result<Vec<(usize, usize)>, LavaError> {
    // let mut builder = Fs::default();
    // let current_path = env::current_dir()?;
    // builder.root(current_path.to_str().expect("no path"));
    // let operator = Operator::new(builder)?.finish();

    let mut file_sizes: Vec<u64> = Vec::with_capacity(lava_files.len());
    let mut plist_chunk_iterators: Vec<PListChunkIterator> = Vec::with_capacity(lava_files.len());

    let mut combined_token_counts: Vec<usize> = Vec::new();
    let mut total_num_documents: u64 = 0;
    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    for file in lava_files {
        let reader_type = reader_type.clone();
        let (file_size, mut reader) = get_file_size_and_reader(file, reader_type).await?;
        let file_size = file_size as u64;

        let results = reader.read_usize_from_end(3).await?;
        let compressed_term_dict_offset = results[0];
        let compressed_plist_offsets_offset = results[1];
        let num_documents = results[2];
        total_num_documents += num_documents;

        let compressed_token_counts = reader
            .read_range(compressed_term_dict_offset, compressed_plist_offsets_offset)
            .await?;

        let mut decompressed_token_counts: Vec<u8> = Vec::new();
        let mut decompressor: Decoder<'_, BufReader<&[u8]>> =
            Decoder::new(&compressed_token_counts[..])?;
        decompressor.read_to_end(&mut decompressed_token_counts)?;
        let token_counts: Vec<usize> = bincode::deserialize(&decompressed_token_counts)?;

        if combined_token_counts.len() == 0 {
            combined_token_counts = token_counts;
        } else {
            // add token_counts to combined_token_counts
            for (i, count) in token_counts.iter().enumerate() {
                combined_token_counts[i] += count;
            }
        }

        let buffer2 = reader
            .read_range(compressed_plist_offsets_offset, file_size - 24)
            .await?;

        decompressor = Decoder::new(&buffer2[..])?;
        let mut decompressed_serialized_plist_offsets: Vec<u8> =
            Vec::with_capacity(buffer2.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist_offsets)?;
        let this_plist_offsets: Vec<u64> =
            bincode::deserialize(&decompressed_serialized_plist_offsets)?;

        if (this_plist_offsets.len() % 2) != 0 {
            let err = LavaError::Parse("data corruption".to_string());
            return Err(err);
        }
        let num_elements = this_plist_offsets.len() / 2;

        let compressed_tokenizer_size = reader.read_usize_from_start(0, 1).await?[0];
        let this_compressed_tokenizer: bytes::Bytes =
            reader.read_range(8, 8 + compressed_tokenizer_size).await?;

        match &compressed_tokenizer {
            Some(value) => assert!(
                this_compressed_tokenizer == value,
                "detected different tokenizers, cannot merge, something is very wrong."
            ),
            None => compressed_tokenizer = Some(this_compressed_tokenizer.to_vec()),
        }

        file_sizes.push(file_size);
        plist_chunk_iterators.push(
            PListChunkIterator::new(
                reader,
                this_plist_offsets[..num_elements].to_vec(),
                this_plist_offsets[num_elements..].to_vec(),
            )
            .await?,
        );
    }

    let mut output_file = File::create(condensed_lava_file)?;

    let compressed_tokenizer = compressed_tokenizer.unwrap();
    // let compressed_tokenizer_len = compressed_tokenizer.len();
    output_file.write_all(&(compressed_tokenizer.len() as u64).to_le_bytes())?;
    output_file.write_all(&compressed_tokenizer)?;

    let mut new_plist_offsets: Vec<u64> = vec![output_file.seek(SeekFrom::Current(0))?];
    let mut new_plist_elems: Vec<u64> = vec![0];
    let mut plist_chunk = PListChunk::new()?;
    let mut counter: u64 = 0;

    for tok in 0..combined_token_counts.len() {
        // Find the smallest current line

        let mut plist: Vec<u64> = vec![];

        for i in 0..plist_chunk_iterators.len() {
            let this_plist: Vec<u64> = plist_chunk_iterators[i].get();
            assert_eq!(this_plist.len() % 2, 0);

            for (j, item) in this_plist.iter().enumerate() {
                if j % 2 == 0 {
                    // page offset
                    plist.push(*item + uid_offsets[i]);
                } else {
                    // quantized score
                    plist.push(*item);
                }
            }

            // this will return error for the last one, but it's ok
            let _ = plist_chunk_iterators[i].advance().await;
        }

        counter += 1;

        let plist = Vec::from_iter(plist.into_iter());
        let written = plist_chunk.add_plist(&plist)?;
        if written > 1024 * 1024 || tok == combined_token_counts.len() - 1 {
            let bytes = plist_chunk.finalize_compression()?;
            let this_len: u64 = bytes.len() as u64;

            output_file.write(&bytes)?;
            new_plist_offsets.push(new_plist_offsets[new_plist_offsets.len() - 1] + this_len);
            new_plist_elems.push(counter);
            plist_chunk = PListChunk::new()?;
        }
    }

    new_plist_offsets.append(&mut new_plist_elems);

    let bytes = bincode::serialize(&combined_token_counts)?;
    let compressed_token_counts = encode_all(&bytes[..], 0).expect("Compression failed");

    let compressed_term_dict_offset = output_file.seek(SeekFrom::Current(0))?;
    output_file.write(&compressed_token_counts)?;

    let serialized = bincode::serialize(&new_plist_offsets).unwrap();
    let compressed_plist_offsets =
        encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");

    let compressed_plist_offsets_offset =
        compressed_term_dict_offset + compressed_token_counts.len() as u64;
    output_file.write(&compressed_plist_offsets)?;

    output_file.write(&(compressed_term_dict_offset as u64).to_le_bytes())?;
    output_file.write(&(compressed_plist_offsets_offset as u64).to_le_bytes())?;
    output_file.write(&(total_num_documents as u64).to_le_bytes())?;

    Ok(vec![(
        compressed_term_dict_offset as usize,
        output_file.seek(SeekFrom::Current(0))? as usize,
    )])
}

pub(crate) async fn search_bm25_async(
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
            ClonableAsyncReader::Local(_) => ReaderType::Local,
        };

        let mut reader = match reader_type {
            ReaderType::AwsSdk | ReaderType::Http => readers[file_id].clone(),
            ReaderType::Local => {
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

#[cfg(test)]
mod tests {
    use crate::formats::readers::ReaderType;

    use super::search_lava_bm25;

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
}
