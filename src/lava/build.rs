use arrow::array::{make_array, Array, ArrayData, LargeStringArray, UInt64Array};
use itertools::Itertools;
use rayon::collections::btree_map;
use serde_json;
use tokenizers::parallelism::MaybeParallelIterator;
use tokenizers::tokenizer::Tokenizer;

use bincode;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::lava::constants::*;
use crate::lava::error::LavaError;
use crate::lava::plist::PListChunk;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use zstd::stream::encode_all;

use crate::vamana::{build_index_par, IndexParams, VamanaIndex};
use crate::vamana::{EuclideanF32, InMemoryAccessMethodF32};
use ndarray::Array2;

use rayon::prelude::*;

fn get_tokenizer(tokenizer_file: Option<String>) -> Result<(Tokenizer, Vec<u8>), LavaError> {
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
    Ok((tokenizer, compressed_tokenizer))
}

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
pub async fn build_lava_kmer(
    output_file_name: String,
    array: ArrayData,
    uid: ArrayData,
    tokenizer_file: Option<String>,
) -> Result<(), LavaError> {
    let array = make_array(array);
    // let uid = make_array(ArrayData::from_pyarrow(uid)?);
    let uid = make_array(uid);

    // if the tokenizer file is provided, check if the file exists. If it does not exist, raise an Error
    let (tokenizer, compressed_tokenizer) = get_tokenizer(tokenizer_file)?;
    let vocab_size: usize = tokenizer.get_vocab_size(false);

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

    // get all trigrams.

    let mut trigrams_inverted_index: BTreeMap<(u32, u32, u32), BTreeSet<u64>> = BTreeMap::new();

    for (i, encoding) in encodings.iter().enumerate() {
        let this_uid = uid.value(i) as usize;
        for j in 0..(encoding.len() as i64 - 2) {
            let j = j as usize;
            // let trigram = (encoding[j], encoding[j + 1], encoding[j + 2]);
            let bigram = (u32::MAX, encoding[j], encoding[j + 1]);
            let unigram = (u32::MAX, u32::MAX, encoding[j]);
            // trigrams_inverted_index
            //     .entry(trigram)
            //     .or_insert_with(BTreeSet::new)
            //     .insert(this_uid as u64);
            trigrams_inverted_index
                .entry(bigram)
                .or_insert_with(BTreeSet::new)
                .insert(this_uid as u64);
            trigrams_inverted_index
                .entry(unigram)
                .or_insert_with(BTreeSet::new)
                .insert(this_uid as u64);
        }

        if encoding.len() >= 2 {
            let bigram = (
                u32::MAX,
                encoding[encoding.len() - 2],
                encoding[encoding.len() - 1],
            );
            trigrams_inverted_index
                .entry(bigram)
                .or_insert_with(BTreeSet::new)
                .insert(this_uid as u64);
            let unigram = (u32::MAX, u32::MAX, encoding[encoding.len() - 2]);
            trigrams_inverted_index
                .entry(unigram)
                .or_insert_with(BTreeSet::new)
                .insert(this_uid as u64);
        }

        if encoding.len() >= 1 {
            let unigram = (u32::MAX, u32::MAX, encoding[encoding.len() - 1]);
            trigrams_inverted_index
                .entry(unigram)
                .or_insert_with(BTreeSet::new)
                .insert(this_uid as u64);
        }
    }

    // figure out the average length of the inverted index posting lists
    // filter the trigrams to include only things where the value length is smaller than 10

    let trigrams_inverted_index = trigrams_inverted_index
        .into_iter()
        .filter(|(_, value)| value.len() < (uid.value(encodings.len() - 1) / 10 * 3) as usize)
        .collect::<Vec<_>>();

    let mut avg_len: f32 = 0.0;
    let mut all_lens: Vec<usize> = vec![];
    for (_, value) in trigrams_inverted_index.iter() {
        avg_len += value.len() as f32;
        all_lens.push(value.len());
    }
    //write out all_lens as a numpy file
    let mut file = File::create("all_lens.npy")?;
    for val in all_lens.iter() {
        file.write_all(&val.to_le_bytes())?;
    }

    avg_len /= trigrams_inverted_index.len() as f32;
    println!("Average length: {}", avg_len);

    let mut file = File::create(output_file_name)?;
    file.write_all(&(compressed_tokenizer.len() as u64).to_le_bytes())?;
    file.write_all(&compressed_tokenizer)?;

    // Handle the compressed data (for example, saving to a file or sending over a network)
    println!("Number of trigrams: {}", trigrams_inverted_index.len());

    let mut plist_offsets: Vec<u64> = vec![file.seek(SeekFrom::Current(0))?];
    let mut plist_elems: Vec<u64> = vec![0];
    let mut plist_chunk = PListChunk::new()?;
    let mut counter: u64 = 0;

    let mut term_dictionary: Vec<(u32, u32, u32)> = Vec::new();

    for (key, value) in trigrams_inverted_index.iter() {
        if value.len() < 5 {
            continue;
        }
        counter += 1;

        term_dictionary.push(*key);

        let written = plist_chunk.add_plist(&value.iter().map(|x| *x as u64).collect_vec())?;
        if written > 1024 * 1024 || counter == trigrams_inverted_index.len() as u64 {
            let bytes = plist_chunk.finalize_compression()?;
            file.write_all(&bytes)?;
            plist_offsets.push(plist_offsets[plist_offsets.len() - 1] + bytes.len() as u64);
            plist_elems.push(counter);
            plist_chunk = PListChunk::new()?;
        }
    }

    println!("{}", counter);

    let serialized_term_dictionary = bincode::serialize(&term_dictionary).unwrap();
    let compressed_term_dictionary = encode_all(&serialized_term_dictionary[..], 0)
        .expect("Compression of term dictionary failed");

    println!("{}", compressed_term_dictionary.len());

    plist_offsets.append(&mut plist_elems);
    let compressed_term_dict_offset = file.seek(SeekFrom::Current(0))?;
    file.write_all(&compressed_term_dictionary)?;

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
    tokenizer_file: Option<String>,
    token_skip_factor: Option<u32>,
) -> Result<(), LavaError> {
    let array = make_array(array);
    // let uid = make_array(ArrayData::from_pyarrow(uid)?);
    let uid = make_array(uid);

    let token_skip_factor = token_skip_factor.unwrap_or(1);

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

    let mut texts: Vec<(u64, &str)> = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        let text = array.value(i);
        texts.push((uid.value(i), text));
    }

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

    let named_encodings = texts
        .into_maybe_par_iter()
        .map(|(uid, text)| {
            // strip out things in skip in text

            let lower: String = text.chars().flat_map(|c| c.to_lowercase()).collect();
            let encoding = tokenizer.encode(lower, false).unwrap();
            let result: Vec<u32> = encoding
                .get_ids()
                .iter()
                .filter(|id| !skip_tokens.contains(id))
                .cloned()
                .collect();
            (vec![uid; result.len()], result)
        })
        .collect::<Vec<(Vec<u64>, Vec<u32>)>>();

    let uids: Vec<u64> = named_encodings
        .iter()
        .map(|(uid, _)| uid)
        .flatten()
        .cloned()
        .collect::<Vec<u64>>();
    let encodings: Vec<u32> = named_encodings
        .into_iter()
        .map(|(_, text)| text)
        .flatten()
        .collect::<Vec<u32>>();

    let mut suffices: Vec<Vec<u32>> = vec![];

    let (encodings, uids) = if token_skip_factor > 1 {
        let encodings: Vec<u32> = encodings
            .into_iter()
            .enumerate() // Enumerate to get the index and value
            .filter(|&(index, _)| index % token_skip_factor as usize == 1) // Keep only elements with odd indices (every second element)
            .map(|(_, value)| value) // Extract the value
            .collect(); // Collect into a vector

        let uids: Vec<u64> = uids
            .into_iter()
            .enumerate() // Enumerate to get the index and value
            .filter(|&(index, _)| index % token_skip_factor as usize == 1) // Keep only elements with odd indices (every second element)
            .map(|(_, value)| value) // Extract the value
            .collect();
        (encodings, uids)
    } else {
        (encodings, uids)
    };

    for i in 10..encodings.len() {
        suffices.push(encodings[i - 10..i].to_vec());
    }

    for i in encodings.len()..encodings.len() + 10 {
        let mut suffix = encodings[i - 10..encodings.len()].to_vec();
        suffix.append(&mut vec![0; i - encodings.len()]);
        suffices.push(suffix);
    }

    let mut sa: Vec<usize> = (0..suffices.len()).collect();

    // let start = std::time::Instant::now();

    sa.par_sort_by(|&a, &b| suffices[a].cmp(&suffices[b]));

    // let duration = start.elapsed();

    let mut idx: Vec<u64> = Vec::with_capacity(encodings.len());
    let mut bwt: Vec<u32> = Vec::with_capacity(encodings.len());
    for i in 0..sa.len() {
        if sa[i] == 0 {
            bwt.push(encodings[encodings.len() - 1]);
            idx.push(uids[uids.len() - 1]);
        } else {
            bwt.push(encodings[(sa[i] - 1) as usize]);
            idx.push(uids[(sa[i] - 1) as usize]);
        }
    }

    let mut file = File::create(output_file_name)?;
    file.write_all(&(compressed_tokenizer.len() as u64).to_le_bytes())?;
    file.write_all(&compressed_tokenizer)?;

    let mut fm_chunk_offsets: Vec<usize> = vec![file.seek(SeekFrom::Current(0))? as usize];

    let mut current_chunk: Vec<u32> = vec![];
    let mut current_chunk_counts: HashMap<u32, u64> = HashMap::new();
    let mut next_chunk_counts: HashMap<u32, u64> = HashMap::new();

    for i in 0..bwt.len() {
        let current_tok = bwt[i];
        next_chunk_counts
            .entry(current_tok)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        current_chunk.push(current_tok);

        if ((i + 1) % FM_CHUNK_TOKS == 0) || i == bwt.len() - 1 {
            let serialized_counts = bincode::serialize(&current_chunk_counts)?;
            let compressed_counts =
                encode_all(&serialized_counts[..], 10).expect("Compression failed");
            file.write_all(&(compressed_counts.len() as u64).to_le_bytes())?;
            file.write_all(&compressed_counts)?;
            let serialized_chunk = bincode::serialize(&current_chunk)?;
            let compressed_chunk =
                encode_all(&serialized_chunk[..], 10).expect("Compression failed");
            file.write_all(&compressed_chunk)?;
            fm_chunk_offsets.push(file.seek(SeekFrom::Current(0))? as usize);
            current_chunk_counts = next_chunk_counts.clone();
            current_chunk = vec![];
        }
    }
    // print out total file size so far
    println!("total file size: {}", file.seek(SeekFrom::Current(0))?);

    let mut cumulative_counts: Vec<u64> = vec![0];
    for i in 0..tokenizer.get_vocab_size(false) {
        cumulative_counts
            .push(cumulative_counts[i] + *current_chunk_counts.get(&(i as u32)).unwrap_or(&0));
    }

    let mut posting_list_offsets: Vec<usize> = vec![file.seek(SeekFrom::Current(0))? as usize];

    for i in (0..idx.len()).step_by(FM_CHUNK_TOKS) {
        let slice = &idx[i..std::cmp::min(idx.len(), i + FM_CHUNK_TOKS)];
        let serialized_slice = bincode::serialize(slice)?;
        let compressed_slice = encode_all(&serialized_slice[..], 0).expect("Compression failed");
        file.write_all(&compressed_slice)?;
        posting_list_offsets.push(file.seek(SeekFrom::Current(0))? as usize);
    }

    let fm_chunk_offsets_offset = file.seek(SeekFrom::Current(0))? as usize;
    let serialized_fm_chunk_offsets = bincode::serialize(&fm_chunk_offsets)?;
    let compressed_fm_chunk_offsets =
        encode_all(&serialized_fm_chunk_offsets[..], 0).expect("Compression failed");
    file.write_all(&compressed_fm_chunk_offsets)?;

    let posting_list_offsets_offset = file.seek(SeekFrom::Current(0))? as usize;
    let serialized_posting_list_offsets = bincode::serialize(&posting_list_offsets)?;
    let compressed_posting_list_offsets =
        encode_all(&serialized_posting_list_offsets[..], 0).expect("Compression failed");
    file.write_all(&compressed_posting_list_offsets)?;

    let total_counts_offset = file.seek(SeekFrom::Current(0))? as usize;
    let serialized_total_counts = bincode::serialize(&cumulative_counts)?;
    let compressed_total_counts: Vec<u8> =
        encode_all(&serialized_total_counts[..], 0).expect("Compression failed");
    file.write_all(&compressed_total_counts)?;

    file.write_all(&(fm_chunk_offsets_offset as u64).to_le_bytes())?;
    file.write_all(&(posting_list_offsets_offset as u64).to_le_bytes())?;
    file.write_all(&(total_counts_offset as u64).to_le_bytes())?;
    file.write_all(&(bwt.len() as u64).to_le_bytes())?;

    Ok(())
}

#[tokio::main]
pub async fn build_lava_vector(
    output_file_name: String,
    array: Array2<f32>,
    uid: ArrayData,
) -> Result<(), LavaError> {
    let uid = make_array(uid);

    let uid = uid
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(LavaError::Parse(
            "Expects uint64 array as second argument".to_string(),
        ))?;

    println!("{} {}", array.len(), uid.len());
    let dim = array.shape()[1];

    if array.shape()[0] != uid.len() {
        return Err(LavaError::Parse(
            "The length of the array and the uid array must be the same".to_string(),
        ));
    }

    let index: VamanaIndex<f32, EuclideanF32, _> = build_index_par::<f32, EuclideanF32, _>(
        InMemoryAccessMethodF32 { data: array },
        IndexParams {
            num_neighbors: 32,
            search_frontier_size: 32,
            pruning_threshold: 2.0,
        },
    );

    let num_points = index.num_points();
    let start = index.start;
    let nlist: ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<usize>,
        ndarray::prelude::Dim<[usize; 2]>,
    > = index.neighbors;

    // nlist.map_inplace(|elem| *elem /= 333);

    // let nlist: Vec<Vec<usize>> = nlist
    //     .outer_iter()
    //     .map(|row| {
    //         let mut unique_elems = row.to_vec();
    //         unique_elems.sort(); // Sort the elements to prepare for deduplication.
    //         unique_elems.dedup(); // Deduplicate.
    //         unique_elems
    //     })
    //     .collect();

    let bytes = bincode::serialize(&nlist)?;
    let compressed_nlist: Vec<u8> = encode_all(&bytes[..], 0).expect("Compression failed");

    // println!("{}", nlist);

    let mut file = File::create(output_file_name)?;
    file.write_all(&(num_points as u64).to_le_bytes())?;
    file.write_all(&(dim as u64).to_le_bytes())?;
    file.write_all(&(start as u64).to_le_bytes())?;
    file.write_all(&compressed_nlist)?;

    Ok(())
}
