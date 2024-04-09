use async_recursion::async_recursion;
use bincode;
use bit_vec::BitVec;
use itertools::Itertools;
use ndarray::Array2;
use opendal::raw::oio::ReadExt;
use opendal::services::Fs;
use opendal::{Operator, Writer};
use std::collections::BTreeSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use tokio::io::AsyncReadExt;
use zstd::bulk::compress;
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;

use crate::formats::io::{
    get_file_sizes_and_readers, AsyncReader, READER_BUFFER_SIZE, WRITER_BUFFER_SIZE,
};
use crate::lava::constants::*;
use crate::lava::error::LavaError;
use crate::lava::fm_chunk::FMChunk;
use crate::lava::plist::PListChunk;
use std::collections::HashMap;

use crate::vamana::{
    access::ReaderAccessMethodF32, merge_indexes_par, EuclideanF32, IndexParams, VamanaIndex,
};

// @Rain chore: we need to simplify all the iterator impls

struct PListIterator {
    reader: AsyncReader,
    plist_offsets: Vec<u64>,
    current_chunk_offset: usize,
    pub current_chunk: Vec<u64>,
}

impl PListIterator {
    // take ownership of the data structures
    pub async fn new(mut reader: AsyncReader, plist_offsets: Vec<u64>) -> Result<Self, LavaError> {
        let plist_chunk = reader
            .read_range_and_decompress(plist_offsets[0], plist_offsets[1])
            .await?;
        Ok(Self {
            reader: reader,
            plist_offsets: plist_offsets,
            current_chunk_offset: 0,
            current_chunk: plist_chunk,
        })
    }

    pub async fn advance(&mut self) -> Result<(), LavaError> {
        self.current_chunk_offset += 1;
        if self.current_chunk_offset + 2 > self.plist_offsets.len() {
            return Err(LavaError::Parse("out of chunks".to_string()));
        }
        self.current_chunk = self
            .reader
            .read_range_and_decompress(
                self.plist_offsets[self.current_chunk_offset],
                self.plist_offsets[self.current_chunk_offset + 1],
            )
            .await?;
        Ok(())
    }
}

struct FMChunkIterator {
    reader: AsyncReader,
    fm_chunk_offsets: Vec<u64>,
    current_chunk_offset: usize,
    pub current_chunk: FMChunk,
}

impl FMChunkIterator {
    // take ownership of the data structures
    pub async fn new(
        mut reader: AsyncReader,
        fm_chunk_offsets: Vec<u64>,
    ) -> Result<Self, LavaError> {
        let buffer3 = reader
            .read_range(fm_chunk_offsets[0], fm_chunk_offsets[1])
            .await?;
        let current_chunk = FMChunk::new(buffer3)?;

        Ok(Self {
            reader: reader,
            fm_chunk_offsets: fm_chunk_offsets,
            current_chunk_offset: 0,
            current_chunk: current_chunk,
        })
    }

    pub async fn advance(&mut self) -> Result<(), LavaError> {
        self.current_chunk_offset += 1;

        if self.current_chunk_offset + 2 > self.fm_chunk_offsets.len() {
            return Err(LavaError::Parse("out of chunks".to_string()));
        }
        let buffer3 = self
            .reader
            .read_range(
                self.fm_chunk_offsets[self.current_chunk_offset],
                self.fm_chunk_offsets[self.current_chunk_offset + 1],
            )
            .await?;
        self.current_chunk = FMChunk::new(buffer3)?;

        Ok(())
    }

    pub async fn reset(&mut self) -> Result<(), LavaError> {
        self.current_chunk = FMChunk::new(
            self.reader
                .read_range(self.fm_chunk_offsets[0], self.fm_chunk_offsets[1])
                .await?,
        )?;
        self.current_chunk_offset = 0;

        Ok(())
    }
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
            PListChunk::search_compressed(buffer3.to_vec(), (0..plist_elems[1]).collect()).unwrap();

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
                (0..(self.plist_elems[self.current_chunk_offset + 1]
                    - self.plist_elems[self.current_chunk_offset]))
                    .collect(),
            )
            .unwrap();
        }

        Ok(())
    }
}

async fn merge_lava_bm25(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> // hawaiian for lava condensation
{
    let mut builder = Fs::default();
    let current_path = env::current_dir()?;
    builder.root(current_path.to_str().expect("no path"));
    let operator = Operator::new(builder)?.finish();

    let mut file_sizes: Vec<u64> = Vec::with_capacity(lava_files.len());
    let mut plist_chunk_iterators: Vec<PListChunkIterator> = Vec::with_capacity(lava_files.len());

    let mut combined_token_counts: Vec<usize> = Vec::new();
    let mut total_num_documents: u64 = 0;
    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    for file in lava_files {
        let file = file.as_ref();
        let file_size: u64 = operator.stat(file).await?.content_length();
        let mut reader: AsyncReader = operator
            .clone()
            .reader_with(file)
            .buffer(READER_BUFFER_SIZE)
            .await?
            .into();

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

        reader.seek(SeekFrom::Start(0)).await?;
        let compressed_tokenizer_size = reader.read_u64_le().await?;
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
    let compressed_tokenizer_len = compressed_tokenizer.len();
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
    Ok(())
}

async fn compute_interleave(
    bwt0_reader: &mut FMChunkIterator,
    bwt1_reader: &mut FMChunkIterator,
    lens: (usize, usize),
    cumulative_counts: &Vec<u64>,
) -> Result<BitVec, LavaError> {
    let (bwt0_len, bwt1_len) = lens;

    let mut interleave = BitVec::from_elem(bwt0_len + bwt1_len, true);
    for i in 0..bwt0_len {
        interleave.set(i, false);
    }

    let mut interleave_iterations = 0;

    for it in 0..10 {
        let mut ind: [usize; 2] = [0, 0];

        let mut bwt0 = &bwt0_reader.current_chunk.bwt_chunk;
        let mut bwt1 = &bwt1_reader.current_chunk.bwt_chunk;

        let mut offsets = cumulative_counts.clone();
        let mut new_interleave = BitVec::from_elem(interleave.len(), false);
        for i in 0..interleave.len() {
            if interleave[i] {
                new_interleave.set(offsets[bwt1[ind[1]] as usize] as usize, true);
                offsets[bwt1[ind[1]] as usize] += 1;
                ind[1] += 1;

                if ind[1] == bwt1.len() {
                    // will return an Err for the last chunk, that's ok
                    let _ = bwt1_reader.advance().await;
                    bwt1 = &bwt1_reader.current_chunk.bwt_chunk;
                    ind[1] = 0;
                }
            } else {
                offsets[bwt0[ind[0]] as usize] += 1;
                ind[0] += 1;

                if ind[0] == bwt0.len() {
                    let _ = bwt0_reader.advance().await;
                    bwt0 = &bwt0_reader.current_chunk.bwt_chunk;
                    ind[0] = 0;
                }
            }
        }

        bwt0_reader.reset().await?;
        bwt1_reader.reset().await?;

        interleave_iterations += 1;
        println!(
            "{} {} ",
            interleave_iterations,
            interleave
                .iter()
                .zip(new_interleave.iter())
                .filter(|&(a_bit, b_bit)| a_bit != b_bit)
                .count()
        );

        if new_interleave == interleave {
            break;
        }
        interleave = new_interleave;
    }

    println!("interleave iterations: {}", interleave_iterations);
    Ok(interleave)
}

async fn merge_lava_substring(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> {
    // first merge the tokenizer, then merge the fm indices then merge the posting lists.
    let mut builder = Fs::default();
    let current_path = env::current_dir()?;
    builder.root(current_path.to_str().expect("no path"));
    let operator = Operator::new(builder)?.finish();

    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    // currently only support merging two files, but can support more in the future.
    assert_eq!(lava_files.len(), 2);
    assert_eq!(uid_offsets.len(), 2);

    let mut ns: Vec<u64> = vec![];
    let mut combined_cumulative_counts: Vec<u64> = vec![];
    let mut fm_chunk_iterators: Vec<FMChunkIterator> = vec![];
    let mut plist_iterators: Vec<PListIterator> = vec![];

    for file in lava_files {
        let file = file.as_ref();
        let file_size: u64 = operator.stat(file).await?.content_length();
        let mut reader: AsyncReader = operator
            .clone()
            .reader_with(file)
            .buffer(READER_BUFFER_SIZE)
            .await?
            .into();

        // @Rain just make two different readers for now because this is hopefully low overhead
        // instead of bothering with wrapping this thing in Arc<Mutex<>>. Lots of tech debt to clean up
        // needed for the FMChunkIterator and PListIterator

        let mut reader1: AsyncReader = operator
            .clone()
            .reader_with(file)
            .buffer(READER_BUFFER_SIZE)
            .await?
            .into();

        let results = reader.read_usize_from_end(4).await?;
        let fm_chunk_offsets_offset = results[0];
        let posting_list_offsets_offset = results[1];
        let total_counts_offset = results[2];
        let n = results[3];

        ns.push(n);

        reader.seek(SeekFrom::Start(0)).await?;
        let compressed_tokenizer_size = reader.read_u64_le().await?;
        let this_compressed_tokenizer: bytes::Bytes =
            reader.read_range(8, 8 + compressed_tokenizer_size).await?;

        match &compressed_tokenizer {
            Some(value) => assert!(
                this_compressed_tokenizer == value,
                "detected different tokenizers, cannot merge, something is very wrong."
            ),
            None => compressed_tokenizer = Some(this_compressed_tokenizer.to_vec()),
        }

        let fm_chunk_offsets: Vec<u64> = reader
            .read_range_and_decompress(fm_chunk_offsets_offset, posting_list_offsets_offset)
            .await?;
        let posting_list_offsets: Vec<u64> = reader
            .read_range_and_decompress(posting_list_offsets_offset, total_counts_offset)
            .await?;
        let cumulative_counts: Vec<u64> = reader
            .read_range_and_decompress(total_counts_offset, (file_size - 32) as u64)
            .await?;

        println!("{} {}", file, cumulative_counts.len());

        fm_chunk_iterators.push(FMChunkIterator::new(reader, fm_chunk_offsets).await?);
        plist_iterators.push(PListIterator::new(reader1, posting_list_offsets).await?);

        if combined_cumulative_counts.len() == 0 {
            combined_cumulative_counts = cumulative_counts;
        } else {
            // add cumulative_counts to combined_cumulative_counts
            for (i, count) in cumulative_counts.iter().enumerate() {
                combined_cumulative_counts[i] += count;
            }
        }
    }

    let mut bwt0_reader = fm_chunk_iterators.remove(0);
    let mut bwt1_reader = fm_chunk_iterators.remove(0);
    let mut plist0_reader = plist_iterators.remove(0);
    let mut plist1_reader = plist_iterators.remove(0);

    let start = std::time::Instant::now();
    let interleave: BitVec = compute_interleave(
        &mut bwt0_reader,
        &mut bwt1_reader,
        (ns[0] as usize, ns[1] as usize),
        &combined_cumulative_counts,
    )
    .await?;

    let _ = bwt0_reader.reset().await?;
    let _ = bwt1_reader.reset().await?;

    let duration = start.elapsed();
    println!("interleave time: {:?}", duration);

    let mut output_file = File::create(condensed_lava_file)?;
    let compressed_tokenizer = compressed_tokenizer.unwrap();
    output_file.write_all(&(compressed_tokenizer.len() as u64).to_le_bytes())?;
    output_file.write_all(&compressed_tokenizer)?;

    let mut bwt_output: Vec<u32> = Vec::with_capacity(interleave.len());
    let mut index_output: Vec<u64> = Vec::with_capacity(interleave.len());

    let mut bwt_ind0 = 0;
    let mut bwt_ind1 = 0;
    let mut idx_ind0 = 0;
    let mut idx_ind1 = 0;

    let mut bwt0 = &bwt0_reader.current_chunk.bwt_chunk;
    let mut bwt1 = &bwt1_reader.current_chunk.bwt_chunk;
    let mut idx0 = &plist0_reader.current_chunk;
    let mut idx1 = &plist1_reader.current_chunk;

    for i in 0..interleave.len() {
        if interleave[i] {
            bwt_output.push(bwt1[bwt_ind1]);
            index_output.push(idx1[idx_ind1] + uid_offsets[1]);

            bwt_ind1 += 1;
            if bwt_ind1 == bwt1.len() {
                let _ = bwt1_reader.advance().await;
                bwt1 = &bwt1_reader.current_chunk.bwt_chunk;
                bwt_ind1 = 0;
            }

            idx_ind1 += 1;
            if idx_ind1 == idx1.len() {
                let _ = plist1_reader.advance().await;
                idx1 = &plist1_reader.current_chunk;
                idx_ind1 = 0;
            }
        } else {
            bwt_output.push(bwt0[bwt_ind0]);
            index_output.push(idx0[idx_ind0] + uid_offsets[0]);

            bwt_ind0 += 1;
            if bwt_ind0 == bwt0.len() {
                let _ = bwt0_reader.advance().await;
                bwt0 = &bwt0_reader.current_chunk.bwt_chunk;
                bwt_ind0 = 0;
            }

            idx_ind0 += 1;
            if idx_ind0 == idx0.len() {
                let _ = plist0_reader.advance().await;
                idx0 = &plist0_reader.current_chunk;
                idx_ind0 = 0;
            }
        }
    }

    let mut current_chunk: Vec<u32> = vec![];
    let mut current_chunk_counts: HashMap<u32, u64> = HashMap::new();
    let mut next_chunk_counts: HashMap<u32, u64> = HashMap::new();
    let mut fm_chunk_offsets: Vec<usize> = vec![output_file.seek(SeekFrom::Current(0))? as usize];

    for i in 0..bwt_output.len() {
        let current_tok = bwt_output[i];
        next_chunk_counts
            .entry(current_tok)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        current_chunk.push(current_tok);

        if ((i + 1) % FM_CHUNK_TOKS == 0) || i == bwt_output.len() - 1 {
            let serialized_counts = bincode::serialize(&current_chunk_counts)?;
            let compressed_counts =
                encode_all(&serialized_counts[..], 0).expect("Compression failed");
            output_file.write_all(&(compressed_counts.len() as u64).to_le_bytes())?;
            output_file.write_all(&compressed_counts)?;
            let serialized_chunk = bincode::serialize(&current_chunk)?;
            let compressed_chunk =
                encode_all(&serialized_chunk[..], 0).expect("Compression failed");
            output_file.write_all(&compressed_chunk)?;
            fm_chunk_offsets.push(output_file.seek(SeekFrom::Current(0))? as usize);
            current_chunk_counts = next_chunk_counts.clone();
            current_chunk = vec![];
        }
    }

    let mut posting_list_offsets: Vec<usize> =
        vec![output_file.seek(SeekFrom::Current(0))? as usize];

    for i in (0..index_output.len()).step_by(FM_CHUNK_TOKS) {
        let slice = &index_output[i..std::cmp::min(index_output.len(), i + FM_CHUNK_TOKS)];
        let serialized_slice = bincode::serialize(slice)?;
        let compressed_slice = encode_all(&serialized_slice[..], 0).expect("Compression failed");
        output_file.write_all(&compressed_slice)?;
        posting_list_offsets.push(output_file.seek(SeekFrom::Current(0))? as usize);
    }

    let fm_chunk_offsets_offset = output_file.seek(SeekFrom::Current(0))? as usize;
    let serialized_fm_chunk_offsets = bincode::serialize(&fm_chunk_offsets)?;
    let compressed_fm_chunk_offsets =
        encode_all(&serialized_fm_chunk_offsets[..], 0).expect("Compression failed");
    output_file.write_all(&compressed_fm_chunk_offsets)?;

    let posting_list_offsets_offset = output_file.seek(SeekFrom::Current(0))? as usize;
    let serialized_posting_list_offsets = bincode::serialize(&posting_list_offsets)?;
    let compressed_posting_list_offsets =
        encode_all(&serialized_posting_list_offsets[..], 0).expect("Compression failed");
    output_file.write_all(&compressed_posting_list_offsets)?;

    let total_counts_offset = output_file.seek(SeekFrom::Current(0))? as usize;
    let serialized_total_counts = bincode::serialize(&combined_cumulative_counts)?;
    let compressed_total_counts: Vec<u8> =
        encode_all(&serialized_total_counts[..], 0).expect("Compression failed");
    output_file.write_all(&compressed_total_counts)?;

    output_file.write_all(&(fm_chunk_offsets_offset as u64).to_le_bytes())?;
    output_file.write_all(&(posting_list_offsets_offset as u64).to_le_bytes())?;
    output_file.write_all(&(total_counts_offset as u64).to_le_bytes())?;
    output_file.write_all(&(bwt_output.len() as u64).to_le_bytes())?;

    Ok(())
}

async fn merge_lava_vector(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    column_name: &str,
    uid_nrows: &Vec<Vec<usize>>,
    uid_to_metadatas: &Vec<Vec<(String, usize, usize, usize, usize)>>,
) -> Result<(), LavaError> {
    assert_eq!(lava_files.len(), 2);
    assert_eq!(uid_nrows.len(), 2);
    assert_eq!(uid_to_metadatas.len(), 2);

    let (file_sizes, mut readers) = get_file_sizes_and_readers(&lava_files).await?;

    let mut indices: Vec<VamanaIndex<f32, EuclideanF32, ReaderAccessMethodF32>> = vec![];

    let mut all_dim: Option<u64> = None;

    for i in 0..2 {
        let mut reader = readers.remove(0);
        let num_points = reader.read_u64_le().await?;
        let dim = reader.read_u64_le().await?;
        match all_dim {
            Some(d) => assert_eq!(dim, d),
            None => {
                all_dim.replace(dim);
            }
        }
        let start = reader.read_u64_le().await?;

        let compressed_nlist = reader.read_range(24, file_sizes[i] as u64).await?;
        let mut decompressor = Decoder::new(&compressed_nlist[..])?;
        let mut serialized_nlist: Vec<u8> = Vec::with_capacity(compressed_nlist.len() as usize);
        decompressor.read_to_end(&mut serialized_nlist)?;
        let nlist: Array2<usize> = bincode::deserialize(&serialized_nlist)?;

        let reader_access_method = ReaderAccessMethodF32 {
            dim: dim as usize,
            num_points: num_points as usize,
            column_name: column_name.to_string(),
            uid_nrows: &uid_nrows[i],
            // uid to (file_path, row_group, page_offset, page_size, dict_page_size)
            uid_to_metadata: &uid_to_metadatas[i],
        };
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

    let index0 = indices.remove(0);
    let index1 = indices.remove(0);
    let index = merge_indexes_par(index0, index1);

    let num_points = index.num_points();
    let start = index.start;
    let nlist = index.neighbors;
    let bytes = bincode::serialize(&nlist)?;
    let compressed_nlist: Vec<u8> = encode_all(&bytes[..], 0).expect("Compression failed");

    let mut file = File::create(condensed_lava_file)?;
    file.write_all(&(num_points as u64).to_le_bytes())?;
    file.write_all(&(all_dim.unwrap() as u64).to_le_bytes())?;
    file.write_all(&(start as u64).to_le_bytes())?;
    file.write_all(&compressed_nlist)?;

    Ok(())
}

#[async_recursion]
async fn async_parallel_merge_files(
    condensed_lava_file: String,
    files: Vec<String>,
    do_not_delete: BTreeSet<String>,
    uid_offsets: Vec<u64>,
    K: usize,
    mode: usize, // 0 for bm25 1 for substring
) -> Result<(), LavaError> {
    assert!(mode == 1 || mode == 0);
    if mode == 1 {
        assert_eq!(K, 2);
    }

    match files.len() {
        0 => Err(LavaError::Parse("out of chunks".to_string())), // Assuming LavaError can be constructed like this
        1 => {
            // the recursion will end here in this case. rename the files[0] to the supposed output name
            std::fs::rename(files[0].clone(), condensed_lava_file).unwrap();
            Ok(())
        }
        _ => {
            // More than one file, need to merge
            let mut tasks = vec![];
            let merged_files_shared = Arc::new(Mutex::new(vec![]));
            let new_uid_offsets_shared = Arc::new(Mutex::new(vec![]));

            let chunked_files: Vec<Vec<String>> = files
                .into_iter()
                .chunks(K)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            let chunked_uid_offsets: Vec<Vec<u64>> = uid_offsets
                .into_iter()
                .chunks(K)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            for (file_chunk, uid_chunk) in chunked_files
                .into_iter()
                .zip(chunked_uid_offsets.into_iter())
            {
                if file_chunk.len() == 1 {
                    // If there's an odd file out, directly move it to the next level
                    merged_files_shared
                        .lock()
                        .unwrap()
                        .push(file_chunk[0].clone());
                    new_uid_offsets_shared
                        .lock()
                        .unwrap()
                        .push(uid_chunk[0].clone());
                    continue;
                }

                let merged_files_clone = Arc::clone(&merged_files_shared);
                let new_uid_offsets_clone = Arc::clone(&new_uid_offsets_shared);
                let do_not_delete_clone = do_not_delete.clone();

                let task = tokio::spawn(async move {
                    let my_uuid = uuid::Uuid::new_v4();
                    let merged_filename = my_uuid.to_string(); // Define this function based on your requirements

                    println!("mergin {:?}", file_chunk);

                    if mode == 0 {
                        merge_lava_bm25(&merged_filename, file_chunk.to_vec(), uid_chunk.to_vec())
                            .await
                    } else {
                        merge_lava_substring(
                            &merged_filename,
                            file_chunk.to_vec(),
                            uid_chunk.to_vec(),
                        )
                        .await
                    }
                    .unwrap();

                    // now go delete the input files

                    for file in file_chunk {
                        if !do_not_delete_clone.contains(&file) {
                            println!("deleting {}", file);
                            std::fs::remove_file(file).unwrap();
                        }
                    }

                    // no race condition since everybody pushes the same value to new_uid_offsets_clone
                    merged_files_clone.lock().unwrap().push(merged_filename);
                    new_uid_offsets_clone.lock().unwrap().push(0);
                    Result::<(), LavaError>::Ok(())
                });

                tasks.push(task);
            }

            // Wait for all tasks to complete
            let _: Vec<_> = futures::future::join_all(tasks)
                .await
                .into_iter()
                .collect::<Result<_, _>>()
                .unwrap();

            // Extract the merged files for the next level of merging
            let merged_files: Vec<String> = Arc::try_unwrap(merged_files_shared)
                .expect("Lock still has multiple owners")
                .into_inner()
                .unwrap();

            let new_uid_offsets = Arc::try_unwrap(new_uid_offsets_shared)
                .expect("Lock still has multiple owners")
                .into_inner()
                .unwrap();

            // Recurse with the newly merged files
            async_parallel_merge_files(
                condensed_lava_file,
                merged_files,
                do_not_delete,
                new_uid_offsets,
                K,
                mode,
            )
            .await
        }
    }
}

#[async_recursion]
async fn async_parallel_merge_vector_files(
    condensed_lava_file: String,
    files: Vec<String>,
    do_not_delete: BTreeSet<String>,
    column_name: &str,
    uid_nrows: Vec<Vec<usize>>,
    uid_to_metadatas: Vec<Vec<(String, usize, usize, usize, usize)>>,
    K: usize,
) -> Result<(), LavaError> {
    assert_eq!(K, 2);

    match files.len() {
        0 => Err(LavaError::Parse("out of chunks".to_string())), // Assuming LavaError can be constructed like this
        1 => {
            // the recursion will end here in this case. rename the files[0] to the supposed output name
            std::fs::rename(files[0].clone(), condensed_lava_file).unwrap();
            Ok(())
        }
        _ => {
            // More than one file, need to merge
            let mut tasks = vec![];
            let merged_files = Arc::new(Mutex::new(vec![]));
            let merged_uid_nrows = Arc::new(Mutex::new(vec![]));
            let merged_uid_to_metadata = Arc::new(Mutex::new(vec![]));

            let chunked_files: Vec<Vec<String>> = files
                .into_iter()
                .chunks(K)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            let chunked_uid_nrows: Vec<Vec<Vec<usize>>> = uid_nrows
                .into_iter() // Use iter() to get an iterator over references
                .chunks(K) // This comes from itertools
                .into_iter()
                .map(|chunk| chunk.collect::<Vec<_>>()) // Collect each chunk into a Vec
                .collect();

            let chunked_uid_metadata: Vec<Vec<Vec<(String, usize, usize, usize, usize)>>> =
                uid_to_metadatas
                    .into_iter()
                    .chunks(K)
                    .into_iter()
                    .map(|chunk| chunk.collect::<Vec<_>>())
                    .collect();

            for ((file_chunk, uid_nrows_chunk), uid_metadata_chunk) in chunked_files
                .into_iter()
                .zip(chunked_uid_nrows.into_iter())
                .zip(chunked_uid_metadata.into_iter())
            {
                if file_chunk.len() == 1 {
                    // If there's an odd file out, directly move it to the next level
                    merged_files.lock().unwrap().push(file_chunk[0].clone());
                    merged_uid_nrows
                        .lock()
                        .unwrap()
                        .push(uid_nrows_chunk[0].clone());
                    merged_uid_to_metadata
                        .lock()
                        .unwrap()
                        .push(uid_metadata_chunk[0].clone());
                    continue;
                }

                let merged_files_clone = Arc::clone(&merged_files);
                let new_uid_nrows_clone = Arc::clone(&merged_uid_nrows);
                let new_uid_metadata_clone = Arc::clone(&merged_uid_to_metadata);
                let do_not_delete_clone = do_not_delete.clone();
                let column_name = column_name.to_string(); // Convert &str to String

                let task = tokio::spawn(async move {
                    let my_uuid = uuid::Uuid::new_v4();
                    let merged_filename = my_uuid.to_string(); // Define this function based on your requirements

                    println!("mergin {:?}", file_chunk);

                    merge_lava_vector(
                        &merged_filename,
                        file_chunk.clone(),
                        &column_name,
                        &uid_nrows_chunk,
                        &uid_metadata_chunk,
                    )
                    .await
                    .unwrap();

                    // now go delete the input filesx

                    for file in file_chunk {
                        if !do_not_delete_clone.contains(&file) {
                            println!("deleting {}", file);
                            std::fs::remove_file(file).unwrap();
                        }
                    }

                    merged_files_clone.lock().unwrap().push(merged_filename);
                    new_uid_nrows_clone
                        .lock()
                        .unwrap()
                        .push(uid_nrows_chunk.concat());
                    new_uid_metadata_clone
                        .lock()
                        .unwrap()
                        .push(uid_metadata_chunk.concat());
                    Result::<(), LavaError>::Ok(())
                });

                tasks.push(task);
            }

            // Wait for all tasks to complete
            let _: Vec<_> = futures::future::join_all(tasks)
                .await
                .into_iter()
                .collect::<Result<_, _>>()
                .unwrap();

            // Extract the merged files for the next level of merging
            let merged_files: Vec<String> = Arc::try_unwrap(merged_files)
                .expect("Lock still has multiple owners")
                .into_inner()
                .unwrap();

            let new_uid_nrows = Arc::try_unwrap(merged_uid_nrows)
                .expect("Lock still has multiple owners")
                .into_inner()
                .unwrap();

            let new_uid_metadatas = Arc::try_unwrap(merged_uid_to_metadata)
                .expect("Lock still has multiple owners")
                .into_inner()
                .unwrap();

            // Recurse with the newly merged files
            async_parallel_merge_vector_files(
                condensed_lava_file,
                merged_files,
                do_not_delete,
                column_name,
                new_uid_nrows,
                new_uid_metadatas,
                K,
            )
            .await
        }
    }
}

#[tokio::main]
pub async fn parallel_merge_files(
    condensed_lava_file: String,
    files: Vec<String>,
    uid_offsets: Vec<u64>,
    K: usize,
    mode: usize, // 0 for bm25 1 for substring
) -> Result<(), LavaError> {
    let do_not_delete = BTreeSet::from_iter(files.clone().into_iter());
    let result = async_parallel_merge_files(
        condensed_lava_file,
        files,
        do_not_delete,
        uid_offsets,
        K,
        mode,
    )
    .await?;
    Ok(result)
}

#[tokio::main]
pub async fn parallel_merge_vector_files(
    condensed_lava_file: String,
    files: Vec<String>,
    column_name: &str,
    uid_nrows: Vec<Vec<usize>>,
    uid_to_metadatas: Vec<Vec<(String, usize, usize, usize, usize)>>,
) -> Result<(), LavaError> {
    let do_not_delete = BTreeSet::from_iter(files.clone().into_iter());
    let result = async_parallel_merge_vector_files(
        condensed_lava_file,
        files,
        do_not_delete,
        column_name,
        uid_nrows,
        uid_to_metadatas,
        2,
    )
    .await?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use crate::lava::merge::parallel_merge_files;

    #[test]
    pub fn test_merge_lava_bm25() {
        let res = parallel_merge_files(
            "merged.lava".to_string(),
            vec!["bump0.lava".to_string(), "bump1.lava".to_string()],
            vec![0, 1000000],
            2,
            0,
        );

        println!("{:?}", res);
    }

    #[test]
    pub fn test_merge_lava_substring() {
        let res = parallel_merge_files(
            "merged.lava".to_string(),
            vec![
                "chinese_index/0.lava".to_string(),
                "chinese_index/1.lava".to_string(),
            ],
            vec![0, 1000000],
            2,
            1,
        );

        println!("{:?}", res);
    }
}
