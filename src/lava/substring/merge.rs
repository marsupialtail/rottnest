use std::{collections::HashMap, io::Seek, io::SeekFrom, io::Write};

use super::constants::*;
use super::fm_chunk::FMChunk;
use crate::{
    formats::readers::{get_file_size_and_reader, AsyncReader, ReaderType},
    lava::error::LavaError,
};
use bit_vec::BitVec;
use std::fs::File;
use zstd::stream::encode_all;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use num_traits;

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
            reader,
            plist_offsets,
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

struct FMChunkIterator<T> 
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Eq + Hash,
{
    reader: AsyncReader,
    fm_chunk_offsets: Vec<u64>,
    current_chunk_offset: usize,
    pub current_chunk: FMChunk<T>,
}

impl<T> FMChunkIterator<T> 
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Eq + Hash,
{
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
            reader,
            fm_chunk_offsets,
            current_chunk_offset: 0,
            current_chunk,
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

async fn compute_interleave<T>(
    bwt0_reader: &mut FMChunkIterator<T>,
    bwt1_reader: &mut FMChunkIterator<T>,
    lens: (usize, usize),
    cumulative_counts: &Vec<u64>,
) -> Result<BitVec, LavaError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Eq + Hash + Copy,
    T: num_traits::AsPrimitive<usize>,
{
    let (bwt0_len, bwt1_len) = lens;

    let mut interleave = BitVec::from_elem(bwt0_len + bwt1_len, true);
    for i in 0..bwt0_len {
        interleave.set(i, false);
    }

    // let mut interleave_iterations = 0;

    for _ in 0..10 {
        let mut ind: [usize; 2] = [0, 0];

        let mut bwt0 = &bwt0_reader.current_chunk.bwt_chunk;
        let mut bwt1 = &bwt1_reader.current_chunk.bwt_chunk;

        let mut offsets = cumulative_counts.clone();
        let mut new_interleave = BitVec::from_elem(interleave.len(), false);
        for i in 0..interleave.len() {
            if interleave[i] {
                new_interleave.set(offsets[bwt1[ind[1]].as_()] as usize, true);
                offsets[bwt1[ind[1]].as_()] += 1;
                ind[1] += 1;

                if ind[1] == bwt1.len() {
                    // will return an Err for the last chunk, that's ok
                    let _ = bwt1_reader.advance().await;
                    bwt1 = &bwt1_reader.current_chunk.bwt_chunk;
                    ind[1] = 0;
                }
            } else {
                offsets[bwt0[ind[0]].as_()] += 1;
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

        // interleave_iterations += 1;
        // println!(
        //     "{}  ",
        //     interleave_iterations,
        // );

        if new_interleave == interleave {
            break;
        }
        interleave = new_interleave;
    }

    // println!("interleave iterations: {}", interleave_iterations);
    Ok(interleave)
}

pub(crate) async fn merge_lava_substring(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
    reader_type: ReaderType,
) -> Result<Vec<(usize, usize)>, LavaError> {
    // first merge the tokenizer, then merge the fm indices then merge the posting lists.
    // let mut builder = Fs::default();
    // let current_path = env::current_dir()?;
    // builder.root(current_path.to_str().expect("no path"));
    // let operator = Operator::new(builder)?.finish();

    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    // currently only support merging two files, but can support more in the future.
    assert_eq!(lava_files.len(), 2);
    assert_eq!(uid_offsets.len(), 2);

    let mut ns: Vec<u64> = vec![];
    let mut combined_cumulative_counts: Vec<u64> = vec![];
    let mut fm_chunk_iterators: Vec<FMChunkIterator<u32>> = vec![];
    let mut plist_iterators: Vec<PListIterator> = vec![];

    for file in lava_files {
        // @Rain just make two different readers for now because this is hopefully low overhead
        // instead of bothering with wrapping this thing in Arc<Mutex<>>. Lots of tech debt to clean up
        // needed for the FMChunkIterator and PListIterator
        let (_, mut reader) = get_file_size_and_reader(file.clone(), reader_type.clone()).await?;
        let (file_size, reader1) =
            get_file_size_and_reader(file.clone(), reader_type.clone()).await?;
        let file_size = file_size as u64;

        let results = reader.read_usize_from_end(4).await?;
        let fm_chunk_offsets_offset = results[0];
        let posting_list_offsets_offset = results[1];
        let total_counts_offset = results[2];
        let n = results[3];

        ns.push(n);

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

        let fm_chunk_offsets: Vec<u64> = reader
            .read_range_and_decompress(fm_chunk_offsets_offset, posting_list_offsets_offset)
            .await?;
        let posting_list_offsets: Vec<u64> = reader
            .read_range_and_decompress(posting_list_offsets_offset, total_counts_offset)
            .await?;
        let cumulative_counts: Vec<u64> = reader
            .read_range_and_decompress(total_counts_offset, (file_size - 32) as u64)
            .await?;

        // println!("{} {}", file, cumulative_counts.len());

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

    // let start = std::time::Instant::now();
    let interleave: BitVec = compute_interleave(
        &mut bwt0_reader,
        &mut bwt1_reader,
        (ns[0] as usize, ns[1] as usize),
        &combined_cumulative_counts,
    )
    .await?;

    let _ = bwt0_reader.reset().await?;
    let _ = bwt1_reader.reset().await?;

    // let duration = start.elapsed();
    // println!("interleave time: {:?}", duration);

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

    let cache_start = output_file.seek(SeekFrom::Current(0))? as usize;

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

    Ok(vec![(
        cache_start,
        output_file.seek(SeekFrom::Current(0))? as usize,
    )])
}

pub(crate) async fn merge_lava_substring_char(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
    reader_type: ReaderType,
) -> Result<Vec<(usize, usize)>, LavaError> {
    // first merge the fm indices then merge the posting lists.
    // No tokenizer needed for character-based indexes

    // currently only support merging two files, but can support more in the future.
    assert_eq!(lava_files.len(), 2);
    assert_eq!(uid_offsets.len(), 2);

    let mut ns: Vec<u64> = vec![];
    let mut combined_cumulative_counts: Vec<u64> = vec![];
    let mut fm_chunk_iterators: Vec<FMChunkIterator<u8>> = vec![];
    let mut plist_iterators: Vec<PListIterator> = vec![];

    for file in lava_files {
        // Create two different readers to avoid wrapping in Arc<Mutex<>>
        let (_, mut reader) = get_file_size_and_reader(file.clone(), reader_type.clone()).await?;
        let (file_size, reader1) =
            get_file_size_and_reader(file.clone(), reader_type.clone()).await?;
        let file_size = file_size as u64;

        let results = reader.read_usize_from_end(4).await?;
        let fm_chunk_offsets_offset = results[0];
        let posting_list_offsets_offset = results[1];
        let total_counts_offset = results[2];
        let n = results[3];

        ns.push(n);

        let fm_chunk_offsets: Vec<u64> = reader
            .read_range_and_decompress(fm_chunk_offsets_offset, posting_list_offsets_offset)
            .await?;
        let posting_list_offsets: Vec<u64> = reader
            .read_range_and_decompress(posting_list_offsets_offset, total_counts_offset)
            .await?;
        let cumulative_counts: Vec<u64> = reader
            .read_range_and_decompress(total_counts_offset, (file_size - 32) as u64)
            .await?;

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

    let interleave: BitVec = compute_interleave(
        &mut bwt0_reader,
        &mut bwt1_reader,
        (ns[0] as usize, ns[1] as usize),
        &combined_cumulative_counts,
    )
    .await?;

    let _ = bwt0_reader.reset().await?;
    let _ = bwt1_reader.reset().await?;

    let mut output_file = File::create(condensed_lava_file)?;
    
    let mut bwt_output: Vec<u8> = Vec::with_capacity(interleave.len());
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

    let mut current_chunk: Vec<u8> = vec![];
    let mut current_chunk_counts: HashMap<u8, u64> = HashMap::new();
    let mut next_chunk_counts: HashMap<u8, u64> = HashMap::new();
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

    let cache_start = output_file.seek(SeekFrom::Current(0))? as usize;

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

    Ok(vec![(
        cache_start,
        output_file.seek(SeekFrom::Current(0))? as usize,
    )])
}
