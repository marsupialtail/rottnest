use bincode;
use zstd::bulk::compress;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;
use tokio::io::AsyncReadExt;

use opendal::raw::oio::ReadExt;
use opendal::services::Fs;

use opendal::{Operator, Writer};
use std::env;

use crate::formats::io::{AsyncReader, READER_BUFFER_SIZE, WRITER_BUFFER_SIZE};
use crate::lava::error::LavaError;
use crate::lava::plist::PListChunk;
use crate::lava::fm_chunk::FMChunk;
use std::collections::HashMap;

struct FMChunkIterator {
    reader: AsyncReader,
    fm_chunk_offsets: Vec<u64>,
    current_chunk_offset: usize,
    current_chunk: FMChunk
}

impl FMChunkIterator {
    // take ownership of the data structures
    pub async fn new(
        mut reader: AsyncReader,
        fm_chunk_offsets: Vec<u64>,
    ) -> Result<Self, LavaError> {

        let buffer3 = reader.read_range(fm_chunk_offsets[0], fm_chunk_offsets[1]).await?;
        let current_chunk = FMChunk::new(buffer3)?;

        Ok(Self {
            reader: reader,
            fm_chunk_offsets: fm_chunk_offsets,
            current_chunk_offset: 0,
            current_chunk: current_chunk,
        })
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

        let buffer3 = reader.read_range(plist_offsets[0], plist_offsets[1]).await?;
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

    pub fn get_current(&mut self) -> Vec<u64> {
        self.current_chunk[self.current_offset_in_chunk as usize].clone()
    }

    pub async fn increase_cursor(&mut self) -> Result<(), LavaError> {
        self.current_offset_in_chunk += 1;
        if self.current_offset_in_chunk == self.current_chunk.len() {
            // read the next chunk
            self.current_offset_in_chunk = 0;
            self.current_chunk_offset += 1;
            if self.current_chunk_offset + 2 > self.plist_offsets.len() {
                return Err(LavaError::Parse("out of chunks".to_string()));
            }
            
            let buffer3 = self.reader.read_range(self.plist_offsets[self.current_chunk_offset], self.plist_offsets[self.current_chunk_offset + 1]).await?;

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

#[tokio::main]
pub async fn merge_lava_bm25(
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
        let this_compressed_tokenizer: bytes::Bytes = reader.read_range(8, 8 + compressed_tokenizer_size).await?;

        match &compressed_tokenizer {
            Some(value) => assert!(this_compressed_tokenizer == value, "detected different tokenizers, cannot merge, something is very wrong."), 
            None => compressed_tokenizer = Some(this_compressed_tokenizer.to_vec())
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

    let mut output_file: Writer = operator
            .clone()
            .writer_with(condensed_lava_file)
            .buffer(WRITER_BUFFER_SIZE)
            .await?;

    let compressed_tokenizer = compressed_tokenizer.unwrap();
    let compressed_tokenizer_len = compressed_tokenizer.len();
    output_file.write(compressed_tokenizer).await?;

    let mut new_plist_offsets: Vec<u64> = vec![compressed_tokenizer_len as u64];
    let mut new_plist_elems: Vec<u64> = vec![0];
    let mut plist_chunk = PListChunk::new()?;
    let mut counter: u64 = 0;

    let mut compressed_term_dict_offset: u64 = compressed_tokenizer_len as u64;

    for tok in 0..combined_token_counts.len() { 
        // Find the smallest current line

        let mut plist: Vec<u64> = vec![];

        for i in 0.. plist_chunk_iterators.len() { 
            
            let this_plist: Vec<u64> = plist_chunk_iterators[i].get_current();
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

            let _ = plist_chunk_iterators[i].increase_cursor().await; 
        }

        counter += 1;

        let plist = Vec::from_iter(plist.into_iter());
        let written = plist_chunk.add_plist(&plist)?;
        if written > 1024 * 1024 || tok == combined_token_counts.len() - 1 {
            let bytes = plist_chunk.finalize_compression()?;
            let this_len: u64 = bytes.len() as u64;
            
            output_file.write(bytes).await?;
            compressed_term_dict_offset += this_len;
            new_plist_offsets
                .push(new_plist_offsets[new_plist_offsets.len() - 1] + this_len);
            new_plist_elems.push(counter);
            plist_chunk = PListChunk::new()?;
        }
    }

    new_plist_offsets.append(&mut new_plist_elems);

    let bytes = bincode::serialize(&combined_token_counts)?;
    let compressed_token_counts = encode_all(&bytes[..], 0).expect("Compression failed");

    let compressed_plist_offsets_offset = compressed_term_dict_offset + compressed_token_counts.len() as u64;
    output_file.write(compressed_token_counts).await?;

    let serialized = bincode::serialize(&new_plist_offsets).unwrap();
    let compressed_plist_offsets =
        encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");
    output_file.write(compressed_plist_offsets).await?;

    output_file.write((compressed_term_dict_offset as u64).to_le_bytes().to_vec()).await?;
    output_file.write((compressed_plist_offsets_offset as u64).to_le_bytes().to_vec()).await?;
    output_file.write((total_num_documents as u64).to_le_bytes().to_vec()).await?;
    output_file.close().await?;
    Ok(())
}

async fn compute_interleave(bwt0_reader: &mut Reader, bwt1_reader: &mut Reader, lens: (usize, usize), counts: &[usize; 256]) -> Result<BitVec, Error> {
    let (bwt0_len, bwt1_len) = lens;

    // construct character starts array
    let mut starts: [usize; 256] = [0; 256];
    let mut sum = 0;
    for i in 0..256 {
        starts[i] = sum;
        sum += counts[i];
    }

    let mut interleave = BitVec::from_elem(bwt0_len + bwt1_len, true);
    for i in 0..bwt0_len {
        interleave.set(i, false);
    }

    let mut interleave_iterations = 0;

    loop {
        let mut ind: [usize; 2] = [0, 0];

        // reset readers
        bwt0_reader.seek(std::io::SeekFrom::Start(0)).await?;
        bwt1_reader.seek(std::io::SeekFrom::Start(0)).await?;

        let mut bwt0 = vec![0u8; BUFFER_SIZE];
        let mut bwt1 = vec![0u8; BUFFER_SIZE];
        bwt0_reader.read(&mut bwt0).await?;
        bwt1_reader.read(&mut bwt1).await?;

        let mut offsets = starts.clone();
        let mut new_interleave = BitVec::from_elem(interleave.len(), false);
        for i in 0..interleave.len() {
            if interleave[i] {
                new_interleave.set(offsets[bwt1[ind[1]] as usize], true);
                offsets[bwt1[ind[1]] as usize] += 1;
                ind[1] += 1;

                if ind[1] == BUFFER_SIZE {
                    bwt1_reader.read(&mut bwt1).await?;
                    ind[1] = 0;
                }
            } else {
                offsets[bwt0[ind[0]] as usize] += 1;
                ind[0] += 1;

                if ind[0] == BUFFER_SIZE {
                    bwt0_reader.read(&mut bwt0).await?;
                    ind[0] = 0;
                }
            }
        }

        interleave_iterations += 1;

        if new_interleave == interleave {
            break;
        }
        interleave = new_interleave;
    }

    println!("interleave iterations: {}", interleave_iterations);
    Ok(interleave)
}

#[tokio::main]
pub async fn merge_lava_substring(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> // hawaiian for lava condensation
{
    // first merge the tokenizer, then merge the fm indices then merge the posting lists. 
    Ok(())

}

#[cfg(test)]
mod tests {
    use super::merge_lava_bm25;

    #[test]
    pub fn test_merge_lava_bm25() {

        let res = merge_lava_bm25(
            "merged.lava",
            vec![
                "bump1.lava".to_string(),
                "bump2.lava".to_string(),
            ],
            vec![0, 1000000],
        );

        println!("{:?}", res);
    }
}