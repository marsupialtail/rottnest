use bincode;
use zstd::bulk::compress;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;

use opendal::raw::oio::ReadExt;
use opendal::services::Fs;

use opendal::{Operator, Writer};
use std::env;

use crate::formats::io::{AsyncReader, READER_BUFFER_SIZE, WRITER_BUFFER_SIZE};
use crate::lava::error::LavaError;
use crate::lava::plist::PListChunk;

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
async fn hoa(
    condensed_lava_file: &str,
    operator: &mut Operator,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> // hawaiian for lava condensation
{
    // instantiate a list of readers from lava_files
    // let mut readers: Vec<Reader> = Vec::with_capacity(lava_files.len());
    
    let mut file_sizes: Vec<u64> = Vec::with_capacity(lava_files.len());
    // let mut plist_offsets: Vec<Vec<u64>> = Vec::with_capacity(lava_files.len());
    // let mut plist_elems: Vec<Vec<u64>> = Vec::with_capacity(lava_files.len());

    let mut plist_chunk_iterators: Vec<PListChunkIterator> = Vec::with_capacity(lava_files.len());

    // read in and decompress all the term dictionaries in memory. The term dictionaries corresponding to English language should be small.

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

        let (compressed_term_dict_offset, compressed_plist_offsets_offset, num_documents) =
            reader.read_offsets().await?;
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

        let this_compressed_tokenizer: bytes::Bytes = reader.read_range(0, this_plist_offsets[0]).await?;

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

pub fn merge_lava(
    condensed_lava_file: String,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
) -> Result<(), LavaError> {
    // you should only merge them on local disk. It's not worth random accessing S3 for this because of the request costs.
    // worry about running out of disk later. Assume you have a fast SSD for now.
    let mut builder = Fs::default();
    let current_path = env::current_dir()?;
    builder.root(current_path.to_str().expect("no path"));
    let mut operator = Operator::new(builder)?.finish();

    hoa(
        condensed_lava_file.as_ref(),
        &mut operator,
        lava_files,
        uid_offsets,
    )
}

#[cfg(test)]
mod tests {
    use super::merge_lava;

    #[test]
    pub fn test_merge_lava() {

        let res = merge_lava(
            "merged.lava".to_string(),
            vec![
                "bump1.lava".to_string(),
                "bump2.lava".to_string(),
            ],
            vec![0, 1000000],
        );

        println!("{:?}", res);
    }
}