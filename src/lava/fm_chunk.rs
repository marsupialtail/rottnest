use std::collections::HashMap;
use zstd::stream::read::Decoder;
use bytes::Bytes;
use std::io::Read;
use zstd::stream::encode_all;
use super::error::LavaError;
pub(crate) struct FMChunk {
    counts_so_far : HashMap<u32, u64>,
    bwt_chunk : Vec<u32>,
}

impl FMChunk {
    pub fn new(
        chunk : Bytes
    ) -> Result<Self, LavaError> {
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

        Ok(Self {
            counts_so_far : counts,
            bwt_chunk : fm_chunk,
        })
    }

    pub fn serialize(&mut self) -> Result<Vec<u8>, LavaError> {
        let serialized_counts = bincode::serialize(&self.counts_so_far)?;
        let mut compressed_counts = encode_all(&serialized_counts[..], 0).expect("Compression failed");
        let mut result: Vec<u8> = vec![];
        result.append(&mut (compressed_counts.len() as u64).to_le_bytes().to_vec());
        result.append(&mut compressed_counts);
        let serialized_chunk = bincode::serialize(&self.bwt_chunk)?;
        let mut compressed_chunk = encode_all(&serialized_chunk[..], 0).expect("Compression failed");
        result.append(&mut compressed_chunk);
        Ok(result)
    }

}