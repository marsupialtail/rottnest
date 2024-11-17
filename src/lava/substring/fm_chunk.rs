use crate::lava::error::LavaError;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;

pub(crate) struct FMChunk<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Eq + std::hash::Hash,
{
    pub counts_so_far: HashMap<T, u64>,
    pub bwt_chunk: Vec<T>,
}

impl<T> FMChunk<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Eq + std::hash::Hash,
{
    pub fn new(chunk: Bytes) -> Result<Self, LavaError> {
        let compressed_counts_size = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let compressed_counts = &chunk[8..(compressed_counts_size + 8) as usize];
        let mut decompressor = Decoder::new(compressed_counts)?;
        let mut serialized_counts: Vec<u8> = Vec::with_capacity(compressed_counts_size as usize);
        decompressor.read_to_end(&mut serialized_counts)?;
        let counts: HashMap<T, u64> = bincode::deserialize(&serialized_counts)?;
        let compressed_fm_chunk = &chunk[(compressed_counts_size + 8) as usize..];
        let mut decompressor = Decoder::new(compressed_fm_chunk)?;
        let mut serialized_fm_chunk: Vec<u8> =
            Vec::with_capacity(compressed_fm_chunk.len() as usize);
        decompressor.read_to_end(&mut serialized_fm_chunk)?;
        let fm_chunk: Vec<T> = bincode::deserialize(&serialized_fm_chunk)?;

        Ok(Self {
            counts_so_far: counts,
            bwt_chunk: fm_chunk,
        })
    }

    #[allow(dead_code)]
    pub fn serialize(&mut self) -> Result<Vec<u8>, LavaError> {
        let serialized_counts = bincode::serialize(&self.counts_so_far)?;
        let mut compressed_counts =
            encode_all(&serialized_counts[..], 0).expect("Compression failed");
        let mut result: Vec<u8> = vec![];
        result.append(&mut (compressed_counts.len() as u64).to_le_bytes().to_vec());
        result.append(&mut compressed_counts);
        let serialized_chunk = bincode::serialize(&self.bwt_chunk)?;
        let mut compressed_chunk =
            encode_all(&serialized_chunk[..], 0).expect("Compression failed");
        result.append(&mut compressed_chunk);
        Ok(result)
    }

    pub fn search(&self, token: T, pos: usize) -> Result<u64, LavaError> {
        let mut result = *self.counts_so_far.get(&token).unwrap_or(&0);
        for j in 0..pos {
            if self.bwt_chunk[j] == token {
                result += 1;
            }
        }
        Ok(result)
    }
}
