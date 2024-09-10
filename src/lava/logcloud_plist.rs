use std::convert::TryInto;
use std::io::{Read, Write};
use std::mem::size_of;
use zstd::stream::read::Decoder;
use zstd::stream::write::Encoder;

pub type PlistSize = u32;

pub struct PListChunk {
    data: Vec<Vec<PlistSize>>,
}

impl PListChunk {
    pub fn new(data: Vec<Vec<PlistSize>>) -> Self {
        PListChunk { data }
    }

    pub fn from_compressed(compressed_data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut decoder = Decoder::new(compressed_data)?;
        let mut data = Vec::new();
        decoder.read_to_end(&mut data)?;

        let num_posting_lists = PlistSize::from_le_bytes(data[data.len() - size_of::<PlistSize>()..].try_into()?);

        let mut bit_array = Vec::with_capacity(num_posting_lists as usize);
        for i in 0..num_posting_lists {
            bit_array.push((data[i as usize / 8] & (1 << (i % 8))) != 0);
        }

        let mut cursor = (num_posting_lists as usize + 7) / 8;

        let mut count_array = Vec::with_capacity(num_posting_lists as usize);
        for &bit in &bit_array {
            if bit {
                count_array.push(PlistSize::from_le_bytes(data[cursor..cursor + size_of::<PlistSize>()].try_into()?));
                cursor += size_of::<PlistSize>();
            } else {
                count_array.push(1);
            }
        }

        let mut plist_data = Vec::with_capacity(num_posting_lists as usize);
        for &count in &count_array {
            let mut posting_list = Vec::with_capacity(count as usize);
            for _ in 0..count {
                posting_list.push(PlistSize::from_le_bytes(data[cursor..cursor + size_of::<PlistSize>()].try_into()?));
                cursor += size_of::<PlistSize>();
            }
            plist_data.push(posting_list);
        }

        Ok(PListChunk { data: plist_data })
    }

    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let num_posting_lists = self.data.len();
        let bit_array_size = (num_posting_lists + 7) / 8;
        let count_array_size = self.data.iter().filter(|list| list.len() > 1).count() * size_of::<PlistSize>();
        let posting_lists_size = self.data.iter().map(|list| list.len() * size_of::<PlistSize>()).sum::<usize>();
        let size = bit_array_size + count_array_size + posting_lists_size + size_of::<PlistSize>();

        let mut serialized = vec![0u8; size];

        for (i, list) in self.data.iter().enumerate() {
            if list.len() > 1 {
                serialized[i / 8] |= 1 << (i % 8);
            }
        }

        let mut cursor = bit_array_size;

        for list in &self.data {
            if list.len() > 1 {
                serialized[cursor..cursor + size_of::<PlistSize>()]
                    .copy_from_slice(&(list.len() as PlistSize).to_le_bytes());
                cursor += size_of::<PlistSize>();
            }
        }

        for list in &self.data {
            for &item in list {
                serialized[cursor..cursor + size_of::<PlistSize>()].copy_from_slice(&item.to_le_bytes());
                cursor += size_of::<PlistSize>();
            }
        }

        serialized[cursor..].copy_from_slice(&(num_posting_lists as PlistSize).to_le_bytes());

        let mut encoder = Encoder::new(Vec::new(), 0)?;
        encoder.write_all(&serialized)?;
        Ok(encoder.finish()?)
    }

    pub fn data(&self) -> &Vec<Vec<PlistSize>> {
        &self.data
    }

    pub fn lookup(&self, key: usize) -> Option<Vec<PlistSize>> {
        self.data.get(key).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_data() {
        let data = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let chunk = PListChunk::new(data.clone());
        assert_eq!(chunk.data(), &data);
    }

    #[test]
    fn test_lookup() {
        let data = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let chunk = PListChunk::new(data);
        assert_eq!(chunk.lookup(0), Some(vec![1, 2, 3]));
        assert_eq!(chunk.lookup(1), Some(vec![4, 5]));
        assert_eq!(chunk.lookup(2), Some(vec![6]));
        assert_eq!(chunk.lookup(3), None);
    }

    #[test]
    fn test_serialize_and_from_compressed() -> Result<(), Box<dyn std::error::Error>> {
        let original_data = vec![vec![1, 2, 3], vec![4], vec![5, 6, 7, 8]];
        let chunk = PListChunk::new(original_data);

        let serialized = chunk.serialize()?;
        let deserialized_chunk = PListChunk::from_compressed(&serialized)?;

        assert_eq!(chunk.data(), deserialized_chunk.data());
        Ok(())
    }

    #[test]
    fn test_empty_and_large_lists() -> Result<(), Box<dyn std::error::Error>> {
        let original_data = vec![vec![], vec![1], vec![2, 3], (0..1000).collect()];
        let chunk = PListChunk::new(original_data);

        let serialized = chunk.serialize()?;
        let deserialized_chunk = PListChunk::from_compressed(&serialized)?;

        assert_eq!(chunk.data(), deserialized_chunk.data());
        Ok(())
    }
}
