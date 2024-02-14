use parquet::data_type::AsBytes;
use std::{
    collections::BTreeSet,
    io::{Cursor, Result},
};
use zstd::stream::read::Decoder;
use zstd::stream::write::Encoder;

use std::io::{BufRead, BufReader, Read, SeekFrom, Write};
use zstd::stream::encode_all;

pub struct PList<'a> {
    compressor: Encoder<'a, Cursor<Vec<u8>>>,
    plist_offsets: Vec<usize>,
    elems: u64,
    last_flushed: u64,
}

impl<'a> PList<'a> {
    pub fn new() -> Result<Self> {
        let plist_compressed = Vec::new(); // Initially empty Vec<u8>
        let cursor = Cursor::new(plist_compressed); // Cursor for Encoder to write into
                                                    // Initialize the encoder with the cursor. The level 0 indicates the default compression level.
        let compressor = Encoder::new(cursor, 10)?;

        Ok(Self {
            compressor,
            plist_offsets: vec![0],
            elems: 0,
            last_flushed: 0, // The actual compressed data Vec<u8> will be accessed differently
        })
    }

    pub fn search_compressed(compressed: Vec<u8>, indices: Vec<u64>) -> Result<Vec<Vec<u64>>> {
        // first read the last 8 bytes
        let compressed_plist_offsets_offset = u64::from_le_bytes(
            compressed[compressed.len() - 8..compressed.len()]
                .try_into()
                .expect("data corruption"),
        );
        let mut decompressed_plists: Vec<u8> = Vec::new();
        let mut decompressor =
            Decoder::new(&compressed[..compressed_plist_offsets_offset as usize])?;
        decompressor.read_to_end(&mut decompressed_plists)?;


        let mut decompressed_plist_offsets: Vec<u8> = Vec::new();
        let mut decompressor = Decoder::new(
            &compressed[compressed_plist_offsets_offset as usize..compressed.len() as usize - 8],
        )?;
        decompressor.read_to_end(&mut decompressed_plist_offsets)?;
        let mut decompressed_plist_offsets: Vec<u64> =
            bincode::deserialize(&decompressed_plist_offsets).unwrap();

        let mut result = Vec::new();
        for index in indices {
            let plist_offset = decompressed_plist_offsets[index as usize];
            let plist_size = decompressed_plist_offsets[index as usize + 1] - plist_offset;
            let plist: Vec<u64> = bincode::deserialize(
                &decompressed_plists
                    [plist_offset as usize..plist_offset as usize + plist_size as usize],
            )
            .unwrap();
            let test: BTreeSet<u64> = bincode::deserialize(
                &decompressed_plists
                    [plist_offset as usize..plist_offset as usize + plist_size as usize],
            )
            .unwrap();
            result.push(plist);
        }

        Ok(result)
    }

    pub fn add_plist(&mut self, plist: &BTreeSet<u64>) -> Result<usize> {
        self.elems += plist.len() as u64;
        // serialize first
        let serialized = bincode::serialize(&plist).unwrap();
        self.plist_offsets
            .push(serialized.len() + self.plist_offsets[self.plist_offsets.len() - 1]);
        self.compressor.write_all(&serialized)?;
        if self.elems > self.last_flushed + 10000 {
            self.compressor.flush()?;
            self.last_flushed = self.elems;
        }

        Ok(self.compressor.get_ref().position() as usize)
    }

    // Example method to finalize compression and retrieve compressed data
    pub fn finalize_compression(mut self) -> Result<Vec<u8>> {
        // Finish compression and retrieve the inner Cursor<Vec<u8>>

        let mut cursor = self.compressor.finish()?;
        let compressed_plist_offsets_offset = cursor.position();

        // now we should compress the plist_offsets too and add it to the end of the compressed_plists
        let serialized = bincode::serialize(&self.plist_offsets).unwrap();

        let compressed_plist_offsets = encode_all(serialized.as_bytes(), 0)?;

        // write the compressed plist offsets
        cursor.write_all(&compressed_plist_offsets)?;

        // push the final 8 bytes that has the offset
        let compressed_plist_offsets_size_bytes = compressed_plist_offsets_offset.to_le_bytes();
        cursor.write_all(&compressed_plist_offsets_size_bytes)?;

        // Retrieve the Vec<u8> from the cursor
        Ok(cursor.into_inner())
    }
}

#[cfg(test)]
mod tests {
    // Import the parent module to the scope of the tests
    use super::*;
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    // A test function
    #[test]
    fn test_plist() -> Result<()> {
        let mut plist = PList::new()?;
        let a = BTreeSet::from([0, 1, 2]);

        let b = plist.add_plist(&a)?;
        println!("written {}", b);
        let result = plist.finalize_compression()?;
        println!("{:?}", result);
        let result = PList::search_compressed(result, vec![0]).unwrap();
        assert_eq!(result[0], a.into_iter().collect::<Vec<u64>>());
        Ok(())
    }

    #[test]
    fn test_big() -> Result<()> {
        let mut rng = thread_rng();
        let mut numbers: Vec<u64> = (0..=100).collect(); // Range of numbers from 0 to 100
        numbers.shuffle(&mut rng); // Shuffle the numbers
        let mut sorted_numbers: Vec<u64> = numbers.into_iter().take(60).collect(); // Take the first 60 numbers
                                                                                   // sorted_numbers.sort(); // Sort the taken numbers
        let sorted_numbers = BTreeSet::from_iter(sorted_numbers.into_iter());
        let mut plist = PList::new()?;

        for i in 0..100 {
            let b = plist.add_plist(&sorted_numbers)?;
            println!("written {}", b);
        }
        let result = plist.finalize_compression()?;
        println!("{:?}", result);

        let result = PList::search_compressed(result, vec![0]).unwrap();
        assert_eq!(result[0], sorted_numbers.into_iter().collect::<Vec<u64>>());

        Ok(())
    }
}
