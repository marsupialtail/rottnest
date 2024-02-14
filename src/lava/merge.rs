use bincode;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;

use opendal::raw::oio::ReadExt;
use opendal::services::Fs;

use opendal::{Operator, Reader};
use std::env;

use crate::formats::reader::AsyncReader;
use crate::lava::error::LavaError;
use crate::lava::plist::PList;

struct PListChunkIterator {
    reader : AsyncReader,
    current_offset_in_chunk: usize,
    current_chunk_offset: usize,
    current_chunk: Vec<Vec<u64>>,
    plist_offsets: Vec<u64>,
    plist_elems: Vec<u64>
}



impl PListChunkIterator {
    // take ownership of the data structures
    pub async fn new (mut reader: AsyncReader, plist_offsets: Vec<u64>, plist_elems: Vec<u64>) -> Result<Self, LavaError> {
        // read the first chunk
        reader.seek(SeekFrom::Start(plist_offsets[0])).await?;
        let mut buffer3: Vec<u8> = vec![0u8; (plist_offsets[1] - plist_offsets[0]) as usize];
        reader.read(&mut buffer3).await?;
        let mut result: Vec<Vec<u64>> = PList::search_compressed(buffer3, (0..plist_elems[1]).collect()).unwrap();
 
        Ok(Self {
            reader: reader,
            current_offset_in_chunk: 0,
            current_chunk_offset: 0,
            current_chunk: result,
            plist_offsets: plist_offsets,
            plist_elems: plist_elems
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
            let mut buffer3: Vec<u8> = vec![0u8; (self.plist_offsets[self.current_chunk_offset + 1] - self.plist_offsets[self.current_chunk_offset]) as usize];
            self.reader.read(&mut buffer3).await?;
            self.current_chunk = PList::search_compressed(buffer3, 
                (0..(self.plist_elems[self.current_chunk_offset + 1] - self.plist_elems[self.current_chunk_offset])).collect()).unwrap();
        }

        Ok(())
    }
}

#[tokio::main]
async fn hoa(
    condensed_lava_file: &str,
    operator: &mut Operator,
    lava_files: Vec<Cow<str>>,
    uid_offsets: Vec<u64>
) -> Result<(), LavaError> // hawaiian for lava condensation
{
    // instantiate a list of readers from lava_files
    // let mut readers: Vec<Reader> = Vec::with_capacity(lava_files.len());
    let mut decompressed_term_dictionaries: Vec<BufReader<Cursor<Vec<u8>>>> =
        Vec::with_capacity(lava_files.len());
    let mut file_sizes: Vec<u64> = Vec::with_capacity(lava_files.len());
    // let mut plist_offsets: Vec<Vec<u64>> = Vec::with_capacity(lava_files.len());
    // let mut plist_elems: Vec<Vec<u64>> = Vec::with_capacity(lava_files.len());

    let mut plist_chunk_iterators: Vec<PListChunkIterator> = Vec::with_capacity(lava_files.len());

    // read in and decompress all the term dictionaries in memory. The term dictionaries corresponding to English language should be small.

    for file in lava_files {
        let file = file.as_ref();
        let file_size: u64 = operator.stat(file).await?.content_length();
        let mut reader: AsyncReader = operator.clone().reader(file).await?.into();

        let (compressed_term_dict_offset, compressed_plist_offsets_offset) =
            reader.read_offsets().await?;

        let compressed_term_dictionary = reader
            .read_range(compressed_term_dict_offset, compressed_plist_offsets_offset)
            .await?;

        let mut decompressed_term_dictionary: Vec<u8> = Vec::new();
        let mut decompressor: Decoder<'_, BufReader<&[u8]>> =
            Decoder::new(&compressed_term_dictionary[..])?;
        decompressor.read_to_end(&mut decompressed_term_dictionary)?;
        let cursor = Cursor::new(decompressed_term_dictionary);
        let buf_reader: BufReader<Cursor<Vec<u8>>> = BufReader::new(cursor);

        let buffer2 = reader
            .read_range(compressed_plist_offsets_offset, file_size - 16)
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

        decompressed_term_dictionaries.push(buf_reader);
        file_sizes.push(file_size);
        plist_chunk_iterators.push(PListChunkIterator::new(reader, 
            this_plist_offsets[.. num_elements].to_vec(), 
            this_plist_offsets[num_elements ..].to_vec()).await?);
    }

    // now do the merge sort

    let mut term_dictionary = String::new();
    let mut current_lines: Vec<Option<String>> = vec![None; decompressed_term_dictionaries.len()];

    // Initialize the current line for each reader
    for (i, reader) in decompressed_term_dictionaries.iter_mut().enumerate() {
        if let Some(Ok(line)) = reader.lines().next() {
            current_lines[i] = Some(line);
        }
    }

    let mut output_file = File::create(condensed_lava_file)?;
    let mut new_plist_offsets: Vec<u64> = vec![0];
    let mut new_plist_elems: Vec<u64> = vec![0];
    let mut plist_chunk = PList::new()?;
    let mut counter: u64 = 0;

    while current_lines.iter().any(Option::is_some) {
        // Find the smallest current line
        let smallest_line = current_lines
            .iter()
            .filter_map(|line| line.as_ref()) // Extract only Some(&String) elements
            .min()
            .unwrap()
            .clone(); // Find the minimum line

        term_dictionary += &smallest_line;
        term_dictionary += "\n";

        // Progress the BufReaders whose last output was smallest line

        let mut plist: BTreeSet<u64> = BTreeSet::new();

        for i in 0..current_lines.len() {
            if current_lines[i].is_some() {
                let line = current_lines[i].as_ref().unwrap();
                if line.eq(&smallest_line) {
                    // we need to read and decompress the plists
                    
                    let this_plist: Vec<u64> = plist_chunk_iterators[i].get_current();
                    // println!("{:?} {:?}", this_plist, uid_offsets[i]);
                    for item in this_plist {
                        plist.insert(item + uid_offsets[i]);
                    }

                    let _ = plist_chunk_iterators[i].increase_cursor().await;

                    let reader = &mut decompressed_term_dictionaries[i];
                    if let Some(Ok(line)) = reader.lines().next() {
                        current_lines[i] = Some(line);
                    } else {
                        current_lines[i] = None;
                    }
                }
            }
        }

        counter += 1;

        // value_vec.sort();
        // println!("{}", key);
        let written = plist_chunk.add_plist(&plist)?;
        if written > 1024 * 1024 || current_lines.iter().all(Option::is_none) {
            let bytes = plist_chunk.finalize_compression()?;
            output_file.write_all(&bytes)?;
            new_plist_offsets.push(new_plist_offsets[new_plist_offsets.len() - 1] + bytes.len() as u64);
            new_plist_elems.push(counter);
            plist_chunk = PList::new()?;
        }
    }

    new_plist_offsets.append(&mut new_plist_elems);

    let bytes = term_dictionary.as_bytes();
    let compressed_term_dictionary = encode_all(bytes, 0).expect("Compression failed");
    let compressed_term_dict_offset = output_file.seek(SeekFrom::Current(0))?;

    println!("merged compress dict size {:?} len {:?}", compressed_term_dictionary.len(), counter);

    output_file.write_all(&compressed_term_dictionary)?;

    let compressed_plist_offsets_offset = output_file.seek(SeekFrom::Current(0))?;
    let serialized = bincode::serialize(&new_plist_offsets).unwrap();
    let compressed_plist_offsets =
        encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");
    output_file.write_all(&compressed_plist_offsets)?;

    output_file.write_all(&(compressed_term_dict_offset as u64).to_le_bytes())?;
    output_file.write_all(&(compressed_plist_offsets_offset as u64).to_le_bytes())?;

    Ok(())
}

pub fn merge_lava(
    condensed_lava_file: Cow<str>,
    lava_files: Vec<Cow<str>>,
    uid_offsets: Vec<u64>
) -> Result<(), LavaError> {
    // you should only merge them on local disk. It's not worth random accessing S3 for this because of the request costs.
    // worry about running out of disk later. Assume you have a fast SSD for now.
    let mut builder = Fs::default();
    let current_path = env::current_dir()?;
    builder.root(current_path.to_str().expect("no path"));
    let mut operator = Operator::new(builder)?.finish();

    hoa(condensed_lava_file.as_ref(), &mut operator, lava_files, uid_offsets)
}
