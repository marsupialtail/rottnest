use bincode;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write};
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;

use anyhow::{anyhow, Result};
use opendal::raw::oio::ReadExt;
use opendal::services::Fs;

use opendal::{Operator, Reader};
use std::env;

use crate::lava::error::LavaError;

#[tokio::main]
async fn hoa(
    condensed_lava_file: &str,
    operator: &mut Operator,
    lava_files: Vec<Cow<str>>,
) -> Result<()> // hawaiian for lava condensation
{
    // instantiate a list of readers from lava_files
    let mut readers: Vec<Reader> = Vec::with_capacity(lava_files.len());
    let mut decompressed_term_dictionaries: Vec<BufReader<Cursor<Vec<u8>>>> =
        Vec::with_capacity(lava_files.len());
    let mut file_sizes: Vec<u64> = Vec::with_capacity(lava_files.len());
    let mut plist_offsets: Vec<Vec<u64>> = Vec::with_capacity(lava_files.len());

    // read in and decompress all the term dictionaries in memory. The term dictionaries corresponding to English language should be small.

    for file in lava_files {
        let file = file.as_ref();
        let file_size: u64 = operator.stat(file).await?.content_length();
        let mut reader: Reader = operator.clone().reader(file).await?;
        reader.seek(SeekFrom::End(-16)).await?;
        let mut buffer1 = [0u8; 8];
        reader.read(&mut buffer1).await?;
        let compressed_term_dict_offset = u64::from_le_bytes(buffer1);
        let mut buffer: [u8; 8] = [0u8; 8];
        reader.read(&mut buffer[..]).await?;
        let compressed_plist_offsets_offset = u64::from_le_bytes(buffer);

        reader
            .seek(SeekFrom::Start(compressed_term_dict_offset))
            .await?;
        let mut compressed_term_dictionary: Vec<u8> =
            vec![0u8; (compressed_plist_offsets_offset - compressed_term_dict_offset) as usize];
        reader.read(&mut compressed_term_dictionary[..]).await?;
        let mut decompressed_term_dictionary: Vec<u8> = Vec::new();
        let mut decompressor: Decoder<'_, BufReader<&[u8]>> =
            Decoder::new(&compressed_term_dictionary[..])?;
        decompressor.read_to_end(&mut decompressed_term_dictionary)?;
        let cursor = Cursor::new(decompressed_term_dictionary);
        let buf_reader: BufReader<Cursor<Vec<u8>>> = BufReader::new(cursor);

        reader
            .seek(SeekFrom::Start(compressed_plist_offsets_offset))
            .await?;
        let mut buffer2: Vec<u8> =
            vec![0u8; (file_size - compressed_plist_offsets_offset - 16) as usize];
        reader.read(&mut buffer2).await?;
        decompressor = Decoder::new(&buffer2[..])?;
        let mut decompressed_serialized_plist_offsets: Vec<u8> =
            Vec::with_capacity(buffer2.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist_offsets)?;
        let this_plist_offsets: Vec<u64> =
            bincode::deserialize(&decompressed_serialized_plist_offsets)
                .map_err(|e| anyhow!(LavaError::from(e)))?;

        plist_offsets.push(this_plist_offsets);
        decompressed_term_dictionaries.push(buf_reader);
        file_sizes.push(file_size);
        readers.push(reader);
    }

    // now do the merge sort

    let mut term_dictionary = String::new();
    let mut current_lines: Vec<Option<String>> = vec![None; decompressed_term_dictionaries.len()];
    let mut plist_cursor: Vec<u64> = vec![0; decompressed_term_dictionaries.len()];

    // Initialize the current line for each reader
    for (i, reader) in decompressed_term_dictionaries.iter_mut().enumerate() {
        if let Some(Ok(line)) = reader.lines().next() {
            current_lines[i] = Some(line);
        }
    }

    let mut output_file = File::create(condensed_lava_file)?;
    let mut new_plist_offsets = Vec::new();
    new_plist_offsets.push(0);

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

        let mut plist: Vec<u64> = Vec::new();

        for i in 0..current_lines.len() {
            if current_lines[i].is_some() {
                let line = current_lines[i].as_ref().unwrap();
                if line.eq(&smallest_line) {
                    // we need to read and decompress the plists
                    let offset = plist_offsets[i][plist_cursor[i] as usize];
                    let size = plist_offsets[i][plist_cursor[i] as usize + 1]
                        - plist_offsets[i][plist_cursor[i] as usize];
                    let file_reader = &mut readers[i];
                    file_reader.seek(SeekFrom::Start(offset as u64)).await?;
                    let mut compressed_plist = vec![0u8; size as usize];
                    file_reader.read(&mut compressed_plist).await?;
                    let mut decompressor = Decoder::new(&compressed_plist[..])?;
                    let mut decompressed_serialized_plist: Vec<u8> =
                        Vec::with_capacity(compressed_plist.len() as usize);
                    decompressor.read_to_end(&mut decompressed_serialized_plist)?;
                    let mut this_plist: Vec<u64> =
                        bincode::deserialize(&decompressed_serialized_plist)
                            .map_err(|e| anyhow!(LavaError::from(e)))?;

                    plist.append(&mut this_plist);

                    plist_cursor[i] += 1;

                    let reader = &mut decompressed_term_dictionaries[i];
                    if let Some(Ok(line)) = reader.lines().next() {
                        current_lines[i] = Some(line);
                    } else {
                        current_lines[i] = None;
                    }
                }
            }
        }

        // print out plist
        let serialized = bincode::serialize(&plist).unwrap();
        let compressed_plist = encode_all(&serialized[..], 0).expect("Compression failed");
        new_plist_offsets
            .push(new_plist_offsets[new_plist_offsets.len() - 1] + compressed_plist.len() as u64);
        output_file.write_all(&compressed_plist)?;
    }

    println!("{:?}", term_dictionary);
    let bytes = term_dictionary.as_bytes();
    let compressed_term_dictionary = encode_all(bytes, 0).expect("Compression failed");
    let compressed_term_dict_offset = output_file.seek(SeekFrom::Current(0))?;

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

pub fn merge_lava(condensed_lava_file: Cow<str>, lava_files: Vec<Cow<str>>) -> Result<()> {
    // you should only merge them on local disk. It's not worth random accessing S3 for this because of the request costs.
    // worry about running out of disk later. Assume you have a fast SSD for now.
    let mut builder = Fs::default();
    let current_path = env::current_dir()?;
    builder.root(current_path.to_str().expect("no path"));
    let mut operator = Operator::new(builder)
        .map_err(|e| anyhow!(LavaError::from(e)))?
        .finish();

    let result = hoa(condensed_lava_file.as_ref(), &mut operator, lava_files);

    result
}
