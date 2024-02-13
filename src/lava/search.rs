use opendal::{
    services::{Fs, S3},
    Operator,
};
use regex::Regex;

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Cursor, Read, SeekFrom},
};

use zstd::stream::read::Decoder;

use opendal::raw::oio::ReadExt;

use opendal::Reader;
use std::env;

use crate::lava::error::LavaError;
use crate::lava::plist::PList;

#[tokio::main]
async fn search_lava_async(
    operator: &mut Operator,
    file: &str,
    query: &str,
) -> Result<Vec<u64>, LavaError> {
    let file_size: u64 = operator.stat(file).await?.content_length();
    let mut reader: Reader = operator.clone().reader(file).await?;

    reader.seek(SeekFrom::End(-16)).await?;
    let mut buffer1 = [0u8; 8];
    reader.read(&mut buffer1).await?;
    let compressed_term_dictionary_offset = u64::from_le_bytes(buffer1);

    let mut buffer: [u8; 8] = [0u8; 8];
    reader.read(&mut buffer[..]).await?;
    let compressed_plist_offsets_offset: u64 = u64::from_le_bytes(buffer);
    // println!("{}", compressed_plist_offsets_offset);

    // now read the term dictionary
    let total_len = (compressed_plist_offsets_offset - compressed_term_dictionary_offset) as usize;
    let mut compressed_term_dictionary: Vec<u8> = vec![0u8; total_len];
    let mut current = 0;
    while current < total_len {
        reader
            .seek(SeekFrom::Start(
                compressed_term_dictionary_offset + current as u64,
            ))
            .await?;
        let size = reader
            .read(&mut compressed_term_dictionary[current..])
            .await?;
        current += size;
    }

    let mut decompressed_term_dictionary: Vec<u8> = Vec::new();
    let mut decompressor: Decoder<'_, BufReader<&[u8]>> =
        Decoder::new(&compressed_term_dictionary[..])?;
    decompressor.read_to_end(&mut decompressed_term_dictionary)?;

    let cursor = Cursor::new(decompressed_term_dictionary);
    let buf_reader: BufReader<Cursor<Vec<u8>>> = BufReader::new(cursor);
    let mut matched: Vec<u64> = Vec::new();
    let re: Regex = Regex::new(query).unwrap();

    let mut counter: u64 = 0;
    for line in buf_reader.lines() {
        let line = line?; // Handle potential errors on each line
        if re.is_match(&line) {
            matched.push(counter);
        }
        counter += 1;
    }

    if matched.len() == 0 {
        return Ok(matched);
    }

    // seek to the offset
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
    let plist_offsets: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist_offsets)?;

    // plist_offsets is the byte offsets of the chunks followed by the cum count of the items in each plist chunk
    if plist_offsets.len() % 2 != 0 {
        let err = LavaError::Parse("data corruption".to_string());
        return Err(err);
    }

    let num_chunks: usize = plist_offsets.len() / 2;
    let term_dict_len: &[u64] = &plist_offsets[num_chunks..];

    let mut plist_result: Vec<u64> = Vec::new();
    let mut chunks_to_search: HashMap<usize, Vec<u64>> = HashMap::new();
    for i in matched {
        let (idx, offset) = match term_dict_len.binary_search(&i) {
            Ok(idx) => (idx, 0),
            Err(idx) => (idx - 1, i - term_dict_len[idx - 1]),
        };

        chunks_to_search
            .entry(idx)
            .or_insert_with(Vec::new)
            .push(offset as u64);
    }

    for (idx, offsets) in chunks_to_search.into_iter() {
        reader
            .seek(SeekFrom::Start(plist_offsets[idx as usize]))
            .await?;
        let mut buffer3: Vec<u8> =
            vec![0u8; (plist_offsets[(idx + 1) as usize] - plist_offsets[idx as usize]) as usize];
        reader.read(&mut buffer3).await?;
        let mut result: Vec<u64> = PList::search_compressed(buffer3, offsets)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        plist_result.append(&mut result);
    }

    Ok(plist_result)
}

struct S3Builder(S3);

impl From<&str> for S3Builder {
    fn from(file: &str) -> Self {
        let mut builder = S3::default();
        let mut iter = file[5..].split("/");

        builder.bucket(iter.next().expect("malformed path"));
        // Set the region. This is required for some services, if you don't care about it, for example Minio service, just set it to "auto", it will be ignored.
        builder.region("us-west-2");
        builder.enable_virtual_host_style();
        builder.endpoint("");
        S3Builder(builder)
    }
}

struct FsBuilder(Fs);

impl From<&str> for FsBuilder {
    fn from(folder: &str) -> Self {
        let mut builder = Fs::default();
        // let current_path = env::current_dir().expect("no path");
        builder.root(folder);
        FsBuilder(builder)
    }
}

struct Operators(Operator);

impl From<S3Builder> for Operators {
    fn from(builder: S3Builder) -> Self {
        Operators(
            Operator::new(builder.0)
                .expect("S3 builder construction error")
                .finish(),
        )
    }
}

impl Operators {
    fn into_inner(self) -> Operator {
        self.0
    }
}

impl From<FsBuilder> for Operators {
    fn from(builder: FsBuilder) -> Self {
        Operators(
            Operator::new(builder.0)
                .expect("Fs Builder construction error")
                .finish(),
        )
    }
}

pub fn search_lava(file: &str, query: &str) -> Result<Vec<u64>, LavaError> {
    let mut operator = if file.starts_with("s3://") {
        Operators::from(S3Builder::from(file)).into_inner()
    } else {
        let current_path = env::current_dir()?;
        Operators::from(FsBuilder::from(current_path.to_str().expect("no path"))).into_inner()
    };

    let filename = if file.starts_with("s3://") {
        file[5..].split("/").collect::<Vec<&str>>().join("/")
    } else {
        file.to_string()
    };

    println!("Searching {}", filename);
    search_lava_async(&mut operator, &filename, query)
}

#[cfg(test)]
mod tests {
    use super::search_lava;

    #[test]
    pub fn test_search_lava() {
        let file = "content_split.lava";
        let query = "helsinki";

        let res = search_lava(file, query).unwrap();

        println!("{:?}", res);
    }
}
