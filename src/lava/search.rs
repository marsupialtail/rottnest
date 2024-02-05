use opendal::{services::S3, Operator};
use pyo3::{pyfunction, PyResult};
use regex::Regex;

use std::io::{BufRead, BufReader, Cursor, Read, SeekFrom};

use zstd::stream::read::Decoder;

use anyhow::Result;
use opendal::raw::oio::ReadExt;
use opendal::services::Fs;

use opendal::Reader;
use std::env;

#[tokio::main]
async fn search_lava_async(operator: &mut Operator, file: &str, query: &str) -> Result<Vec<u64>> {
    let file_size: u64 = operator.stat(file).await?.content_length();
    let mut reader: Reader = operator.clone().reader(file).await?;

    reader.seek(SeekFrom::End(-16)).await?;
    let mut buffer1 = [0u8; 8];
    reader.read(&mut buffer1).await?;
    let compressed_term_dictionary_offset = u64::from_le_bytes(buffer1);

    let mut buffer: [u8; 8] = [0u8; 8];
    reader.read(&mut buffer[..]).await?;
    let compressed_plist_offsets_offset: u64 = u64::from_le_bytes(buffer);

    // now read the term dictionary
    let mut compressed_term_dictionary: Vec<u8> =
        vec![0u8; (compressed_plist_offsets_offset - compressed_term_dictionary_offset) as usize];
    reader
        .seek(SeekFrom::Start(compressed_term_dictionary_offset))
        .await?;
    reader.read(&mut compressed_term_dictionary[..]).await?;

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
    let plist_offsets: Vec<u64> = bincode::deserialize(&decompressed_serialized_plist_offsets)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Bincode deserialization error: {}", e))
        })?;

    // now read the plist offsets that you need, whose indices are in matched

    let mut plist_result: Vec<u64> = Vec::new();
    for i in matched {
        reader
            .seek(SeekFrom::Start(plist_offsets[i as usize]))
            .await?;
        let mut buffer3: Vec<u8> =
            vec![0u8; (plist_offsets[(i + 1) as usize] - plist_offsets[i as usize]) as usize];
        reader.read(&mut buffer3).await?;
        decompressor = Decoder::new(&buffer3[..])?;
        let mut decompressed_serialized_plist: Vec<u8> = Vec::with_capacity(buffer3.len() as usize);
        decompressor.read_to_end(&mut decompressed_serialized_plist)?;
        let mut plist: Vec<u64> =
            bincode::deserialize(&decompressed_serialized_plist).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Bincode deserialization error: {}",
                    e
                ))
            })?;
        plist_result.append(&mut plist);
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

#[pyfunction]
pub fn search_lava(file: &str, query: &str) -> PyResult<Vec<u64>> {
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
    let result: Result<Vec<u64>, anyhow::Error> =
        search_lava_async(&mut operator, &filename, query);

    Ok(result
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("searching error: {}", e)))?)
}
