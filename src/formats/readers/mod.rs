use crate::formats::cache;
use crate::lava::error::LavaError;
use async_trait::async_trait;
use bytes::Bytes;
use local_reader::AsyncLocalReader;
use std::collections::BTreeMap;
use std::{env, os};
use std::{
    io::Read,
    ops::{Deref, DerefMut},
};
use zstd::stream::read::Decoder;

use self::{aws_reader::AsyncAwsReader, http_reader::AsyncHttpReader};
mod aws_reader;
mod http_reader;
mod local_reader;

#[async_trait]
pub trait Reader: Send + Sync {
    fn update_filename(&mut self, filename: String) -> Result<(), LavaError>;
    async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError>;
    async fn read_usize_from_end(&mut self, offset: i64, n: u64) -> Result<Vec<u64>, LavaError>;
    async fn read_usize_from_start(&mut self, offset: u64, n: u64) -> Result<Vec<u64>, LavaError>;
}

pub const READER_BUFFER_SIZE: usize = 4 * 1024 * 1024;
pub const WRITER_BUFFER_SIZE: usize = 4 * 1024 * 1024;

pub struct AsyncReader {
    pub reader: ClonableAsyncReader,
    pub filename: String,
}

impl Deref for AsyncReader {
    type Target = dyn Reader;

    fn deref(&self) -> &Self::Target {
        &*self.reader
    }
}

impl DerefMut for AsyncReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.reader
    }
}

impl Clone for AsyncReader {
    fn clone(&self) -> Self {
        Self {
            reader: match &self.reader {
                ClonableAsyncReader::Local(reader) => ClonableAsyncReader::Local(reader.clone()),
                ClonableAsyncReader::AwsSdk(reader) => ClonableAsyncReader::AwsSdk(reader.clone()),
                ClonableAsyncReader::Http(reader) => ClonableAsyncReader::Http(reader.clone()),
            },
            filename: self.filename.clone(),
        }
    }
}

pub enum ClonableAsyncReader {
    Local(AsyncLocalReader),
    AwsSdk(AsyncAwsReader),
    Http(AsyncHttpReader),
}

impl Deref for ClonableAsyncReader {
    type Target = dyn Reader;

    fn deref(&self) -> &Self::Target {
        match self {
            ClonableAsyncReader::Local(reader) => reader,
            ClonableAsyncReader::AwsSdk(reader) => reader,
            ClonableAsyncReader::Http(reader) => reader,
        }
    }
}

impl DerefMut for ClonableAsyncReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            ClonableAsyncReader::Local(reader) => reader,
            ClonableAsyncReader::AwsSdk(reader) => reader,
            ClonableAsyncReader::Http(reader) => reader,
        }
    }
}

impl AsyncReader {
    pub fn new(reader: ClonableAsyncReader, filename: String) -> Self {
        Self { reader, filename }
    }

    pub fn update_filename(&mut self, filename: String) -> Result<(), LavaError> {
        self.deref_mut().update_filename(filename)
    }

    pub async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
        }

        // only check the cache if self.filename has extension .lava
        if self.filename.ends_with(".lava") {
            if "true"
                == env::var_os("CACHE_ENABLE")
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default()
            {
                // let path = std::path::Path::new(&value);
                // find path/filename.cache
                let mut conn = cache::get_redis_connection().await?;
                println!("looking in cache: {}", self.filename);
                let ranges = conn.get_ranges(&self.filename).await?;

                // see if this exists
                if ranges.len() > 0 {
                    for ((start, end)) in ranges {
                        if from >= start as u64 && to <= end as u64 {
                            println!("cache hit");
                            let data = conn.get_data(&self.filename, from, to).await?;
                            let data = data
                                [(from - start as u64) as usize..(to - start as u64) as usize]
                                .to_vec();
                            return Ok(Bytes::from(data));
                        }
                    }
                }
            }
        }

        self.deref_mut().read_range(from, to).await
    }

    // theoretically we should try to return different types here, but Vec<u64> is def. the most common
    pub async fn read_range_and_decompress(
        &mut self,
        from: u64,
        to: u64,
    ) -> Result<Vec<u64>, LavaError> {
        let compressed_posting_list_offsets = self.read_range(from, to).await?;
        let mut decompressor = Decoder::new(&compressed_posting_list_offsets[..])?;
        let mut serialized_posting_list_offsets: Vec<u8> =
            Vec::with_capacity(compressed_posting_list_offsets.len() as usize);
        decompressor.read_to_end(&mut serialized_posting_list_offsets)?;
        let result: Vec<u64> = bincode::deserialize(&serialized_posting_list_offsets)?;
        Ok(result)
    }

    pub async fn read_usize_from_end(&mut self, n: u64) -> Result<Vec<u64>, LavaError> {
        self.deref_mut()
            .read_usize_from_end(-8 * (n as i64), n)
            .await
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ReaderType {
    #[default]
    Local,
    AwsSdk,
    Http,
}

impl From<String> for ReaderType {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "local" => ReaderType::Local,
            "aws" => ReaderType::AwsSdk,
            "http" => ReaderType::Http,
            _ => Default::default(),
        }
    }
}

pub async fn get_file_sizes_and_readers(
    files: &[String],
    reader_type: ReaderType,
) -> Result<(Vec<usize>, Vec<AsyncReader>), LavaError> {
    let tasks: Vec<_> = files
        .iter()
        .map(|file| {
            let file = file.clone();
            let reader_type = reader_type.clone();
            tokio::spawn(async move { get_file_size_and_reader(file, reader_type).await })
        })
        .collect();

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;

    // Process results, separating out file sizes and readers
    let mut file_sizes = Vec::new();
    let mut readers = Vec::new();

    for result in results {
        match result {
            Ok(Ok((size, reader))) => {
                file_sizes.push(size);
                readers.push(reader);
            }
            Ok(Err(e)) => return Err(e), // Handle error from inner task
            Err(e) => {
                return Err(LavaError::Parse(format!(
                    "Task join error: {}",
                    e.to_string()
                )))
            } // Handle join error
        }
    }

    Ok((file_sizes, readers))
}

pub async fn get_readers(
    files: &[String],
    reader_type: ReaderType,
) -> Result<Vec<AsyncReader>, LavaError> {
    let tasks: Vec<_> = files
        .iter()
        .map(|file| {
            let file = file.clone();
            let reader_type = reader_type.clone();
            tokio::spawn(async move { get_reader(file, reader_type).await })
        })
        .collect();

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;

    // Process results, separating out file sizes and readers
    let mut readers = Vec::new();

    for result in results {
        match result {
            Ok(Ok(reader)) => {
                readers.push(reader);
            }
            Ok(Err(e)) => return Err(e), // Handle error from inner task
            Err(e) => {
                return Err(LavaError::Parse(format!(
                    "Task join error: {}",
                    e.to_string()
                )))
            } // Handle join error
        }
    }

    Ok(readers)
}

pub async fn get_file_size_and_reader(
    file: String,
    reader_type: ReaderType,
) -> Result<(usize, AsyncReader), LavaError> {
    // always choose opendal for none s3 file
    let reader_type = if file.starts_with("http://") || file.starts_with("https://") {
        ReaderType::Http
    } else if file.starts_with("s3://") {
        ReaderType::AwsSdk
    } else {
        Default::default()
    };

    let (file_size, reader) = match reader_type {
        ReaderType::Local => {
            let (file_size, reader) = local_reader::get_reader(file).await?;
            let filename = reader.filename.clone();
            let reader = AsyncReader::new(ClonableAsyncReader::Local(reader), filename);
            (file_size, reader)
        }
        ReaderType::AwsSdk => {
            let (file_size, reader) = aws_reader::get_file_size_and_reader(file).await?;
            let filename = reader.filename.clone();
            let async_reader = AsyncReader::new(ClonableAsyncReader::AwsSdk(reader), filename);
            (file_size, async_reader)
        }
        ReaderType::Http => {
            let (file_size, reader) = http_reader::get_reader(file).await?;
            let filename = reader.url.clone();
            let async_reader = AsyncReader::new(ClonableAsyncReader::Http(reader), filename);
            (file_size, async_reader)
        }
    };

    Ok((file_size, reader))
}

pub async fn get_reader(file: String, reader_type: ReaderType) -> Result<AsyncReader, LavaError> {
    // always choose opendal for none s3 file
    let reader_type = if file.starts_with("http://") || file.starts_with("https://") {
        ReaderType::Http
    } else if file.starts_with("s3://") {
        ReaderType::AwsSdk
    } else {
        Default::default()
    };

    let reader = match reader_type {
        ReaderType::Local => {
            let (_file_size, reader) = local_reader::get_reader(file).await?;
            let filename = reader.filename.clone();
            let reader = AsyncReader::new(ClonableAsyncReader::Local(reader), filename);
            reader
        }
        ReaderType::AwsSdk => {
            let reader = aws_reader::get_reader(file).await?;
            let filename = reader.filename.clone();
            let async_reader = AsyncReader::new(ClonableAsyncReader::AwsSdk(reader), filename);
            async_reader
        }
        ReaderType::Http => {
            let (_file_size, reader) = http_reader::get_reader(file).await?;
            let filename = reader.url.clone();
            let async_reader = AsyncReader::new(ClonableAsyncReader::Http(reader), filename);
            async_reader
        }
    };

    Ok(reader)
}
