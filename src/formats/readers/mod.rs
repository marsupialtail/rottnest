use async_trait::async_trait;
use bytes::Bytes;
use std::{
    io::Read,
    ops::{Deref, DerefMut},
};
use zstd::stream::read::Decoder;

use crate::lava::error::LavaError;

mod opendal_reader;
mod aws_reader;

#[async_trait]
pub trait Reader : Send + Sync {
    async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError>;
    async fn read_usize_from_end(&mut self, offset: i64, n: u64) -> Result<Vec<u64>, LavaError>;
    async fn read_usize_from_start(&mut self, offset: u64, n: u64) -> Result<Vec<u64>, LavaError>;
}

pub const READER_BUFFER_SIZE: usize = 4 * 1024 * 1024;
pub const WRITER_BUFFER_SIZE: usize = 4 * 1024 * 1024;

pub struct AsyncReader {
    reader: Box<dyn Reader>,
    pub filename: String,
}

impl Deref for AsyncReader {
    type Target = dyn Reader;

    fn deref(&self) -> &Self::Target {
        self.reader.as_ref()
    }
}

impl DerefMut for AsyncReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.reader.as_mut()
    }
}

impl AsyncReader {
    pub fn new(reader: Box<dyn Reader>, filename: String) -> Self {
        Self { reader, filename }
    }

    pub async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
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

#[derive(Debug, Clone, Default)]
pub enum ReaderType {
    #[default]
    Opendal,
    AwsSdk,
}

impl From<String> for ReaderType {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "opendal" => ReaderType::Opendal,
            "aws" => ReaderType::AwsSdk,
            _ => Default::default(),
        }
    }
}

pub async fn get_file_sizes_and_readers(
    files: &[String], reader_type: ReaderType
) -> Result<(Vec<usize>, Vec<AsyncReader>), LavaError> {
    let tasks: Vec<_> = files
        .iter()
        .map(|file| {
            let file = file.clone();
            let reader_type = reader_type.clone();
            tokio::spawn(async move {
                get_file_size_and_reader(file, reader_type).await
            })
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
            Err(e) => return Err(LavaError::Parse(format!("Task join error: {}", e.to_string()))), // Handle join error
        }
    }

    Ok((file_sizes, readers))
}

pub async fn get_file_size_and_reader(
    file: String, reader_type: ReaderType
) -> Result<(usize, AsyncReader), LavaError> {

    // always choose opendal for none s3 file
    let reader_type = if file.starts_with("s3://") {
        reader_type
    } else {
        Default::default()
    };

    let (file_size, reader) = match reader_type {
        ReaderType::Opendal => {
            let (file_size, reader) = opendal_reader::get_reader(file).await?;
            let filename = reader.filename.clone();
            let reader = AsyncReader::new(Box::new(reader), filename);
            (file_size, reader)
        }
        ReaderType::AwsSdk => {
            let (file_size, reader) = aws_reader::get_reader(file).await?;
            let filename = reader.filename.clone();
            let async_reader = AsyncReader::new(Box::new(reader), filename);
            (file_size, async_reader)
        }
    };
    

    Ok((file_size, reader))
}