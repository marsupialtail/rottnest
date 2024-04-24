use async_trait::async_trait;
use bytes::Bytes;
use std::{
    io::Read,
    ops::{Deref, DerefMut},
};
use zstd::stream::read::Decoder;

use crate::lava::error::LavaError;

#[cfg(feature = "opendal")]
mod opendal_reader;

#[cfg(feature = "aws_sdk")]
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

pub async fn get_file_sizes_and_readers(
    files: &[String],
) -> Result<(Vec<usize>, Vec<AsyncReader>), LavaError> {
    #[cfg(feature = "opendal")]
    {
        let (file_sizes, readers) = opendal_reader::get_file_sizes_and_readers(files).await?;
        let async_readers = readers
            .into_iter()
            .map(|reader| {
                let filename = reader.filename.clone();
                AsyncReader::new(Box::new(reader), filename)
            })
            .collect();

        Ok((file_sizes, async_readers))
    }

    #[cfg(feature = "aws_sdk")]
    {
        let (file_sizes, readers) = aws_reader::get_file_sizes_and_readers(files).await?;
        let async_readers = readers
            .into_iter()
            .map(|reader| {
                let filename = reader.filename.clone();
                AsyncReader::new(Box::new(reader), filename)
            })
            .collect();
        Ok((file_sizes, async_readers))
    }

    #[cfg(not(any(feature = "opendal", feature = "aws_sdk")))]
    {
        let _ = files;
        Err(LavaError::Unsupported("Must set either opendal or aws_sdk feature.".to_string()))
    }
}

pub async fn get_file_size_and_reader(
    file: String,
) -> Result<(usize, AsyncReader), LavaError> {
    #[cfg(feature = "opendal")]
    {
        let (file_size, reader) = opendal_reader::get_file_size_and_reader(file).await?;
        let filename = reader.filename.clone();
        let async_reader = AsyncReader::new(Box::new(reader), filename);
        Ok((file_size, async_reader))
    }
    #[cfg(feature = "aws_sdk")]
    {
        let (file_size, reader) = aws_reader::get_file_size_and_reader(file).await?;
        let filename = reader.filename.clone();
        let async_reader = AsyncReader::new(Box::new(reader), filename);
        Ok((file_size, async_reader))
    }

    #[cfg(not(any(feature = "opendal", feature = "aws_sdk")))]
    {
        let _ = file;
        Err(LavaError::Unsupported("Must set either opendal or aws_sdk feature.".to_string()))
    }

}