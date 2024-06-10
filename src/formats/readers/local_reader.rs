use std::{
    io::SeekFrom,
    ops::{Deref, DerefMut}
};

use async_trait::async_trait;
use bytes::Bytes;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
};

use crate::lava::error::LavaError;

pub struct AsyncLocalReader {
    reader: File,
    pub file_size: u64,
    pub filename: String,
}

impl Deref for AsyncLocalReader {
    type Target = File;
    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl DerefMut for AsyncLocalReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.reader
    }
}

impl Clone for AsyncLocalReader {
    fn clone(&self) -> Self {
        let file = std::fs::File::open(self.filename.clone()).unwrap();
        let reader = File::from_std(file);
        Self {
            reader,
            filename: self.filename.clone(),
            file_size: self.file_size,
        }
    }
}

impl AsyncLocalReader {
    pub fn new(reader: File, filename: String) -> Self {
        Self {
            reader,
            filename,
            file_size: 0,
        }
    }

    async fn stat(&self) -> Result<u64, LavaError> {
        self.metadata()
            .await
            .map_err(|e| LavaError::Io(e))
            .map(|m| m.len())
    }
}

#[async_trait]
impl super::Reader for AsyncLocalReader {
    async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
        }

        let mut buffer = vec![0; (to - from) as usize];
        self.seek(SeekFrom::Start(from))
            .await
            .map_err(|e| LavaError::Io(e))?;
        self.read_exact(&mut buffer)
            .await
            .map_err(|e| LavaError::Io(e))?;

        Ok(Bytes::from(buffer))
    }

    async fn read_usize_from_end(&mut self, offset: i64, n: u64) -> Result<Vec<u64>, LavaError> {
        let mut result: Vec<u64> = vec![];
        let from = self.file_size as i64 + offset;
        let to = from + (n as i64) * 8;
        let bytes = self.read_range(from as u64, to as u64).await?;
        bytes.chunks_exact(8).for_each(|chunk| {
            result.push(u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        });
        Ok(result)
    }

    async fn read_usize_from_start(&mut self, offset: u64, n: u64) -> Result<Vec<u64>, LavaError> {
        let mut result = vec![];
        let from = offset as i64;
        let to = from + (n as i64) * 8;
        let bytes = self.read_range(from as u64, to as u64).await?;
        bytes.chunks_exact(8).for_each(|chunk| {
            result.push(u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        });
        Ok(result)
    }
}

pub(crate) async fn get_reader(filename: String) -> Result<(usize, AsyncLocalReader), LavaError> {
    let file = File::open(filename.clone()).await.map_err(|e| LavaError::Io(e))?;
    let mut reader = AsyncLocalReader::new(file, filename);
    let file_size = reader.stat().await?;

    if file_size == 0 {
        return Err(LavaError::Parse("File size is zero".to_string()));
    }
    reader.file_size = file_size;

    Ok((file_size as usize, reader))
}
