use bytes::Bytes;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use google_cloud_storage::client::{Client, ClientConfig};
use google_cloud_storage::http::objects::get::GetObjectRequest;
use google_cloud_storage::http::objects::download::Range;

use crate::lava::error::LavaError;

#[derive(Clone)]
pub struct AsyncGcsReader {
    reader: Client,
    pub bucket: String,
    pub filename: String,
    pub file_size: u64,
}

impl Deref for AsyncGcsReader {
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl DerefMut for AsyncGcsReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.reader
    }
}

impl AsyncGcsReader {
    pub fn new(reader: Client, bucket: String, filename: String) -> Self {
        Self {
            reader,
            bucket,
            filename,
            file_size: 0,
        }
    }

    async fn stat(&self) -> Result<u64, LavaError> {
        let object = self.reader
            .get_object(&GetObjectRequest {
                bucket: self.bucket.clone(),
                object: self.filename.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| LavaError::Gcs(format!("Failed to get object metadata: {}", e)))?;
        
        Ok(object.size as u64)
    }
}

#[async_trait]
impl super::Reader for AsyncGcsReader {
    fn update_filename(&mut self, file: String) -> Result<(), LavaError> {
        if !file.starts_with("gs://") && !file.starts_with("gcs://") {
            return Err(LavaError::Parse("File scheme not supported".to_string()));
        }

        let prefix_len = if file.starts_with("gs://") { 5 } else { 6 };
        let tokens = file[prefix_len..].split('/').collect::<Vec<_>>();
        let bucket = tokens[0].to_string();
        let filename = tokens[1..].join("/");
        self.bucket = bucket;
        self.filename = filename;

        Ok(())
    }

    async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
        }

        let total = to - from;
        
        let data = self.reader
            .download_object(&GetObjectRequest {
                bucket: self.bucket.clone(),
                object: self.filename.clone(),
                ..Default::default()
            }, &Range(Some(from), Some(to - 1)))
            .await
            .map_err(|e| LavaError::Gcs(format!("Failed to download object range: {}", e)))?;

        if data.len() < total as usize {
            return Err(LavaError::Io(std::io::ErrorKind::Interrupted.into()));
        }

        Ok(Bytes::from(data))
    }

    async fn read_usize_from_end(&mut self, offset: i64, n: u64) -> Result<Vec<u64>, LavaError> {
        let mut result: Vec<u64> = vec![];
        if self.file_size == 0 {
            panic!("file size of reader is uninitialized");
        }
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
        let mut result: Vec<u64> = vec![];
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

pub(crate) async fn get_file_size_and_reader(
    file: String,
) -> Result<(usize, AsyncGcsReader), LavaError> {
    let mut reader = get_reader(file.clone()).await?;
    let file_size = reader.stat().await?;
    if file_size == 0 {
        return Err(LavaError::Parse("File size is zero".to_string()));
    }
    reader.file_size = file_size;

    Ok((file_size as usize, reader))
}

pub(crate) async fn get_reader(file: String) -> Result<AsyncGcsReader, LavaError> {
    if !file.starts_with("gs://") && !file.starts_with("gcs://") {
        return Err(LavaError::Parse("File scheme not supported".to_string()));
    }

    // Create GCS client with default config (uses Application Default Credentials)
    let config = ClientConfig::default()
        .with_auth()
        .await
        .map_err(|e| LavaError::Gcs(format!("Failed to authenticate: {}", e)))?;
    
    let client = Client::new(config);

    let prefix_len = if file.starts_with("gs://") { 5 } else { 6 };
    let tokens = file[prefix_len..].split('/').collect::<Vec<_>>();
    let bucket = tokens[0].to_string();
    let filename = tokens[1..].join("/");

    Ok(AsyncGcsReader::new(
        client,
        bucket.clone(),
        filename.clone(),
    ))
}