use async_trait::async_trait;
use bytes::Bytes;
use reqwest::{self, Client};

use std::ops::{Deref, DerefMut};

use crate::lava::error::LavaError;

#[derive(Clone)]
pub struct AsyncHttpReader {
    reader: Client,
    pub url: String,
    pub file_size: u64,
}

impl Deref for AsyncHttpReader {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl DerefMut for AsyncHttpReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.reader
    }
}

impl AsyncHttpReader {
    pub fn new(reader: Client, url: String) -> Self {
        Self {
            reader,
            url,
            file_size: 0,
        }
    }

    async fn stat(&self) -> Result<u64, LavaError> {
        let response = self.head(&self.url).send().await?;

        // Checking the response status
        let length: u64 = if response.status().is_success() {
            // Retrieving the Content-Length header which indicates the size of the file
            if let Some(content_length) = response.headers().get(reqwest::header::CONTENT_LENGTH) {
                content_length.to_str().unwrap().parse().unwrap()
            } else {
                panic!("Content-Length header is missing.");
            }
        } else {
            panic!("Failed to retrieve headers, status: {}", response.status());
        };

        Ok(length)
    }
}

#[async_trait]
impl super::Reader for AsyncHttpReader {
    async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
        }

        let url = &self.url;
        let response = self
            .get(url)
            .header("Range", format!("bytes={}-{}", from, to - 1))
            .send()
            .await?;

        let content: Bytes = if response.status().is_success() {
            response.bytes().await?
        } else {
            panic!("Failed to retrieve content, status: {}", response.status());
        };

        Ok(content)
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

pub(crate) async fn get_reader(url: String) -> Result<(usize, AsyncHttpReader), LavaError> {
    // Determine the operator based on the file scheme
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(LavaError::Parse("File scheme not supported".to_string()));
    }

    let mut reader = AsyncHttpReader::new(reqwest::Client::new(), url);
    let file_size = reader.stat().await?;
    if file_size == 0 {
        return Err(LavaError::Parse("File size is zero".to_string()));
    }
    reader.file_size = file_size;

    Ok((file_size as usize, reader))
}
