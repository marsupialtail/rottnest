use bytes::{Bytes, BytesMut};
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use aws_sdk_s3::Client;

use crate::lava::error::LavaError;


pub struct AsyncAwsReader {
    reader: Client,
    pub bucket: String,
    pub filename: String,
    pub file_size: u64,
}

impl Deref for AsyncAwsReader {
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl DerefMut for AsyncAwsReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.reader
    }
}

impl AsyncAwsReader {
    pub fn new(reader: Client, bucket: String, filename: String) -> Self {
        Self {
            reader,
            bucket,
            filename,
            file_size: 0,
        }
    }

    async fn stat(&self) -> Result<u64, LavaError> {
        let (bucket, filename) = (&self.bucket, &self.filename);
        self.head_object()
            .bucket(bucket)
            .key(filename)
            .send()
            .await
            .map_err(|e| LavaError::AwsSdk(e.to_string()))
            .map(|res| match res.content_length() {
                Some(size) if size > 0 => size as u64,
                _ => 0,
            })
    }
}

#[async_trait]
impl super::Reader for AsyncAwsReader {
    async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
        }

        let total = to - from;
        let mut res = BytesMut::with_capacity(total as usize);
        let (bucket, filename) = (&self.bucket, &self.filename);

        let mut object = self
                    .get_object()
                    .bucket(bucket)
                    .key(filename)
                    .set_range(Some(format!("bytes={}-{}", from, to).to_string()))
                    .send()
                    .await
                    .map_err(|e| LavaError::AwsSdk(e.to_string()))?;
        
        while let Some(chunk) = object.body.try_next().await.map_err(|e| LavaError::AwsSdk(e.to_string()))? {
            res.extend_from_slice(&chunk);
        }

        if res.len() < total as usize {
            return Err(LavaError::Io(std::io::ErrorKind::Interrupted.into()));
        }

        Ok(res.freeze())
    }

    async fn read_usize_from_end(&mut self, offset: i64, n: u64) -> Result<Vec<u64>, LavaError> {
        let mut result: Vec<u64> = vec![];
        let from = self.file_size as i64 + offset;
        let to = from + (n as i64) * 8;
        let bytes = self.read_range(from as u64, to as u64).await?;
        bytes.chunks_exact(8).for_each(|chunk| {
            result.push(u64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]));
        });
        Ok(result)
    }

    async fn read_usize_from_start(&mut self, offset: u64, n: u64) -> Result<Vec<u64>, LavaError> {
        let mut result: Vec<u64> = vec![];
        let from = offset as i64;
        let to = from + (n as i64) * 8;
        let bytes = self.read_range(from as u64, to as u64).await?;
        bytes.chunks_exact(8).for_each(|chunk| {
            result.push(u64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]));
        });
        Ok(result)
    }
}

#[derive(Clone)]
pub struct Config(aws_config::SdkConfig);

impl Config {
    pub async fn from_env() -> Self {
        let config = aws_config::load_from_env().await;
        Config(config)
    }
}

#[derive(Clone)]
pub struct Operator(aws_sdk_s3::Client);

impl From<Config> for Operator {
    fn from(config: Config) -> Self {
        Operator(aws_sdk_s3::Client::new(&config.0))
    }
}

impl Operator {
    fn into_inner(self) -> aws_sdk_s3::Client {
        self.0
    }
}

pub async fn get_file_size_and_reader(
    file: String,
) -> Result<(usize, AsyncAwsReader), LavaError> {
    // Extract filename
    if !file.starts_with("s3://") {
        return Err(LavaError::Parse("File scheme not supported".to_string()));
    }

    let config = Config::from_env().await;
    let operator = Operator::from(config);

    let tokens = file[5..].split('/').collect::<Vec<_>>();
    let bucket = tokens[0].to_string();
    let filename = tokens[1..].join("/");

    // Create the reader
    let mut reader = AsyncAwsReader::new(operator.into_inner(), bucket.clone(), filename.clone());

    // Get the file size
    let file_size = reader.stat().await?;

    if file_size == 0 {
        return Err(LavaError::Parse("File size is zero".to_string()));
    }
    reader.file_size = file_size;

    Ok((file_size as usize, reader))
}

pub async fn get_file_sizes_and_readers(
    files: &[String],
) -> Result<(Vec<usize>, Vec<AsyncAwsReader>), LavaError> {
    let tasks: Vec<_> = files
        .iter()
        .map(|file| {
            let file = file.clone(); // Clone file name to move into the async block
            tokio::spawn(async move {
                get_file_size_and_reader(file).await
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
