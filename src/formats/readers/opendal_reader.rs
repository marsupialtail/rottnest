use async_trait::async_trait;
use bytes::Bytes;
use opendal::services::{Fs, S3};
use opendal::{FuturesAsyncReader, Operator};
use std::env;
use std::io::SeekFrom;
use std::ops::{Deref, DerefMut};
use futures::{AsyncSeekExt, AsyncReadExt};


use crate::lava::error::LavaError;

pub const READER_BUFFER_SIZE: usize = 4 * 1024 * 1024;

pub struct AsyncOpendalReader {
    reader: FuturesAsyncReader,
    pub filename: String,
}

impl Deref for AsyncOpendalReader {
    type Target = FuturesAsyncReader;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl DerefMut for AsyncOpendalReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.reader
    }
}

impl AsyncOpendalReader {
    pub fn new(reader: FuturesAsyncReader, filename: String) -> Self {
        Self { reader, filename }
    }
}

#[async_trait]
impl super::Reader for AsyncOpendalReader {
    async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
        }

        let reader = self;

        let total = to - from;
        let mut res = vec![0; total as usize];
        reader.seek(SeekFrom::Start(from)).await?;
        reader.read_exact(&mut res).await?;
        Ok(res.into())
    }

    async fn read_usize_from_end(&mut self, offset: i64, n: u64) -> Result<Vec<u64>, LavaError> {
        let reader = self;
        reader.seek(SeekFrom::End(offset)).await?;
        let mut result: Vec<u64> = vec![];
        for _print in 0..n {
            let mut buffer = vec![0; 8];
            reader.read_exact(&mut buffer).await?;
            result.push(u64::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6],
                buffer[7],
            ]));
        }
        Ok(result)
    }

    async fn read_usize_from_start(&mut self, offset: u64, n: u64) -> Result<Vec<u64>, LavaError> {
        let reader = self;
        reader.seek(SeekFrom::Start(offset as u64)).await?;
        let mut result: Vec<u64> = vec![];
        for _ in 0..n {
            let mut buffer = vec![0; 8];
            reader.read(&mut buffer).await?;
            result.push(u64::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6],
                buffer[7],
            ]));
        }
        Ok(result)
    }
}

//dupilcate code for now
pub(crate) struct S3Builder(S3);

impl From<&str> for S3Builder {
    fn from(file: &str) -> Self {
        let mut builder = S3::default();
        let mut iter = file[5..].split("/");

        builder.bucket(iter.next().expect("malformed path"));
        // Set the region. This is required for some services, if you don't care about it, for example Minio service, just set it to "auto", it will be ignored.
        if let Ok(value) = env::var("AWS_ENDPOINT_URL") {
            builder.endpoint(&value);
        }
        if let Ok(value) = env::var("AWS_REGION") {
            builder.region(&value);
        }
        if let Ok(_value) = env::var("AWS_VIRTUAL_HOST_STYLE") {
            builder.enable_virtual_host_style();
        }

        S3Builder(builder)
    }
}

pub(crate) struct FsBuilder(Fs);

impl From<&str> for FsBuilder {
    fn from(folder: &str) -> Self {
        let mut builder = Fs::default();
        // let current_path = env::current_dir().expect("no path");
        builder.root(folder);
        FsBuilder(builder)
    }
}

pub(crate) struct Operators(Operator);

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
    pub(crate) fn into_inner(self) -> Operator {
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

pub(crate) async fn get_reader(
    file: String,
) -> Result<(usize, AsyncOpendalReader), LavaError> {
    // Determine the operator based on the file scheme
    let operator = if file.starts_with("s3://") {
        Operators::from(S3Builder::from(file.as_str())).into_inner()
    } else {
        let current_path = env::current_dir()?;
        Operators::from(FsBuilder::from(current_path.to_str().expect("no path"))).into_inner()
    };

    // Extract filename
    let filename = if file.starts_with("s3://") {
        file[5..].split('/').collect::<Vec<_>>()[1..].join("/")
    } else {
        file.clone()
    };

    // Get the file size
    let file_size: u64 = operator.stat(&filename).await?.content_length();

    // Create the reader
    let reader: AsyncOpendalReader = AsyncOpendalReader::new(
        operator
            .clone()
            .reader_with(&filename)
            .chunk(READER_BUFFER_SIZE)
            .await?
            .into_futures_async_read(0..file_size as u64),
        filename.clone(),
    );

    Ok((file_size as usize, reader))
}
