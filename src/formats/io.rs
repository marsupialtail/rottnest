use bytes::{Bytes, BytesMut};
use opendal::services::{Fs, S3};
use opendal::Operator;
use opendal::Reader;
use std::env;
use std::io::{Read, SeekFrom};
use std::ops::{Deref, DerefMut};
use tokio::pin;
use zstd::stream::read::Decoder;

use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::lava::error::LavaError;

pub const READER_BUFFER_SIZE: usize = 4 * 1024 * 1024;
pub const WRITER_BUFFER_SIZE: usize = 4 * 1024 * 1024;

pub struct AsyncReader {
    reader: Reader,
}

impl Deref for AsyncReader {
    type Target = Reader;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

impl DerefMut for AsyncReader {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.reader
    }
}

impl From<Reader> for AsyncReader {
    fn from(reader: Reader) -> Self {
        Self::new(reader)
    }
}

impl AsyncReader {
    pub fn new(reader: Reader) -> Self {
        Self { reader }
    }

    pub async fn read_range(&mut self, from: u64, to: u64) -> Result<Bytes, LavaError> {
        if from >= to {
            return Err(LavaError::Io(std::io::ErrorKind::InvalidData.into()));
        }

        let reader = self;
        pin!(reader);

        let mut current = 0;
        let total = to - from;
        let mut res = BytesMut::with_capacity(total as usize);

        while current < total {
            let mut buffer = res.split_off(current as usize);
            reader.seek(SeekFrom::Start(from + current)).await?;
            let size = reader.read_buf(&mut buffer).await?;
            res.unsplit(buffer);
            current += size as u64;
        }

        if res.len() < total as usize {
            return Err(LavaError::Io(std::io::ErrorKind::Interrupted.into()));
        }

        Ok(res.freeze())
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
        let reader = self;
        pin!(reader);
        reader.seek(SeekFrom::End(-(n as i64 * 8))).await?;
        let mut result: Vec<u64> = vec![];
        for i in 0..n {
            result.push(reader.read_u64_le().await?);
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

pub(crate) async fn get_file_sizes_and_readers(
    files: &[String],
) -> Result<(Vec<usize>, Vec<AsyncReader>), LavaError> {
    let tasks: Vec<_> = files
        .iter()
        .map(|file| {
            let file = file.clone(); // Clone file name to move into the async block
            tokio::spawn(async move {
                // Determine the operator based on the file scheme
                let operator = if file.starts_with("s3://") {
                    Operators::from(S3Builder::from(file.as_str())).into_inner()
                } else {
                    let current_path = env::current_dir()?;
                    Operators::from(FsBuilder::from(current_path.to_str().expect("no path")))
                        .into_inner()
                };

                // Extract filename
                let filename = if file.starts_with("s3://") {
                    file[5..].split('/').collect::<Vec<_>>()[1..].join("/")
                } else {
                    file.clone()
                };

                // Create the reader
                let reader: AsyncReader = operator
                    .clone()
                    .reader_with(&filename)
                    .buffer(READER_BUFFER_SIZE)
                    .await?
                    .into();

                // Get the file size
                let file_size: u64 = operator.stat(&filename).await?.content_length();

                Ok::<_, LavaError>((file_size as usize, reader))
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
            Err(e) => return Err(LavaError::Parse("Task join error: {}".to_string())), // Handle join error
        }
    }

    Ok((file_sizes, readers))
}
