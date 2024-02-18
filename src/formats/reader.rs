use bytes::{Bytes, BytesMut};
use opendal::services::{Fs, S3};
use opendal::Operator;
use opendal::Reader;

use std::io::SeekFrom;
use std::ops::{Deref, DerefMut};

use tokio::pin;

use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::lava::error::LavaError;

pub const READER_BUFFER_SIZE: usize = 4 * 1024 * 1024;

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

    pub async fn read_offsets(&mut self) -> Result<(u64, u64), LavaError> {
        let reader = self;
        pin!(reader);
        reader.seek(SeekFrom::End(-16)).await?;
        let compressed_term_dictionary_offset = reader.read_u64_le().await?;
        let compressed_plist_offsets_offset = reader.read_u64_le().await?;
        Ok((
            compressed_term_dictionary_offset,
            compressed_plist_offsets_offset,
        ))
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
        builder.region("us-west-2");
        builder.enable_virtual_host_style();
        builder.endpoint("https://tos-s3-cn-beijing.volces.com");
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
