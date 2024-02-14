use bytes::{Bytes, BytesMut};
use opendal::Reader;

use std::io::SeekFrom;
use std::ops::{Deref, DerefMut};

use tokio::pin;

use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::lava::error::LavaError;

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
