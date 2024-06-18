use crate::{
    formats::readers::{get_file_size_and_reader, get_reader, AsyncReader, ReaderType},
    lava::error::LavaError,
};
use std::collections::BTreeMap;
use std::io::Write;

use super::redis_client::get_redis_connection;

#[tokio::main]
pub async fn populate_cache(
    ranges: BTreeMap<String, Vec<(usize, usize)>>,
    reader_type: ReaderType
) -> Result<(), LavaError> {

    let mut conn = get_redis_connection().await?;

    for (file_path, ranges) in &ranges {
        let (_, mut reader) = get_file_size_and_reader(file_path.to_string(), reader_type.clone()).await?;
        let cached_ranges = conn.get_ranges(&file_path).await?;
        // check if exists
        if cached_ranges.len() == 0 {
            println!("writing to cache: {}", file_path);
            for (from, to) in ranges {
                let data = reader.read_range(*from as u64, *to as u64).await?;
                conn.set_data(&file_path, *from as u64, *to as u64, data.to_vec()).await?;
            }

            conn.set_ranges(&file_path, ranges).await?;
        }
        
    }
    Ok(())

}