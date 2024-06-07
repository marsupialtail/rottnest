use crate::{
    formats::readers::{get_file_size_and_reader, get_reader, AsyncReader, ReaderType},
    lava::error::LavaError,
};
use std::collections::BTreeMap;
use std::io::Write;

#[tokio::main]
pub async fn populate_cache(
    ranges: BTreeMap<String, Vec<(usize, usize)>>,
    cache_dir: &str,
    reader_type: ReaderType
) -> Result<(), LavaError> {

    
    for (file_path, ranges) in &ranges {
        let (_, mut reader) = get_file_size_and_reader(file_path.to_string(), reader_type.clone()).await?;
        let path = std::path::Path::new(cache_dir);
        let cache_file = path.join(&file_path.split("/").last().unwrap());
        let path = cache_file.with_extension("cache");
        println!("looking in cache: {}", path.display());
        // check if exists
        if ! path.exists() {
            println!("writing to cache: {}", path.display());
            let mut regions: BTreeMap<(usize, usize), Vec<u8>> = BTreeMap::new();
            for (from, to) in ranges {
                let data = reader.read_range(*from as u64, *to as u64).await?;
                regions.insert((*from, *to), data.to_vec());
            }

            let mut file = std::fs::File::create(path)?;
            let bytes = bincode::serialize(&regions)?;
            file.write_all(&bytes)?;
        }
        
    }
    Ok(())

}