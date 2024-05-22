use polars_lazy::prelude::*;
use std::error::Error;
pub mod formats;
pub mod lava;
pub mod vamana;
// use formats::io::{get_file_sizes_and_readers, AsyncReader, READER_BUFFER_SIZE};
use rand::{thread_rng, Rng};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let num_clients = args[1].parse::<u64>()?;
    let df = LazyFrame::scan_parquet(args[3].clone(), Default::default())?
        .collect()?;
    let filenames: Vec<String> = df.column("filename")?.str()?.into_iter().map(|x| x.unwrap().to_string()).collect();
    let page_offsets: Vec<u64> = df.column("page_offset_right")?.i64()?.into_iter().map(|x| x.unwrap() as u64).collect();
    let page_sizes: Vec<u64> = df.column("page_byte_size")?.i64()?.into_iter().map(|x| x.unwrap() as u64).collect();

    let mut join_set = JoinSet::new();

    let config = aws_config::load_from_env().await;
    let client = aws_sdk_s3::Client::new(&config);

    let start = Instant::now();
    let bucket = &args[2];
    for ((filename, page_offset), page_size) in filenames.into_iter().zip(page_offsets.into_iter()).zip(page_sizes.into_iter()) {
        
        let client_c = client.clone();
        let bucket_c = bucket.to_string();
        join_set.spawn(async move {

            for i in 0 .. 1 {
                let from = page_offset;
                let to = from + page_size - 1;
                println!("Downloading {filename} from {from} to {to}");
                let mut object = client_c
                    .get_object()
                    .bucket(bucket_c.clone())
                    .key(filename.clone())
                    .set_range(Some(format!("bytes={}-{}", from, to).to_string()))
                    .send()
                    .await
                    .unwrap();

                let mut byte_count = 0_usize;
                while let Some(bytes) = object.body.try_next().await.unwrap() {
                    let bytes_len = bytes.len();
                    // file.write_all(&bytes)?;
                    // trace!("Intermediate write of {bytes_len}");
                    byte_count += bytes_len;
                }
            }

        });
    }

    while let Some(x) = join_set.join_next().await {
    }
    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

    Ok(())
}
