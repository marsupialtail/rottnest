pub mod formats;
pub mod lava;
pub mod vamana;
use formats::io::{get_file_sizes_and_readers, AsyncReader, READER_BUFFER_SIZE};
use rand::{thread_rng, Rng};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let PAGE_SIZE: u64 = &args[3].parse::<u64>()? * 1024;
    let TOTAL_ITERATIONS: u64 = args[4].parse::<u64>()?;
    let input = &args[1];
    let filenames: Vec<String> = (0..input.parse::<i32>()?)
        .map(|i| {
            format!(
                "part-00{:03}-b658c136-5501-4a68-a1c8-7e09e87944ba-c000.snappy.parquet",
                i
            )
        })
        .collect();

    let files = filenames
        .iter()
        .map(|filename| format!("{}/{}", &args[2], filename))
        .collect::<Vec<_>>();

    let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
    let mut join_set = JoinSet::new();

    let start = Instant::now();
    for (file_size, mut reader) in file_sizes.into_iter().zip(readers.into_iter()) {
        join_set.spawn(async move {
            let mut i = 0;
            let count = TOTAL_ITERATIONS;

            let sleep_time = thread_rng().gen_range(0..10);
            std::thread::sleep(Duration::from_millis(sleep_time));

            // println!("thread id {:?}", std::thread::current().id());
            while i < count {
                i += 1;
                let from = thread_rng().gen_range(0..(file_size as u64 - PAGE_SIZE));
                let to = from + PAGE_SIZE;
                let res = reader.read_range(from, to).await.unwrap();
                println!("Read {} bytes from {}", res.len(), reader.filename);
            }
        });
    }

    while let Some(_) = join_set.join_next().await {
        // println!("Task completed");
    }
    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

    Ok(())
}
