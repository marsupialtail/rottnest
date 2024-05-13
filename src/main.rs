pub mod formats;
pub mod lava;
pub mod vamana;
use formats::readers::get_file_sizes_and_readers;
use formats::readers::ReaderType;
use rand::{thread_rng, Rng};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    // let args = vec!["", "2", "example_data", "1", "10"];
    let page_size: u64 = &args[3].parse::<u64>()? * 1024;
    let total_iterations: u64 = args[4].parse::<u64>()?;
    let input = &args[1];
    let filenames: Vec<String> = (0..input.parse::<i32>()?)
        .map(|i| {
            format!(
                "{}.parquet",
                i
            )
        })
        .collect();

    let files = filenames
        .iter()
        .map(|filename| format!("{}/{}", &args[2], filename))
        .collect::<Vec<_>>();

    let (file_sizes, readers) = get_file_sizes_and_readers(&files, ReaderType::Opendal).await?;

    // let result = read_indexed_pages(
    //     column_name: String,
    //     file_paths: Vec<String>,
    //     row_groups: Vec<usize>,
    //     page_offsets: Vec<u64>,
    //     page_sizes: Vec<usize>,
    //     dict_page_sizes: Vec<usize>, // 0 means no dict page
    //     ReaderType::AwsSdk
    // );


    let mut join_set = JoinSet::new();

    let start = Instant::now();
    for (file_size, mut reader) in file_sizes.into_iter().zip(readers.into_iter()) {
        join_set.spawn(async move {
            let mut i = 0;
            let count = total_iterations;

            let sleep_time = thread_rng().gen_range(0..10);
            std::thread::sleep(Duration::from_millis(sleep_time));

            // println!("thread id {:?}", std::thread::current().id());
            while i < count {
                i += 1;
                let from = thread_rng().gen_range(0..(file_size as u64 - page_size));
                let to = from + page_size;
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
