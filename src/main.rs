pub mod formats;
pub mod lava;
pub mod vamana;
// use formats::io::{get_file_sizes_and_readers, AsyncReader, READER_BUFFER_SIZE};
use rand::{thread_rng, Rng};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

// #[tokio::main]
// async fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let args: Vec<String> = std::env::args().collect();
//     let PAGE_SIZE: u64 = &args[3].parse::<u64>()? * 1024;
//     let TOTAL_ITERATIONS: u64 = args[4].parse::<u64>()?;
//     let input = &args[1];
//     let filenames: Vec<String> = (0..input.parse::<i32>()?)
//         .map(|i| {
//             format!(
//                 "part-00{:03}-b658c136-5501-4a68-a1c8-7e09e87944ba-c000.snappy.parquet",
//                 i
//             )
//         })
//         .collect();

//     let files = filenames
//         .iter()
//         .map(|filename| format!("{}/{}", &args[2], filename))
//         .collect::<Vec<_>>();

//     let (file_sizes, readers) = get_file_sizes_and_readers(&files).await?;
//     let mut join_set = JoinSet::new();

//     let start = Instant::now();
//     for (file_size, mut reader) in file_sizes.into_iter().zip(readers.into_iter()) {
//         join_set.spawn(async move {
//             let mut i = 0;
//             let count = TOTAL_ITERATIONS;

//             let sleep_time = thread_rng().gen_range(0..10);
//             std::thread::sleep(Duration::from_millis(sleep_time));

//             // println!("thread id {:?}", std::thread::current().id());
//             while i < count {
//                 i += 1;
//                 let from = thread_rng().gen_range(0..(file_size as u64 - PAGE_SIZE));
//                 let to = from + PAGE_SIZE;
//                 let res = reader.read_range(from, to).await.unwrap();
//                 println!("Read {} bytes from {}", res.len(), reader.filename);
//             }
//         });
//     }

//     while let Some(_) = join_set.join_next().await {
//         // println!("Task completed");
//     }
//     let duration = start.elapsed();
//     println!("Time elapsed is: {:?}", duration);

//     Ok(())
// }

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

    let mut join_set = JoinSet::new();

    let config = aws_config::load_from_env().await;
    let client = aws_sdk_s3::Client::new(&config);

    let start = Instant::now();
    let bucket = &args[2];
    for filename in filenames.into_iter() {
        let client_c = client.clone();
        let bucket_c = bucket.to_string();
        join_set.spawn(async move {
            let mut object = client_c
                .get_object()
                .bucket(bucket_c)
                .key(filename)
                .set_range(Some("0-1000".to_string()))
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
            byte_count
        });
    }

    while let Some(x) = join_set.join_next().await {
        println!("{:?}", x.unwrap());
        // println!("Task completed");
    }
    let duration = start.elapsed();
    println!("Time elapsed is: {:?}", duration);

    Ok(())
}
