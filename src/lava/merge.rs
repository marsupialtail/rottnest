use async_recursion::async_recursion;

use itertools::Itertools;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use crate::formats::readers::ReaderType;

use crate::lava::bm25::merge_lava_bm25;
use crate::lava::error::LavaError;
use crate::lava::substring::merge_lava_substring;
use crate::lava::substring::merge_lava_substring_char;
use crate::lava::uuid::merge_lava_uuid;

// @Rain chore: we need to simplify all the iterator impls

#[async_recursion]
async fn async_parallel_merge_files(
    condensed_lava_file: String,
    files: Vec<String>,
    do_not_delete: BTreeSet<String>,
    uid_offsets: Vec<u64>,
    k: usize,
    mode: usize, // 0 for bm25 1 for substring 2 for uuid 3 for substring_char
    reader_type: ReaderType,
    cache_ranges: Option<Vec<Vec<(usize, usize)>>>,
) -> Result<Vec<(usize, usize)>, LavaError> {
    assert!(mode == 0 || mode == 1 || mode == 2 || mode == 3);
    if mode == 1 || mode == 2 || mode == 3 {
        assert_eq!(k, 2);
    }

    match files.len() {
        0 => Err(LavaError::Parse("out of chunks".to_string())), // Assuming LavaError can be constructed like this
        1 => {
            // the recursion will end here in this case. rename the files[0] to the supposed output name
            std::fs::rename(files[0].clone(), condensed_lava_file).unwrap();
            let mut cache_ranges = cache_ranges.unwrap();
            assert!(cache_ranges.len() == 1);
            Ok(cache_ranges.remove(0))
        }
        _ => {
            // More than one file, need to merge
            let mut tasks = vec![];
            let merged_files_shared = Arc::new(Mutex::new(vec![]));
            let new_uid_offsets_shared = Arc::new(Mutex::new(vec![]));

            let chunked_files: Vec<Vec<String>> = files
                .into_iter()
                .chunks(k)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            let chunked_uid_offsets: Vec<Vec<u64>> = uid_offsets
                .into_iter()
                .chunks(k)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            for (file_chunk, uid_chunk) in chunked_files
                .into_iter()
                .zip(chunked_uid_offsets.into_iter())
            {
                if file_chunk.len() == 1 {
                    // If there's an odd file out, directly move it to the next level
                    merged_files_shared
                        .lock()
                        .unwrap()
                        .push(file_chunk[0].clone());
                    new_uid_offsets_shared
                        .lock()
                        .unwrap()
                        .push(uid_chunk[0].clone());
                    continue;
                }

                let merged_files_clone = Arc::clone(&merged_files_shared);
                let new_uid_offsets_clone = Arc::clone(&new_uid_offsets_shared);
                let do_not_delete_clone = do_not_delete.clone();
                let reader_type = reader_type.clone();

                let task: tokio::task::JoinHandle<Vec<(usize, usize)>> = tokio::spawn(async move {
                    let my_uuid = uuid::Uuid::new_v4();
                    let merged_filename = my_uuid.to_string(); // Define this function based on your requirements

                    println!("mergin {:?}", file_chunk);

                    let cache_ranges: Vec<(usize, usize)> = match mode {
                        0 => {
                            merge_lava_bm25(
                                &merged_filename,
                                file_chunk.to_vec(),
                                uid_chunk.to_vec(),
                                reader_type.clone(),
                            )
                            .await
                        }
                        1 => {
                            merge_lava_substring(
                                &merged_filename,
                                file_chunk.to_vec(),
                                uid_chunk.to_vec(),
                                reader_type.clone(),
                            )
                            .await
                        }
                        2 => {
                            merge_lava_uuid(
                                &merged_filename,
                                file_chunk.to_vec(),
                                uid_chunk.to_vec(),
                                reader_type.clone(),
                            )
                            .await
                        }
                        3 => {
                            merge_lava_substring_char(
                                &merged_filename,
                                file_chunk.to_vec(),
                                uid_chunk.to_vec(),
                                reader_type.clone(),
                            )
                            .await
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();

                    // now go delete the input files

                    for file in file_chunk {
                        if !do_not_delete_clone.contains(&file) {
                            println!("deleting {}", file);
                            std::fs::remove_file(file).unwrap();
                        }
                    }

                    // no race condition since everybody pushes the same value to new_uid_offsets_clone
                    merged_files_clone.lock().unwrap().push(merged_filename);
                    new_uid_offsets_clone.lock().unwrap().push(0);
                    cache_ranges
                });

                tasks.push(task);
            }

            // Wait for all tasks to complete, MUST BE IN ORDER due to cache_ranges!
            let cache_ranges: Vec<Vec<(usize, usize)>> = futures::future::join_all(tasks)
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

            // Extract the merged files for the next level of merging
            let merged_files: Vec<String> = Arc::try_unwrap(merged_files_shared)
                .expect("Lock still has multiple owners")
                .into_inner()
                .unwrap();

            let new_uid_offsets = Arc::try_unwrap(new_uid_offsets_shared)
                .expect("Lock still has multiple owners")
                .into_inner()
                .unwrap();

            // Recurse with the newly merged files
            async_parallel_merge_files(
                condensed_lava_file,
                merged_files,
                do_not_delete,
                new_uid_offsets,
                k,
                mode,
                reader_type.clone(),
                Some(cache_ranges),
            )
            .await
        }
    }
}

#[tokio::main]
pub async fn parallel_merge_files(
    condensed_lava_file: String,
    files: Vec<String>,
    uid_offsets: Vec<u64>,
    k: usize,
    mode: usize, // 0 for bm25 1 for substring 2 for uuid 3 for substring_char
    reader_type: ReaderType,
) -> Result<Vec<(usize, usize)>, LavaError> {
    let do_not_delete = BTreeSet::from_iter(files.clone().into_iter());
    let result = async_parallel_merge_files(
        condensed_lava_file,
        files,
        do_not_delete,
        uid_offsets,
        k,
        mode,
        reader_type,
        None,
    )
    .await?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use crate::{formats::readers::ReaderType, lava::merge::parallel_merge_files};

    #[test]
    pub fn test_merge_lava_bm25() {
        let res = parallel_merge_files(
            "merged.lava".to_string(),
            vec!["bump0.lava".to_string(), "bump1.lava".to_string()],
            vec![0, 1000000],
            2,
            0,
            ReaderType::default(),
        );

        println!("{:?}", res);
    }

    #[test]
    pub fn test_merge_lava_substring() {
        let res = parallel_merge_files(
            "merged.lava".to_string(),
            vec![
                "chinese_index/0.lava".to_string(),
                "chinese_index/1.lava".to_string(),
            ],
            vec![0, 1000000],
            2,
            1,
            ReaderType::default(),
        );

        println!("{:?}", res);
    }
}
