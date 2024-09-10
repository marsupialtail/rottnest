// use log::{info, warn};
// use parquet::{column::reader, format::DictionaryPageHeader};
// use rand::Rng;
// use tokio::{task::JoinSet, time::sleep};

// use crate::{
//     formats::readers::{
//         get_file_size_and_reader, get_file_sizes_and_readers, AsyncReader, ClonableAsyncReader, ReaderType,
//     },
//     lava::error::LavaError,
//     lava::logcloud_plist::{PListChunk, PlistSize},
// };
// use serde::de::DeserializeOwned;
// use std::{
//     collections::{HashMap, HashSet},
//     io::Read,
//     time::{Duration, Instant},
// };
// use zstd::stream::{encode_all, read::Decoder};

// async fn read_and_decompress<T>(reader: &mut AsyncReader, start: u64, size: u64) -> Result<T, LavaError>
// where
//     T: DeserializeOwned,
// {
//     let compressed = reader.read_range(start, start + size).await?;
//     let mut decompressor = Decoder::new(&compressed[..]).unwrap();
//     let mut decompressed = Vec::new();
//     std::io::copy(&mut decompressor, &mut decompressed)?;
//     let result: T = bincode::deserialize(&decompressed)?;
//     Ok(result)
// }

// pub struct LogCloud {
//     pub kauai: AsyncReader,
//     pub oahu: AsyncReader,
//     pub hawaii: AsyncReader,
// }

// // std::pair<int, std::vector<plist_size_t>> search_kauai(VirtualFileRegion * vfr, std::string query, int k) {

// async fn query_kauai(
//     reader: &mut AsyncReader,
//     file_size: usize,
//     query: &str,
//     k: u32,
// ) -> Result<(u32, Vec<PlistSize>), LavaError> {
//     let byte_offsets = reader.read_usize_from_end(6).await?;

//     let dictionary: String = read_and_decompress(reader, 0, byte_offsets[0]).await?;

//     // Read and decompress templates
//     let template: String = read_and_decompress(reader, byte_offsets[0], byte_offsets[1] - byte_offsets[0]).await?;

//     // Read template posting lists
//     let template_plist: Vec<Vec<PlistSize>> =
//         read_and_decompress(reader, byte_offsets[1], byte_offsets[2] - byte_offsets[1]).await?;

//     // Read and decompress outlier strings
//     let outlier: String = read_and_decompress(reader, byte_offsets[2], byte_offsets[3] - byte_offsets[2]).await?;

//     // Read outlier posting lists
//     let outlier_plist: Vec<Vec<PlistSize>> =
//         read_and_decompress(reader, byte_offsets[3], byte_offsets[4] - byte_offsets[3]).await?;

//     // Read and decompress outlier type strings
//     let outlier_type: String = read_and_decompress(reader, byte_offsets[4], byte_offsets[5] - byte_offsets[4]).await?;

//     // Read outlier type posting lists
//     let outlier_type_pl_size = file_size as u64 - byte_offsets[5] - 6 * std::mem::size_of::<usize>() as u64;
//     let outlier_type_plist: Vec<Vec<PlistSize>> =
//         read_and_decompress(reader, byte_offsets[5], outlier_type_pl_size).await?;

//     for (_, line) in dictionary.lines().enumerate() {
//         if line.contains(query) {
//             println!("query matched dictionary item, brute force {}", query);
//             return Ok((0, Vec::new()));
//         }
//     }

//     let mut matched_row_groups = Vec::new();

//     let search_text = |query: &str,
//                        source_str: &str,
//                        plists: &[Vec<PlistSize>],
//                        matched_row_groups: &mut Vec<PlistSize>,
//                        write: bool| {
//         if write {
//             println!("{}", source_str);
//         }
//         for (line_no, line) in source_str.lines().enumerate() {
//             if let Some(_) = line.find(query) {
//                 println!("{} {}", line, line_no);
//                 let posting_list = &plists[line_no];
//                 for &row_group in posting_list {
//                     print!("{} ", row_group);
//                     matched_row_groups.push(row_group);
//                 }
//                 println!();
//             }
//         }
//     };

//     search_text(query, &template, &template_plist, &mut matched_row_groups, false);

//     // Print matched row groups
//     for &row_group in &matched_row_groups {
//         print!("{} ", row_group);
//     }
//     println!();

//     search_text(query, &outlier, &outlier_plist, &mut matched_row_groups, false);

//     if matched_row_groups.len() >= k.try_into().unwrap() {
//         println!("inexact query for top K satisfied by template and outlier {}", query);
//         return Ok((1, matched_row_groups));
//     }

//     // Search in outlier types
//     search_text(query, &outlier_type, &outlier_type_plist, &mut matched_row_groups, false);

//     if matched_row_groups.len() >= k.try_into().unwrap() {
//         println!("inexact query for top K satisfied by template, outlier and outlier types {}", query);
//         Ok((1, matched_row_groups))
//     } else {
//         println!("inexact query for top K not satisfied by template, outlier and outlier types {}", query);
//         Ok((2, matched_row_groups))
//     }
// }

// async fn search_oahu(
//     reader: &mut AsyncReader,
//     file_size: usize,
//     query_type: i32,
//     chunks: Vec<usize>,
//     query_str: &str,
// ) -> Result<Vec<PlistSize>, LavaError> {
//     // Read the metadata page length
//     let metadata_page_length = reader.read_usize_from_end(1).await?[0];

//     // Read the metadata page
//     let metadata_page = reader
//         .read_range(
//             file_size as u64 - metadata_page_length as u64 - std::mem::size_of::<usize>() as u64,
//             file_size as u64 - std::mem::size_of::<usize>() as u64,
//         )
//         .await?;

//     let mut decompressor = Decoder::new(&metadata_page[..]).unwrap();
//     let mut decompressed_metadata_page: Vec<u8> = Vec::with_capacity(metadata_page.len() as usize);
//     decompressor.read_to_end(&mut decompressed_metadata_page).unwrap();

//     // Read metadata
//     let num_types = u64::from_le_bytes(decompressed_metadata_page[0..8].try_into().unwrap()) as usize;
//     let num_blocks = u64::from_le_bytes(decompressed_metadata_page[8..16].try_into().unwrap()) as usize;

//     let mut offset = 16;
//     let type_order: Vec<i32> = (0..num_types)
//         .map(|i| {
//             let start = offset + i * 8;
//             i32::from_le_bytes(decompressed_metadata_page[start..start + 8].try_into().unwrap())
//         })
//         .collect();

//     offset += num_types * 8;
//     let type_offsets: Vec<usize> = (0..=num_types)
//         .map(|i| {
//             let start = offset + i * 8;
//             u64::from_le_bytes(decompressed_metadata_page[start..start + 8].try_into().unwrap()) as usize
//         })
//         .collect();

//     offset += (num_types + 1) * 8;
//     let block_offsets: Vec<usize> = (0..=num_blocks)
//         .map(|i| {
//             let start = offset + i * 8;
//             u64::from_le_bytes(decompressed_metadata_page[start..start + 8].try_into().unwrap()) as usize
//         })
//         .collect();

//     // Find query_type in type_order
//     let type_index = type_order
//         .iter()
//         .position(|&x| x == query_type)
//         .ok_or_else(|| LavaError::Parse("Query type not found".to_string()))?;

//     let type_offset = type_offsets[type_index];
//     let num_chunks = type_offsets[type_index + 1] - type_offset;

//     // Process blocks using JoinSet
//     let mut set = JoinSet::new();

//     for chunk in chunks.into_iter().take(num_chunks) {
//         let block_offset = block_offsets[type_offset + chunk] as u64;
//         let next_block_offset = block_offsets[type_offset + chunk + 1] as u64;
//         let block_size = next_block_offset - block_offset;

//         let mut reader_clone = reader.clone(); // Assuming AsyncReader implements Clone
//         let query_str_clone = query_str.to_string();

//         set.spawn(async move {
//             let block = reader_clone.read_range(block_offset, block_offset + block_size).await.unwrap();

//             let compressed_strings_length = u64::from_le_bytes(block[0..8].try_into().unwrap()) as usize;
//             let compressed_strings = &block[8..8 + compressed_strings_length];

//             let mut decompressor = Decoder::new(compressed_strings).unwrap();
//             let mut decompressed_strings: Vec<u8> = Vec::with_capacity(compressed_strings.len() as usize);
//             decompressor.read_to_end(&mut decompressed_strings).unwrap();

//             let compressed_plist = &block[8 + compressed_strings_length..];
//             let plist = PListChunk::from_compressed(compressed_plist).unwrap();

//             let mut row_groups = Vec::new();
//             for (line_number, line) in String::from_utf8_lossy(&decompressed_strings).lines().enumerate() {
//                 if format!("\n{}\n", line).contains(&query_str_clone) {
//                     row_groups.extend(plist.lookup(line_number).unwrap());
//                 }
//             }

//             row_groups
//         });
//     }

//     let mut all_row_groups = Vec::new();
//     while let Some(result) = set.join_next().await {
//         let result = result.unwrap();
//         all_row_groups.extend(result);
//     }

//     Ok(all_row_groups)
// }

// const B: usize = 1024 * 1024;
// const GIVEUP: usize = 100;

// async fn search_vfr(
//     reader: &mut AsyncReader,
//     wavelet_offset: u64,
//     wavelet_size: u64,
//     logidx_offset: u64,
//     logidx_size: u64,
//     query: &str,
// ) -> Result<Vec<usize>, LavaError> {
//     let start_time = Instant::now();
//     let compressed_offsets_byte_offset: usize = read_and_decompress(reader, logidx_offset + logidx_size - 8, 8).await?;
//     let duration = start_time.elapsed();
//     println!(
//         "log_idx decompress offsets took {} milliseconds, this could choke for concurrent requests!",
//         duration.as_millis()
//     );

//     let compressed_offsets: Vec<u8> = read_and_decompress(
//         reader,
//         logidx_offset + compressed_offsets_byte_offset as u64,
//         (logidx_size - compressed_offsets_byte_offset as u64 - 8) as u64,
//     )
//     .await?;

//     let chunk_offsets: Vec<usize> = bincode::deserialize(&compressed_offsets)?;

//     async fn batch_log_idx_lookup(
//         chunk_offsets: &[usize],
//         reader: &mut AsyncReader,
//         logidx_offset: u64,
//         start_idx: usize,
//         end_idx: usize,
//     ) -> Result<Vec<usize>, LavaError> {
//         let start_chunk_offset = chunk_offsets[start_idx / B];
//         let end_chunk_offset = chunk_offsets[end_idx / B + 1];
//         let total_chunks = end_idx / B - start_idx / B + 1;

//         let compressed_chunks: Vec<u8> = read_and_decompress(
//             reader,
//             logidx_offset + start_chunk_offset as u64,
//             (end_chunk_offset - start_chunk_offset) as u64,
//         )
//         .await?;

//         let mut results = Vec::new();
//         for i in 0..total_chunks {
//             let chunk_start = chunk_offsets[start_idx / B + i] - start_chunk_offset;
//             let chunk_end = chunk_offsets[start_idx / B + i + 1] - start_chunk_offset;
//             let log_idx: Vec<usize> = bincode::deserialize(&compressed_chunks[chunk_start..chunk_end])?;

//             let start = if i == 0 { start_idx % B } else { 0 };
//             let end = if i == total_chunks - 1 { end_idx % B } else { log_idx.len() };

//             results.extend_from_slice(&log_idx[start..end]);
//         }

//         Ok(results)
//     }

//     let (start, end) = search_wavelet_tree_file(reader, wavelet_offset, wavelet_size query).await?;

//     if start == -1 || end == -1 {
//         info!("no matches");
//         return Ok(vec![usize::MAX]);
//     }

//     let start = start as usize;
//     let end = end as usize;

//     if false {
//         // (end - start > GIVEUP) {
//         warn!("too many matches, giving up");
//         Ok(vec![usize::MAX])
//     } else {
//         let matched_pos = batch_log_idx_lookup(&chunk_offsets, log_idx_reader, start, end).await?;

//         info!("start: {}", start);
//         info!("end: {}", end);

//         Ok(matched_pos)
//     }
// }

// async fn search_hawaii(
//     reader: &mut AsyncReader,
//     file_size: usize,
//     types: Vec<i32>,
//     query: String,
// ) -> Result<HashMap<i32, HashSet<usize>>, LavaError> {
//     // Read metadata page size
//     let metadata_page_length = reader.read_usize_from_end(1).await?[0];

//     // Read and decompress metadata page
//     let decompressed_metadata_page: Vec<u8> = read_and_decompress(
//         reader,
//         file_size as u64 - metadata_page_length as u64 - std::mem::size_of::<usize>() as u64,
//         metadata_page_length as u64,
//     )
//     .await?;

//     // Parse metadata
//     let mut offset = 0;
//     let num_types = u64::from_le_bytes(decompressed_metadata_page[offset..offset + 8].try_into().unwrap()) as usize;
//     offset += 8;
//     println!("num types: {}", num_types);

//     let num_groups = u64::from_le_bytes(decompressed_metadata_page[offset..offset + 8].try_into().unwrap()) as usize;
//     offset += 8;
//     println!("num groups: {}", num_groups);

//     let type_order: Vec<i32> = (0..num_types)
//         .map(|i| {
//             let start = offset + i * 8;
//             let type_value = i32::from_le_bytes(decompressed_metadata_page[start..start + 8].try_into().unwrap());
//             println!("type order: {}", type_value);
//             type_value
//         })
//         .collect();
//     offset += num_types * 8;

//     let chunks_in_group: Vec<usize> = (0..num_types)
//         .map(|i| {
//             let start = offset + i * 8;
//             let chunks = usize::from_le_bytes(decompressed_metadata_page[start..start + 8].try_into().unwrap());
//             println!("chunks in group: {}", chunks);
//             chunks
//         })
//         .collect();
//     offset += num_types * 8;

//     let type_offsets: Vec<usize> = (0..=num_types)
//         .map(|i| {
//             let start = offset + i * 8;
//             let type_offset = usize::from_le_bytes(decompressed_metadata_page[start..start + 8].try_into().unwrap());
//             println!("type offsets: {}", type_offset);
//             type_offset
//         })
//         .collect();
//     offset += (num_types + 1) * 8;

//     let group_offsets: Vec<usize> = (0..num_groups * 2 + 1)
//         .map(|i| {
//             let start = offset + i * 8;
//             let group_offset = usize::from_le_bytes(decompressed_metadata_page[start..start + 8].try_into().unwrap());
//             println!("group offsets: {}", group_offset);
//             group_offset
//         })
//         .collect();

//     let mut set = JoinSet::new();
//     for &type_value in &types {
//         let reader_clone = reader.clone(); // Assuming AsyncReader implements Clone
//         let query_clone = query.clone();
//         let type_order_clone = type_order.clone();
//         let chunks_in_group_clone = chunks_in_group.clone();
//         let type_offsets_clone = type_offsets.clone();
//         let group_offsets_clone = group_offsets.clone();

//         set.spawn(async move {
//             let type_index = type_order.iter().position(|&x| x == type_value).unwrap_or(num_types);

//             if type_index == num_types {
//                 return Ok((type_value, HashSet::from([usize::MAX])));
//             }

//             let chunks_in_group_for_type = chunks_in_group[type_index];
//             let type_offset = type_offsets[type_index];
//             let num_iters = type_offsets[type_index + 1] - type_offsets[type_index];

//             println!("searching wavelet tree {} {}", type_value, num_iters);

//             let mut chunks = HashSet::new();
//             for i in (type_offset..type_offset + num_iters).step_by(2) {
//                 // Random delay
//                 sleep(Duration::from_millis(rand::thread_rng().gen_range(0..1000))).await;

//                 let group_id = (i - type_offset) / 2;
//                 let group_chunk_offset = group_id * chunks_in_group_for_type;

//                 let wavelet_offset = group_offsets[i];
//                 let logidx_offset = group_offsets[i + 1];
//                 let next_wavelet_offset = group_offsets[i + 2];
//                 let wavelet_size = logidx_offset - wavelet_offset;
//                 let logidx_size = next_wavelet_offset - logidx_offset;

//                 let matched_pos = search_vfr(reader, wavelet_offset, logidx_offset, &query)?;

//                 for pos in matched_pos {
//                     chunks.insert(group_chunk_offset + pos);
//                 }
//             }

//             Ok((type_value, chunks))
//         });
//     }

//     let mut type_chunks = HashMap::new();
//     while let Some(result) = set.join_next().await {
//         match result {
//             Ok(Ok((type_value, chunks))) => {
//                 type_chunks.insert(type_value, chunks);
//             }
//             Ok(Err(e)) => eprintln!("Error processing type: {:?}", e),
//             Err(e) => eprintln!("Task join error: {:?}", e),
//         }
//     }

//     Ok(type_chunks)
// }

// impl LogCloud {
//     pub fn new(kauai: AsyncReader, oahu: AsyncReader, hawaii: AsyncReader) -> Self {
//         Self { kauai, oahu, hawaii }
//     }
// }
