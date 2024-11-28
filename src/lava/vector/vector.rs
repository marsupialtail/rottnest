use crate::{
    formats::readers::{get_file_sizes_and_readers, get_reader, ReaderType},
    lava::error::LavaError,
};
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use futures::stream::{FuturesUnordered, StreamExt};
use ndarray::{concatenate, stack, Array1, Array2, Axis};
use std::cmp::Ordering;
use std::io::{self, Cursor, Read};
use std::time::Instant;
use zstd::stream::Decoder;
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let mut vec = Vec::with_capacity(bytes.len() / 4);
    let mut i = 0;
    while i < bytes.len() {
        let value = LittleEndian::read_f32(&bytes[i..i + 4]);
        vec.push(value);
        i += 4;
    }
    vec
}
async fn search_lava_vector_async(
    files: Vec<String>,
    query: Vec<f32>,
    nprobes: usize,
    reader_type: ReaderType,
) -> Result<(Vec<usize>, Vec<Array1<u8>>, Vec<(usize, Array1<u8>)>), LavaError> {
    let start = Instant::now();

    let (_, mut readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;

    let mut futures = Vec::new();

    for _ in 0..readers.len() {
        let mut reader = readers.remove(0);

        futures.push(tokio::spawn(async move {
            let results = reader.read_usize_from_end(4).await.unwrap();

            let centroid_vectors_compressed_bytes =
                reader.read_range(results[2], results[3]).await.unwrap();

            // decompress them
            let mut decompressor =
                Decoder::new(centroid_vectors_compressed_bytes.as_ref()).unwrap();
            let mut centroid_vectors: Vec<u8> =
                Vec::with_capacity(centroid_vectors_compressed_bytes.len() as usize);
            decompressor.read_to_end(&mut centroid_vectors).unwrap();

            let centroid_vectors = bytes_to_f32_vec(&centroid_vectors);
            let num_vectors = centroid_vectors.len() / 128;
            let array2 =
                Array2::<f32>::from_shape_vec((num_vectors, 128), centroid_vectors).unwrap();

            (num_vectors, array2)
        }));
    }

    let result: Vec<Result<(usize, Array2<f32>), tokio::task::JoinError>> =
        futures::future::join_all(futures).await;

    let end = Instant::now();
    println!("Time stage 1 read: {:?}", end - start);

    let start = Instant::now();

    let arr_lens = result
        .iter()
        .map(|x| x.as_ref().unwrap().0)
        .collect::<Vec<_>>();
    // get cumulative arr len starting from 0
    let cumsum = arr_lens
        .iter()
        .scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    let arrays: Vec<Array2<f32>> = result.into_iter().map(|x| x.unwrap().1).collect();
    let centroids = concatenate(
        Axis(0),
        arrays
            .iter()
            .map(|array| array.view())
            .collect::<Vec<_>>()
            .as_slice(),
    )
    .unwrap();
    let query = Array1::<f32>::from_vec(query);
    let query_broadcast = query.broadcast(centroids.dim()).unwrap();

    let difference = &centroids - &query_broadcast;
    let norms = difference.map_axis(Axis(1), |row| row.dot(&row).sqrt());
    let mut indices_and_values: Vec<(usize, f32)> = norms
        .iter()
        .enumerate()
        .map(|(idx, &val)| (idx, val))
        .collect();

    indices_and_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let smallest_indices: Vec<usize> = indices_and_values
        .iter()
        .map(|&(idx, _)| idx)
        .take(nprobes)
        .collect();

    let mut file_indices: Vec<Vec<usize>> = vec![vec![]; files.len()];
    for idx in smallest_indices.iter() {
        // figure out which file idx based on cumsum. need to find the index of the thing that is just bigger than idx

        let file_idx = cumsum
            .iter()
            .enumerate()
            .find(|(_, &val)| val > *idx)
            .unwrap()
            .0;
        let last_cumsum = if file_idx == 0 {
            0
        } else {
            cumsum[file_idx - 1]
        };
        let remainder = idx - last_cumsum;
        file_indices[file_idx].push(remainder);
    }

    let end = Instant::now();
    println!("Time math: {:?}", end - start);

    let start = Instant::now();

    let (_, mut readers) = get_file_sizes_and_readers(&files, reader_type.clone()).await?;

    let mut file_ids = vec![];
    let mut futures = Vec::new();

    for file_id in 0..readers.len() {
        let mut reader = readers.remove(0);
        if file_indices[file_id].len() == 0 {
            continue;
        }
        let my_idx: Vec<usize> = file_indices[file_id].clone();
        file_ids.push(file_id);

        futures.push(tokio::spawn(async move {
            let results = reader.read_usize_from_end(4).await.unwrap();

            let pq_bytes = reader.read_range(results[0], results[1]).await.unwrap();

            let compressed_centroid_offset_bytes =
                reader.read_range(results[1], results[2]).await.unwrap();
            let mut decompressor = Decoder::new(compressed_centroid_offset_bytes.as_ref()).unwrap();
            let mut centroid_offsets_bytes: Vec<u8> =
                Vec::with_capacity(compressed_centroid_offset_bytes.len() as usize);
            decompressor
                .read_to_end(&mut centroid_offsets_bytes)
                .unwrap();

            // now reinterpret centroid_offsets_bytes as a Vec<u64>

            let mut centroid_offsets = Vec::with_capacity(centroid_offsets_bytes.len() / 8);
            let mut cursor = Cursor::new(centroid_offsets_bytes);

            while cursor.position() < cursor.get_ref().len() as u64 {
                let value = cursor.read_u64::<LittleEndian>().unwrap();
                centroid_offsets.push(value);
            }

            let mut this_result: Vec<(usize, u64, u64)> = vec![];

            for idx in my_idx.iter() {
                this_result.push((file_id, centroid_offsets[*idx], centroid_offsets[*idx + 1]));
            }
            (this_result, Array1::<u8>::from_vec(pq_bytes.to_vec()))
        }));
    }

    let result: Vec<Result<(Vec<(usize, u64, u64)>, Array1<u8>), tokio::task::JoinError>> =
        futures::future::join_all(futures).await;
    let result: Vec<(Vec<(usize, u64, u64)>, Array1<u8>)> =
        result.into_iter().map(|x| x.unwrap()).collect();

    let pq_bytes: Vec<Array1<u8>> = result.iter().map(|x| x.1.clone()).collect::<Vec<_>>();

    let end = Instant::now();
    println!("Time stage 2 read: {:?}", end - start);

    let start = Instant::now();
    let reader = get_reader(files[file_ids[0]].clone(), reader_type.clone())
        .await
        .unwrap();

    let mut futures = FuturesUnordered::new();
    for i in 0..result.len() {
        let to_read = result[i].0.clone();
        for (file_id, start, end) in to_read.into_iter() {
            let mut reader_c = reader.clone();
            reader_c.update_filename(files[file_id].clone()).unwrap();

            futures.push(tokio::spawn(async move {
                let start_time = Instant::now();
                let codes_and_plist = reader_c.read_range(start, end).await.unwrap();
                // println!(
                //     "Time to read {:?}, {:?}",
                //     Instant::now() - start_time,
                //     codes_and_plist.len()
                // );
                (file_id, Array1::<u8>::from_vec(codes_and_plist.to_vec()))
            }));
        }
    }

    let mut ranges: Vec<(usize, Array1<u8>)> = vec![];

    while let Some(x) = futures.next().await {
        ranges.push(x.unwrap());
    }

    let end = Instant::now();
    println!("Time stage 3 read: {:?}", end - start);

    Ok((file_ids, pq_bytes, ranges))
}

pub fn search_lava_vector(
    files: Vec<String>,
    query: Vec<f32>,
    nprobes: usize,
    reader_type: ReaderType,
) -> Result<(Vec<usize>, Vec<Array1<u8>>, Vec<(usize, Array1<u8>)>), LavaError> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let res = rt.block_on(search_lava_vector_async(files, query, nprobes, reader_type));
    rt.shutdown_background();
    res
}
