use crate::{
    formats::readers::{
        get_file_size_and_reader, get_file_sizes_and_readers, get_reader, get_readers, AsyncReader,
        ClonableAsyncReader, ReaderType,
    },
    lava::error::LavaError,
};
use log::info;
use rand::distributions::Alphanumeric;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::vec::Vec;
use zstd::stream::{decode_all, encode_all};
const ALPHABET: usize = 256;
const LOG_ALPHABET: usize = 8;
const CHUNK_BITS: usize = 32768;

pub(crate) type Bitvector = Vec<bool>;
pub(crate) type WaveletTree = Vec<Bitvector>;

fn bitvector_rank(bitvector: &Bitvector, bit: bool, pos: usize) -> usize {
    bitvector.iter().take(pos).filter(|&&b| b == bit).count()
}

fn wavelet_tree_rank(tree: &WaveletTree, c: u8, pos: usize) -> usize {
    let mut counter = 0;
    let mut curr_pos = pos;

    for i in 0..LOG_ALPHABET {
        let bit = (c >> (LOG_ALPHABET - 1 - i)) & 1 == 1;
        let bitvector = &tree[counter];
        curr_pos = bitvector_rank(bitvector, bit, curr_pos);
        counter = counter * 2 + if bit { 1 } else { 0 } + 1;
    }

    curr_pos
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| chunk.iter().enumerate().fold(0, |acc, (i, &bit)| acc | ((bit as u8) << (7 - i))))
        .collect()
}

pub(crate) fn write_wavelet_tree_to_disk(
    tree: &WaveletTree,
    file: &mut File,
) -> std::io::Result<(Vec<usize>, Vec<usize>)> {
    let mut total_length = 0;
    let mut offsets = vec![0];
    let mut level_offsets = vec![0];

    for (i, bitvector) in tree.iter().enumerate() {
        if bitvector.is_empty() {
            level_offsets.push(*level_offsets.last().unwrap());
            continue;
        }

        let mut rank_0 = 0;
        let mut rank_1 = 0;

        info!("{}", i);

        for chunk in bitvector.chunks(CHUNK_BITS) {
            let mut packed_chunks = Vec::new();
            packed_chunks.extend_from_slice(&(rank_0 as u64).to_le_bytes());
            packed_chunks.extend_from_slice(&(rank_1 as u64).to_le_bytes());
            packed_chunks.extend(pack_bits(chunk));

            rank_0 += bitvector_rank(&chunk.to_vec(), false, chunk.len());
            rank_1 += bitvector_rank(&chunk.to_vec(), true, chunk.len());

            let compressed_chunk = encode_all(&packed_chunks[..], 0)?;
            file.write_all(&compressed_chunk)?;

            offsets.push(offsets.last().unwrap() + compressed_chunk.len());
            total_length += compressed_chunk.len();
        }
        level_offsets.push(offsets.len() - 1);
    }

    info!("{}", total_length);
    info!("number of chunks {}", offsets.len());

    let metadata: (Vec<usize>, Vec<usize>) = (offsets, level_offsets);

    Ok(metadata)
}

pub(crate) fn search_wavelet_tree(tree: &WaveletTree, c: &[usize], p: &[u8], n: usize) -> (usize, usize) {
    let mut start = 0;
    let mut end = n + 1;

    for &ch in p.iter().rev() {
        println!("c: {}", ch as char);

        start = c[ch as usize] + wavelet_tree_rank(tree, ch, start);
        end = c[ch as usize] + wavelet_tree_rank(tree, ch, end);

        println!("start: {}", start);
        println!("end: {}", end);

        if start >= end {
            println!("not found");
            return (usize::MAX, usize::MAX);
        }
    }

    println!("start: {}", start);
    println!("end: {}", end);
    println!("range: {}", end - start);
    (start, end)
}

async fn read_chunk_from_reader(
    reader: &mut AsyncReader,
    start_byte: usize,
    end_byte: usize,
) -> (usize, usize, Bitvector) {
    let compressed_chunk = reader.read_range(start_byte as u64, end_byte as u64).await.unwrap();
    let decompressed_chunk = decode_all(&compressed_chunk[..]).unwrap();

    let rank_0 = usize::from_le_bytes(decompressed_chunk[..8].try_into().unwrap());
    let rank_1 = usize::from_le_bytes(decompressed_chunk[8..16].try_into().unwrap());

    let chunk: Bitvector = decompressed_chunk[16..]
        .iter()
        .flat_map(|&byte| (0..8).map(move |i| (byte >> (7 - i)) & 1 == 1))
        .take(CHUNK_BITS)
        .collect();

    (rank_0, rank_1, chunk)
}

async fn wavelet_tree_rank_from_reader(
    reader: &mut AsyncReader,
    level_offsets: &[usize],
    offsets: &[usize],
    c: u8,
    pos: usize,
) -> std::io::Result<usize> {
    let mut curr_pos = pos;
    let mut counter = 0;

    for i in 0..LOG_ALPHABET {
        let bit = (c >> (LOG_ALPHABET - 1 - i)) & 1 == 1;
        let chunk_id = level_offsets[counter] + curr_pos / CHUNK_BITS;
        let chunk_start = offsets[chunk_id];
        let chunk_end = offsets[chunk_id + 1];

        let (rank_0, rank_1, chunk) = read_chunk_from_reader(reader, chunk_start, chunk_end).await;
        curr_pos = bitvector_rank(&chunk, bit, curr_pos % CHUNK_BITS) + if bit { rank_1 } else { rank_0 };

        counter = counter * 2 + if bit { 1 } else { 0 } + 1;
    }

    Ok(curr_pos)
}

pub(crate) async fn search_wavelet_tree_from_reader(
    reader: &mut AsyncReader,
    p: &[u8],
    n: usize,
    offsets: &[usize],
    level_offsets: &[usize],
    c: &[usize],
) -> std::io::Result<(usize, usize)> {
    let mut start = 0;
    let mut end = n + 1;
    let mut previous_range = usize::MAX;

    for &ch in p.iter().rev() {
        println!("c: {}", ch as char);

        start = c[ch as usize] + wavelet_tree_rank_from_reader(reader, &level_offsets, &offsets, ch, start).await?;
        end = c[ch as usize] + wavelet_tree_rank_from_reader(reader, &level_offsets, &offsets, ch, end).await?;

        println!("start: {}", start);
        println!("end: {}", end);
        println!("range: {}", end - start);

        if start >= end {
            info!("not found");
            return Ok((usize::MAX, usize::MAX));
        }

        if end - start == previous_range {
            info!("early exit");
            return Ok((start, end));
        }

        previous_range = end - start;
    }

    Ok((start, end))
}

pub(crate) fn construct_wavelet_tree(p: &[u8]) -> WaveletTree {
    let mut tree = vec![Bitvector::new(); ALPHABET];

    for &c in p {
        let mut counter = 0;
        tree[counter].push((c >> (LOG_ALPHABET - 1)) & 1 == 1);

        for i in 0..(LOG_ALPHABET - 1) {
            let bit = (c >> (LOG_ALPHABET - 1 - i)) & 1 == 1;
            counter = counter * 2 + if bit { 1 } else { 0 } + 1;
            tree[counter].push((c >> (LOG_ALPHABET - 2 - i)) & 1 == 1);
        }
    }

    tree
}
