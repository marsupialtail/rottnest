use itertools::Itertools;
use log::{info, warn};
use rand::Rng;
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use tokio::{task::JoinSet, time::sleep};

use crate::{
    formats::readers::{
        get_file_size_and_reader, get_file_sizes_and_readers, get_reader, AsyncReader,
        ClonableAsyncReader, ReaderType, READ_RANGE_COUNTER,
    },
    lava::{
        error::LavaError,
        logcloud::logcloud_common::{get_all_types, get_type, PListChunk, PlistSize},
        substring::_search_lava_substring_char,
        substring::{_build_lava_substring_char, _build_lava_substring_char_wavelet},
    },
};
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::{
    collections::{HashMap, HashSet},
    sync::atomic::Ordering,
};

use std::{
    fs::{self, read_dir},
    io::{self, Read},
};
use zstd::stream::{encode_all, read::Decoder};

const BRUTE_THRESHOLD: usize = 5;
const USE_EXPERIMENTAL_NUMERICS: bool = false;

async fn read_and_decompress<T>(
    reader: &mut AsyncReader,
    start: u64,
    size: u64,
) -> Result<T, LavaError>
where
    T: DeserializeOwned,
{
    let compressed = reader.read_range(start, start + size).await?;
    let mut decompressor = Decoder::new(&compressed[..]).unwrap();
    let mut decompressed = Vec::new();
    std::io::copy(&mut decompressor, &mut decompressed)?;
    let result: T = bincode::deserialize(&decompressed)?;
    Ok(result)
}

const DICT_THRESHOLD: usize = 1000;

fn merge_files(
    input_filenames: &[String],
    input_filenames_linenumbers: &[String],
    output_filename: &str,
    output_filename_linenumbers: &str,
) -> io::Result<()> {
    let mut input_files: Vec<BufReader<File>> = Vec::new();
    let mut input_files_linenumbers: Vec<BufReader<File>> = Vec::new();

    // Open input files and input files for line numbers
    for filename in input_filenames {
        input_files.push(BufReader::new(File::open(filename)?));
    }

    for filename in input_filenames_linenumbers {
        input_files_linenumbers.push(BufReader::new(File::open(filename)?));
    }

    let mut current_lines: Vec<String> = vec![String::new(); input_files.len()];
    let mut current_linenumbers: Vec<Vec<i32>> = vec![Vec::new(); input_files_linenumbers.len()];

    let mut output_file = File::create(output_filename)?;
    let mut output_file_linenumbers = File::create(output_filename_linenumbers)?;
    let mut dict_file = File::create("compressed/compacted_type_0")?;

    // Read the first line from each file
    for (i, file) in input_files.iter_mut().enumerate() {
        file.read_line(&mut current_lines[i])?;
    }

    for (i, file) in input_files_linenumbers.iter_mut().enumerate() {
        let mut line = String::new();
        file.read_line(&mut line)?;
        current_linenumbers[i] = line
            .split_whitespace()
            .filter_map(|n| n.parse::<i32>().ok())
            .collect();
    }

    while current_lines.iter().any(|s| !s.is_empty()) {
        // Find the smallest string in `current_lines` without holding a reference to it
        let it = current_lines
            .iter()
            .filter(|s| !s.is_empty())
            .min()
            .cloned()
            .unwrap_or_else(|| String::new());

        if it.is_empty() {
            // If `it` is empty, print a warning with the current lines
            for line in &current_lines {
                eprintln!("problem: {}", line);
            }
        }

        let mut it_linenumbers: HashSet<i32> = HashSet::new();
        for (i, line) in current_lines.iter().enumerate() {
            if line == &it {
                for &num in &current_linenumbers[i] {
                    it_linenumbers.insert(num);
                }
            }
        }

        if it_linenumbers.len() > DICT_THRESHOLD {
            write!(dict_file, "{}", it)?;
        } else {
            write!(output_file, "{}", it)?;
            for num in it_linenumbers.into_iter().sorted() {
                write!(output_file_linenumbers, "{} ", num)?;
            }
            writeln!(output_file_linenumbers)?;
        }

        // Now, mutate `current_lines` after the read-only operations are done
        for (i, line) in current_lines.iter_mut().enumerate() {
            if *line == it {
                let mut next_line = String::new();
                if input_files[i].read_line(&mut next_line)? == 0 {
                    line.clear();
                    input_files[i].get_mut().sync_all()?;
                    input_files_linenumbers[i].get_mut().sync_all()?;
                } else {
                    *line = next_line;
                    let mut lineno_line = String::new();
                    input_files_linenumbers[i].read_line(&mut lineno_line)?;
                    current_linenumbers[i] = lineno_line
                        .split_whitespace()
                        .filter_map(|n| n.parse::<i32>().ok())
                        .collect();
                }
            }
        }
    }

    Ok(())
}

fn compact(num_groups: usize) -> io::Result<()> {
    // Handle outliers
    let mut input_filenames = Vec::new();
    let mut input_filenames_linenumbers = Vec::new();
    for i in 0..num_groups {
        if !Path::new(&format!("compressed/{}/outlier", i)).exists() {
            continue;
        }
        input_filenames.push(format!("compressed/{}/outlier", i));
        input_filenames_linenumbers.push(format!("compressed/{}/outlier_lineno", i));
    }

    if !input_filenames.is_empty() {
        merge_files(
            &input_filenames,
            &input_filenames_linenumbers,
            "compressed/outlier",
            "compressed/outlier_lineno",
        )?;
    }

    // Process types 1 to 63
    for type_ in 1..=63 {
        let mut input_filenames = Vec::new();
        let mut input_filenames_linenumbers = Vec::new();

        for i in 0..num_groups {
            let filename = format!("compressed/{}/compacted_type_{}", i, type_);
            if !Path::new(&filename).exists() {
                continue;
            }

            input_filenames.push(format!("compressed/{}/compacted_type_{}", i, type_));
            input_filenames_linenumbers
                .push(format!("compressed/{}/compacted_type_{}_lineno", i, type_));
        }

        if input_filenames.is_empty() {
            continue;
        }

        let output_filename = format!("compressed/compacted_type_{}", type_);
        let output_filename_linenumbers = format!("compressed/compacted_type_{}_lineno", type_);

        merge_files(
            &input_filenames,
            &input_filenames_linenumbers,
            &output_filename,
            &output_filename_linenumbers,
        )?;
    }

    println!("Files merged");
    Ok(())
}

pub fn write_kauai(filename: &str, num_groups: usize) -> std::io::Result<()> {
    let mut fp = File::create(format!("{}.kauai", filename))?;
    let mut byte_offsets = Vec::new();

    // Read and compress dictionary
    let dictionary_str = std::fs::read_to_string("compressed/compacted_type_0")?;
    // let compressed_dictionary = encode_all(&bincode::serialize(&dictionary_str.as_bytes()).unwrap()[..], 10)?;
    // fp.write_all(&compressed_dictionary)?;
    byte_offsets.push(fp.metadata()?.len());

    let mut templates = Vec::new();
    let mut template_posting_lists: Vec<Vec<PlistSize>> = Vec::new();
    let mut outliers = Vec::new();
    let mut outlier_linenos: Vec<Vec<PlistSize>> = Vec::new();

    let mut lineno: PlistSize = 0;

    println!("Reading templates...");

    for group_number in 0..num_groups {
        let mut group_template_idx = HashMap::new();
        let template_file = BufReader::new(File::open(format!(
            "compressed/{}_{}.templates",
            filename, group_number
        ))?);

        for line in template_file.lines().skip(1) {
            let line = line?;
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() >= 3 {
                let key = tokens[0][1..].parse::<usize>().unwrap();
                let value = tokens[2..].join(" ");
                templates.push(value);
                template_posting_lists.push(Vec::new());
                group_template_idx.insert(key, templates.len() - 1);
            }
        }

        let total_chunks = (0..)
            .take_while(|&chunk| {
                Path::new(&format!(
                    "compressed/{}/chunk{:04}.eid",
                    group_number, chunk
                ))
                .exists()
            })
            .count();

        for chunk in 0..total_chunks {
            println!("Reading chunk {}", chunk);
            let eid_file =
                File::open(format!("compressed/{}/chunk{:04}.eid", group_number, chunk))?;
            // let mut outlier_file = File::open(format!("compressed/{}/chunk{:04}.outlier", group_number, chunk))?;
            let mut outlier_file: Option<BufReader<File>> = if File::open(format!(
                "compressed/{}/chunk{:04}.outlier",
                group_number, chunk
            ))
            .is_ok()
            {
                Some(BufReader::new(File::open(format!(
                    "compressed/{}/chunk{:04}.outlier",
                    group_number, chunk
                ))?))
            } else {
                None
            };

            for line in BufReader::new(eid_file).lines() {
                let eid: i64 = line?.parse().unwrap();
                if eid < 0 {
                    let mut item = String::new();
                    // read the next line of the outlier file
                    if let Some(ref mut outlier_file) = outlier_file {
                        outlier_file.read_line(&mut item)?;
                    }
                    // outlier_file.read_to_string(&mut item)?;
                    outliers.push(item);
                    outlier_linenos.push(vec![lineno as u32]);
                } else {
                    let idx = if group_template_idx.contains_key(&(eid as usize)) {
                        group_template_idx[&(eid as usize)]
                    } else {
                        panic!(
                            "Template not found for eid: {}, {:?}",
                            eid, group_template_idx
                        );
                    };
                    // if template_posting_lists[idx].is_empty()
                    //     || template_posting_lists[idx].last() != Some(&(lineno as u32))
                    // {
                    //     template_posting_lists[idx].push(lineno as u32);
                    // }
                }
                lineno = lineno.wrapping_add(1);
                if lineno == 0 {
                    return Err(std::io::Error::new(std::io::ErrorKind::Other, "overflow"));
                }
            }
        }
    }

    // Write templates and template_posting_lists to files
    // {
    //     let mut template_fp = File::create("compressed/template")?;
    //     let mut template_lineno_fp = File::create("compressed/template_lineno")?;
    //     for (template, posting_list) in templates.iter().zip(&template_posting_lists) {
    //         writeln!(template_fp, "{}", template)?;
    //         for &num in posting_list {
    //             write!(template_lineno_fp, "{} ", num)?;
    //         }
    //         writeln!(template_lineno_fp)?;
    //     }
    // }

    // Remove templates with empty posting lists
    // let templates = templates
    //     .iter()
    //     .enumerate()
    //     .filter(|&(i, _)| !template_posting_lists[i].is_empty())
    //     .map(|(_, template)| template.clone())
    //     .collect::<Vec<_>>();

    // template_posting_lists.retain(|list| !list.is_empty());

    let template_str = templates.join("\n") + "\n";

    // print the length of the template str
    println!("template length: {}", template_str.len());

    let mut outlier_type_str = String::new();
    let mut outlier_type_linenos = Vec::new();
    let outlier_file = File::open("compressed/outlier")?;
    let outlier_lineno_file = File::open("compressed/outlier_lineno")?;
    for (line, outlier_type_line) in BufReader::new(outlier_lineno_file)
        .lines()
        .zip(BufReader::new(outlier_file).lines())
    {
        let line = line?;
        let outlier_type_line = outlier_type_line?;
        outlier_type_str.push_str(&outlier_type_line);
        outlier_type_str.push('\n');
        let numbers: Vec<PlistSize> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        outlier_type_linenos.push(numbers);
    }
    // print out sum of each outlier length
    println!(
        "outlier length {:?}",
        outliers.iter().map(|s| s.len()).sum::<usize>()
    );
    let outlier_str = outliers.join("") + "\n";

    let kauai_metadata: (
        String,
        String,
        Vec<Vec<u32>>,
        String,
        Vec<Vec<u32>>,
        String,
        Vec<Vec<u32>>,
    ) = (
        dictionary_str,
        template_str,
        template_posting_lists,
        outlier_str,
        outlier_linenos,
        outlier_type_str,
        outlier_type_linenos,
    );

    let compressed_metadata_page =
        encode_all(&bincode::serialize(&kauai_metadata).unwrap()[..], 10).unwrap();
    fp.write_all(&compressed_metadata_page)?;
    fp.write_all(&compressed_metadata_page.len().to_le_bytes())?;

    fp.flush()?;

    Ok(())
}

// std::pair<int, std::vector<plist_size_t>> search_kauai(VirtualFileRegion * vfr, std::string query, int k) {

async fn search_kauai(
    file_id: usize,
    mut reader: AsyncReader,
    file_size: usize,
    query: String,
    limit: u32,
) -> Result<(u32, Vec<(usize, PlistSize)>), LavaError> {
    let metadata_page_length = reader.read_usize_from_end(1).await?[0];
    // Read the metadata page
    let start_time = std::time::Instant::now();
    let metadata_page: (
        String,
        String,
        Vec<Vec<PlistSize>>,
        String,
        Vec<Vec<PlistSize>>,
        String,
        Vec<Vec<PlistSize>>,
    ) = read_and_decompress(
        &mut reader,
        file_size as u64 - metadata_page_length as u64 - 8,
        metadata_page_length as u64,
    )
    .await?;
    let (
        dictionary,
        template,
        template_plist,
        outlier,
        outlier_plist,
        outlier_type,
        outlier_type_plist,
    ) = metadata_page;

    // print out all the sizes
    println!("dictionary length: {}", dictionary.len());
    println!("template length: {}", template.len());
    println!(
        "template_pl total size : {}",
        template_plist.iter().map(|x| x.len()).sum::<usize>()
    );
    println!("outlier length: {}", outlier.len());
    println!(
        "outlier_pl total size : {}",
        outlier_plist.iter().map(|x| x.len()).sum::<usize>()
    );
    println!("outlier_type length: {}", outlier_type.len());
    println!(
        "outlier_type_pl total size : {}",
        outlier_type_plist.iter().map(|x| x.len()).sum::<usize>()
    );

    let end_time = std::time::Instant::now();

    for (_, line) in dictionary.lines().enumerate() {
        if line.contains(&query) {
            println!("query matched dictionary item, brute force {}", query);
            return Ok((0, Vec::new()));
        }
    }

    let mut match_uids = Vec::new();

    let search_text = |query: &str,
                       source_str: &str,
                       plists: &[Vec<PlistSize>],
                       match_uids: &mut Vec<(usize, PlistSize)>,
                       write: bool| {
        if write {
            println!("{}", source_str);
        }
        for (line_no, line) in source_str.lines().enumerate() {
            if let Some(_) = line.find(query) {
                println!("{} {}", line, line_no);
                let posting_list = &plists[line_no];
                for &uid in posting_list {
                    print!("{} ", uid);
                    match_uids.push((file_id, uid));
                }
                println!();
            }
        }
    };

    // search_text(&query, &template, &template_plist, &mut match_uids, false);
    for (_, line) in template.lines().enumerate() {
        if line.contains(&query) {
            println!("query matched template, brute force {}", query);
            return Ok((0, Vec::new()));
        }
    }

    // Print matched row groups
    for &uid in &match_uids {
        print!("{:?} ", uid);
    }

    search_text(&query, &outlier, &outlier_plist, &mut match_uids, false);

    if match_uids.len() >= limit.try_into().unwrap() {
        println!(
            "inexact query for top K satisfied by template and outlier {}",
            query
        );
        return Ok((1, match_uids));
    }

    // Search in outlier types
    search_text(
        &query,
        &outlier_type,
        &outlier_type_plist,
        &mut match_uids,
        false,
    );
    return Ok((1, match_uids));
}

fn write_1_block(
    fp: &mut File,
    numbers: Vec<usize>,
    lineno_buffer: &[Vec<PlistSize>],
    byte_offsets: &mut Vec<usize>,
) {
    let compressed_buffer =
        zstd::encode_all(&bincode::serialize(&numbers).unwrap()[..], 10).unwrap();
    let plist = PListChunk::new(lineno_buffer.to_vec());
    let serialized = plist.serialize().unwrap();

    fp.write_all(&(compressed_buffer.len() as u64).to_le_bytes())
        .unwrap();
    fp.write_all(&compressed_buffer).unwrap();
    fp.write_all(&serialized).unwrap();

    byte_offsets
        .push(byte_offsets.last().unwrap() + compressed_buffer.len() + serialized.len() + 8);
}

fn write_block(
    fp: &mut File,
    buffer: &str,
    lineno_buffer: &[Vec<PlistSize>],
    byte_offsets: &mut Vec<usize>,
) {
    let compressed_buffer = zstd::encode_all(buffer.as_bytes(), 0).unwrap();
    let plist = PListChunk::new(lineno_buffer.to_vec());
    let serialized = plist.serialize().unwrap();

    fp.write_all(&(compressed_buffer.len() as u64).to_le_bytes())
        .unwrap();
    fp.write_all(&compressed_buffer).unwrap();
    fp.write_all(&serialized).unwrap();

    byte_offsets
        .push(byte_offsets.last().unwrap() + compressed_buffer.len() + serialized.len() + 8);
}

const BLOCK_BYTE_LIMIT: usize = 1000000;

pub fn write_oahu(output_name: &str) -> Vec<(u64, String)> {
    // Get all types by listing compressed/compacted_type* files
    let mut types: Vec<i32> = Vec::new();
    for entry in read_dir("compressed").unwrap() {
        let path = entry.unwrap().path();
        let path_str = path.to_str().unwrap();
        if path_str.contains("compacted_type") && !path_str.contains("lineno") {
            println!("Processing file: {}", path_str);
            let type_num = path_str
                .split("compacted_type_")
                .nth(1)
                .unwrap()
                .parse::<i32>()
                .unwrap();
            if type_num != 0 {
                types.push(type_num);
            }
        }
    }

    let mut fp = File::create(format!("{}.oahu", output_name)).unwrap();
    let mut byte_offsets = vec![0];
    let mut type_offsets = vec![0];

    let mut type_uncompressed_lines_in_block = HashMap::new();
    let mut type_chunks = HashMap::new();

    let mut for_hawaii: Vec<(u64, String)> = vec![];
    let mut hawaii_types: Vec<i32> = vec![];

    for &type_num in &types {
        println!("Processing type: {}", type_num);

        let string_file_path = format!("compressed/compacted_type_{}", type_num);
        let lineno_file_path = format!("compressed/compacted_type_{}_lineno", type_num);

        let string_file = File::open(string_file_path).unwrap();
        let lineno_file = File::open(lineno_file_path).unwrap();

        let mut lineno_buffer = Vec::new();

        let mut uncompressed_lines_in_block = 0;
        let mut blocks_written = 0;
        let mut lines_in_buffer = 0;
        if USE_EXPERIMENTAL_NUMERICS && (type_num == 1) {
            let mut all_numbers: Vec<usize> = vec![];
            for (str_line, lineno_line) in BufReader::new(string_file)
                .lines()
                .zip(BufReader::new(lineno_file).lines())
            {
                let str_line = str_line.unwrap();
                let lineno_line = lineno_line.unwrap();
                //cast the str_line to a usize
                let number: usize = str_line.parse().unwrap();
                all_numbers.push(number);
                let numbers: Vec<u32> = lineno_line
                    .split_whitespace()
                    .map(|n| n.parse().unwrap())
                    .collect();
                lineno_buffer.push(numbers);
            }
            // sort numbers, lineno_buffer by numbers
            let mut paired: Vec<_> = all_numbers
                .into_iter()
                .zip(lineno_buffer.into_iter())
                .collect();
            paired.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
            all_numbers = paired.iter().map(|a| a.0).collect();
            lineno_buffer = paired.into_iter().map(|a| a.1).collect();

            write_1_block(&mut fp, all_numbers, &lineno_buffer, &mut byte_offsets);
        } else {
            let mut buffer = String::new();
            let mut this_for_hawaii: Vec<(u64, String)> = vec![];

            for (str_line, lineno_line) in BufReader::new(string_file)
                .lines()
                .zip(BufReader::new(lineno_file).lines())
            {
                let str_line = str_line.unwrap();
                let lineno_line = lineno_line.unwrap();

                buffer.push_str(&str_line);
                buffer.push('\n');
                lines_in_buffer += 1;
                this_for_hawaii.push((byte_offsets.len() as u64 - 1, str_line));

                let numbers: Vec<u32> = lineno_line
                    .split_whitespace()
                    .map(|n| n.parse().unwrap())
                    .collect();
                lineno_buffer.push(numbers);

                if uncompressed_lines_in_block == 0 && buffer.len() > BLOCK_BYTE_LIMIT / 2 {
                    let compressed_buffer = zstd::encode_all(buffer.as_bytes(), 0).unwrap();
                    uncompressed_lines_in_block =
                        ((BLOCK_BYTE_LIMIT as f32 / compressed_buffer.len() as f32)
                            * lines_in_buffer as f32) as usize;
                }

                if uncompressed_lines_in_block > 0 && lines_in_buffer == uncompressed_lines_in_block
                {
                    write_block(&mut fp, &buffer, &lineno_buffer, &mut byte_offsets);
                    buffer.clear();
                    lines_in_buffer = 0;
                    lineno_buffer.clear();
                    blocks_written += 1;
                }
            }

            if !buffer.is_empty() {
                write_block(&mut fp, &buffer, &lineno_buffer, &mut byte_offsets);
                blocks_written += 1;
            }

            if blocks_written > BRUTE_THRESHOLD {
                for_hawaii.extend(this_for_hawaii);
                hawaii_types.push(type_num);
            } else {
                this_for_hawaii.clear();
            }
            type_chunks.insert(type_num, blocks_written);
            type_uncompressed_lines_in_block.insert(type_num, uncompressed_lines_in_block);
        }

        type_offsets.push(byte_offsets.len() - 1);
    }

    println!("type_chunks: {:?}", type_chunks);
    println!(
        "type_uncompressed_lines_in_block: {:?}",
        type_uncompressed_lines_in_block
    );

    let metadata_page: (Vec<i32>, Vec<usize>, Vec<usize>, Vec<i32>) =
        (types, type_offsets, byte_offsets, hawaii_types);
    let compressed_metadata =
        encode_all(&bincode::serialize(&metadata_page).unwrap()[..], 10).unwrap();
    fp.write_all(&compressed_metadata).unwrap();
    fp.write_all(&(compressed_metadata.len() as u64).to_le_bytes())
        .unwrap();

    println!("{:?}", for_hawaii.len());

    for_hawaii
}

pub async fn search_hawaii_oahu(
    file_id: usize,
    hawaii_filename: String,
    mut reader_oahu: AsyncReader,
    oahu_size: usize,
    query: String,
    limit: usize,
    wavelet_tree: bool,
    exact: bool,
) -> Result<Vec<(usize, PlistSize)>, LavaError> {
    info!("query: {}", query);

    let types_to_search = if exact {
        vec![get_type(&query)]
    } else {
        get_all_types(get_type(&query))
    };
    for &type_to_search in &types_to_search {
        info!("type to search: {}", type_to_search);
    }

    let metadata_page_length = reader_oahu.read_usize_from_end(1).await?[0];
    // Read the metadata page
    let metadata_page: (Vec<i32>, Vec<usize>, Vec<usize>, Vec<i32>) = read_and_decompress(
        &mut reader_oahu,
        oahu_size as u64 - metadata_page_length as u64 - 8,
        metadata_page_length as u64,
    )
    .await?;
    let (types, type_offsets, byte_offsets, hawaii_types) = metadata_page;

    println!("Hawaii types: {:?}", hawaii_types);

    // see if anything in hawaii_types intersects with type_to_search

    let type_intersection = hawaii_types
        .iter()
        .filter(|&&type_| types_to_search.contains(&type_))
        .copied()
        .collect::<Vec<i32>>();

    let remainder_types = types_to_search
        .iter()
        .filter(|&&type_| !type_intersection.contains(&type_))
        .copied()
        .collect::<Vec<i32>>();

    let mut chunks: Vec<u64> = if type_intersection.is_empty() {
        vec![]
    } else {
        _search_lava_substring_char(
            vec![hawaii_filename],
            query.clone(),
            limit,
            ReaderType::default(),
            None,
            None,
            wavelet_tree,
        )
        .await
        .unwrap()
        .into_iter()
        .map(|(_, x)| x)
        .collect_vec()
    };
    println!("chunks {:?}", chunks);

    let mut all_uids = Vec::new();

    for remainder_type in remainder_types.iter() {
        let type_index = types.iter().position(|&x| x == *remainder_type);
        if type_index.is_none() {
            continue;
        }
        let type_index = type_index.unwrap();
        if USE_EXPERIMENTAL_NUMERICS && (*remainder_type == 1) {
            let block = reader_oahu
                .read_range(
                    byte_offsets[type_offsets[type_index]] as u64,
                    byte_offsets[type_offsets[type_index] + 1] as u64,
                )
                .await
                .unwrap();
            let compressed_nums_length =
                u64::from_le_bytes(block[0..8].try_into().unwrap()) as usize;
            println!("compressed_nums_length {:?}", compressed_nums_length);
            println!("total bytes {:?}", block.len());
            let mut decompressor = Decoder::new(&block[8..8 + compressed_nums_length]).unwrap();
            let mut decompressed = Vec::new();
            std::io::copy(&mut decompressor, &mut decompressed)?;
            let all_numbers: Vec<usize> = bincode::deserialize(&decompressed)?;

            let compressed_plist = &block[8 + compressed_nums_length..];
            let plist = PListChunk::from_compressed(compressed_plist).unwrap();

            for (line_number, this_number) in all_numbers.iter().enumerate() {
                if this_number.to_string().contains(&query) {
                    all_uids.extend(
                        plist
                            .lookup(line_number)
                            .unwrap()
                            .iter()
                            .map(|x| (file_id, *x)),
                    );
                }
            }
        } else {
            chunks.extend(type_offsets[type_index] as u64..type_offsets[type_index + 1] as u64);
        }
    }
    println!("chunks {:?}", chunks);

    let search_chunk = |block: bytes::Bytes, query_clone: String| {
        let compressed_strings_length =
            u64::from_le_bytes(block[0..8].try_into().unwrap()) as usize;
        let compressed_strings = &block[8..8 + compressed_strings_length];

        let mut decompressor = Decoder::new(compressed_strings).unwrap();
        let mut decompressed_strings: Vec<u8> =
            Vec::with_capacity(compressed_strings.len() as usize);
        decompressor.read_to_end(&mut decompressed_strings).unwrap();

        let compressed_plist = &block[8 + compressed_strings_length..];
        let plist = PListChunk::from_compressed(compressed_plist).unwrap();

        let mut uids = Vec::new();
        for (line_number, line) in String::from_utf8_lossy(&decompressed_strings)
            .lines()
            .enumerate()
        {
            if format!("\n{}\n", line).contains(&query_clone) {
                uids.extend(plist.lookup(line_number).unwrap());
            }
        }

        uids
    };

    let mut set = JoinSet::new();

    for chunk in chunks {
        let mut reader_clone = reader_oahu.clone(); // Assuming AsyncReader implements Clone
        let query_clone = query.to_string();
        let byte_offsets_clone = byte_offsets.clone();

        set.spawn(async move {
            let block: bytes::Bytes = reader_clone
                .read_range(
                    byte_offsets_clone[chunk as usize] as u64,
                    byte_offsets_clone[chunk as usize + 1] as u64,
                )
                .await
                .unwrap();
            search_chunk(block, query_clone)
        });
    }

    while let Some(result) = set.join_next().await {
        let result = result.unwrap();
        all_uids.extend(result.into_iter().map(|x| (file_id, x)));
        if all_uids.len() >= limit {
            return Ok(all_uids);
        }
    }

    return Ok(all_uids);
}

#[tokio::main]
pub async fn index_logcloud(index_name: &str, num_groups: usize, use_wavelet: Option<bool>) {
    let use_wavelet = use_wavelet.unwrap_or(false);
    let _ = compact(num_groups);
    let _ = write_kauai(index_name, num_groups).unwrap();
    let texts: Vec<(u64, String)> = write_oahu(index_name);
    if use_wavelet {
        let _ = _build_lava_substring_char_wavelet(format!("{}.hawaii", index_name), texts, 1)
            .await
            .unwrap();
    } else {
        let _ = _build_lava_substring_char(format!("{}.hawaii", index_name), texts, 1)
            .await
            .unwrap();
    }
}

#[tokio::main]
pub async fn index_analysis(split_index_prefixes: Vec<String>, reader_type: ReaderType) -> () {
    let mut oahu_filenames = split_index_prefixes
        .iter()
        .map(|split_index_prefix| format!("{}.oahu", split_index_prefix))
        .collect::<Vec<_>>();

    let mut hawaii_filenames = split_index_prefixes
        .iter()
        .map(|split_index_prefix| format!("{}.hawaii", split_index_prefix))
        .collect::<Vec<_>>();

    let (oahu_sizes, mut reader_oahus) =
        get_file_sizes_and_readers(&oahu_filenames, reader_type.clone())
            .await
            .unwrap();
    let (hawaii_sizes, mut reader_hawaiis) =
        get_file_sizes_and_readers(&hawaii_filenames, reader_type.clone())
            .await
            .unwrap();

    let mut total_fm_index_size = 0;
    let mut total_suffix_array_size = 0;
    let mut total_compressed_strings_length = 0;
    let mut total_compressed_plist_length = 0;
    let mut total_csr_length = 0;
    let mut total_roaring_length = 0;

    for (hawaii_size, mut reader_hawaii) in hawaii_sizes.into_iter().zip(reader_hawaiis.into_iter())
    {
        let results = reader_hawaii.read_usize_from_end(4).await.unwrap();
        let posting_list_offsets_offset = results[1];
        let total_counts_offset = results[2];

        let posting_list_offsets: Vec<u64> = reader_hawaii
            .read_range_and_decompress(posting_list_offsets_offset, total_counts_offset)
            .await
            .unwrap();

        total_fm_index_size += posting_list_offsets[0];
        total_suffix_array_size +=
            posting_list_offsets[posting_list_offsets.len() - 1] - posting_list_offsets[0];
    }

    for (oahu_size, mut reader_oahu) in oahu_sizes.into_iter().zip(reader_oahus.into_iter()) {
        let metadata_page_length = reader_oahu.read_usize_from_end(1).await.unwrap()[0];
        // Read the metadata page
        let metadata_page: (Vec<i32>, Vec<usize>, Vec<usize>, Vec<i32>) = read_and_decompress(
            &mut reader_oahu,
            oahu_size as u64 - metadata_page_length as u64 - 8,
            metadata_page_length as u64,
        )
        .await
        .unwrap();
        let (types, type_offsets, byte_offsets, hawaii_types) = metadata_page;

        for i in 0..byte_offsets.len() - 1 {
            let block = reader_oahu
                .read_range(byte_offsets[i] as u64, byte_offsets[i + 1] as u64)
                .await
                .unwrap();
            let compressed_strings_length =
                u64::from_le_bytes(block[0..8].try_into().unwrap()) as usize;
            let compressed_strings = &block[8..8 + compressed_strings_length];

            let mut decompressor = Decoder::new(compressed_strings).unwrap();
            let mut decompressed_strings: Vec<u8> =
                Vec::with_capacity(compressed_strings.len() as usize);
            decompressor.read_to_end(&mut decompressed_strings).unwrap();

            let compressed_plist = &block[8 + compressed_strings_length..];
            let plist = PListChunk::from_compressed(compressed_plist).unwrap();

            let plist_data = plist.data();

            let mut csr_offsets = vec![0];
            let mut values = Vec::new();

            let mut block_roaring_length = 0;
            let mut total_serialized_string = Vec::new();
            let mut roaring_offsets = vec![0];

            for plist in plist_data {
                let mut r = RoaringBitmap::new();

                csr_offsets.push(csr_offsets.last().unwrap() + plist.len());

                for &i in plist {
                    r.insert(i);
                    values.push(i);
                }

                let mut serialized_bitmap = vec![];
                r.serialize_into(&mut serialized_bitmap).unwrap();
                total_serialized_string.extend_from_slice(&serialized_bitmap);
                roaring_offsets.push(roaring_offsets.last().unwrap() + serialized_bitmap.len());
            }

            let compressed_roaring_offsets =
                encode_all(&bincode::serialize(&roaring_offsets).unwrap()[..], 10).unwrap();
            block_roaring_length = encode_all(&total_serialized_string[..], 10).unwrap().len()
                + compressed_roaring_offsets.len();

            // Compress CSR offsets and values
            let compressed_csr_offsets =
                encode_all(&bincode::serialize(&csr_offsets).unwrap()[..], 10).unwrap();
            let compressed_values =
                encode_all(&bincode::serialize(&values).unwrap()[..], 10).unwrap();

            // println!("Block {} compressed csr offsets length: {}", i, compressed_csr_offsets.len());
            // println!("Block {} compressed values length: {}", i, compressed_values.len());
            // println!("Block {} compressed strings length: {}", i, compressed_strings_length);
            // println!("Block {} compressed plist length: {}", i, compressed_plist.len());
            // println!("Block {} compressed roaring length: {}", i, block_roaring_length);

            total_compressed_strings_length += compressed_strings_length;
            total_compressed_plist_length += compressed_plist.len();
            total_csr_length += compressed_csr_offsets.len() + compressed_values.len();
            total_roaring_length += block_roaring_length;
        }
    }

    println!("Total fm index size: {}", total_fm_index_size);
    println!("Total suffix array size: {}", total_suffix_array_size);
    println!(
        "Total compressed strings length: {}",
        total_compressed_strings_length
    );
    println!(
        "Total compressed plist length: {}",
        total_compressed_plist_length
    );
    println!("Total compressed roaring length: {}", total_roaring_length);
    println!("Total compressed csr length: {}", total_csr_length);
}

#[tokio::main]
pub async fn search_logcloud(
    split_index_prefixes: Vec<String>,
    query: String,
    limit: usize,
    reader_type: ReaderType,
    wavelet_tree: bool,
    exact: bool,
) -> Result<(u32, Vec<(usize, PlistSize)>), LavaError> {
    info!("split_index_prefixes: {:?}", split_index_prefixes);

    let start_time = std::time::Instant::now();

    let kauai_filenames = split_index_prefixes
        .iter()
        .map(|split_index_prefix| format!("{}.kauai", split_index_prefix))
        .collect::<Vec<_>>();

    let (kauai_sizes, reader_kauais) =
        get_file_sizes_and_readers(&kauai_filenames, reader_type.clone()).await?;

    let mut set = JoinSet::new();
    for (file_id, (kauai_size, reader_kauai)) in kauai_sizes
        .into_iter()
        .zip(reader_kauais.into_iter())
        .enumerate()
    {
        let query_clone = query.clone();
        set.spawn(async move {
            search_kauai(
                file_id,
                reader_kauai,
                kauai_size,
                query_clone,
                limit.try_into().unwrap(),
            )
            .await
            .unwrap()
        });
    }

    let mut all_uids: Vec<(usize, PlistSize)> = Vec::new();
    while let Some(result) = set.join_next().await {
        let result = result.unwrap();
        match result.0 {
            0 => {
                println!("brute force");
                return Ok((0, vec![]));
            }
            1 => {
                all_uids.extend(result.1);

                if all_uids.len() >= limit {
                    return Ok((1, all_uids));
                }
            }
            _ => {
                return Err(LavaError::Parse(
                    "Unexpected result from search_kauai".to_string(),
                ))
            }
        }
    }

    println!("kauai time {:?}", start_time.elapsed());

    let start_time = std::time::Instant::now();

    // at this point we are not able to satisfy our query with kauai files alone, must query oahu and possibly hawaii files.
    // we should do an exponential search strategy, i.e. 1 2 4 8 etc, but that's too much work for now

    let oahu_filenames = split_index_prefixes
        .iter()
        .map(|split_index_prefix| format!("{}.oahu", split_index_prefix))
        .collect::<Vec<_>>();

    let mut hawaii_filenames = split_index_prefixes
        .iter()
        .map(|split_index_prefix| format!("{}.hawaii", split_index_prefix))
        .collect::<Vec<_>>();

    let (oahu_sizes, mut reader_oahus) =
        get_file_sizes_and_readers(&oahu_filenames, reader_type.clone()).await?;

    let mut set = JoinSet::new();
    let new_limit = limit - all_uids.len();

    for (file_id, (oahu_size, reader_oahu)) in oahu_sizes
        .into_iter()
        .zip(reader_oahus.into_iter())
        .enumerate()
    {
        let hawaii_filename = hawaii_filenames.remove(0);
        let query_clone = query.clone();
        set.spawn(async move {
            search_hawaii_oahu(
                file_id,
                hawaii_filename,
                reader_oahu,
                oahu_size,
                query_clone,
                new_limit,
                wavelet_tree,
                exact,
            )
            .await
            .unwrap()
        });
    }

    while let Some(result) = set.join_next().await {
        let result = result.unwrap();
        all_uids.extend(result);
        if all_uids.len() >= limit {
            return Ok((1, all_uids));
        }
    }

    println!("oahu time {:?}", start_time.elapsed());

    let count = READ_RANGE_COUNTER.load(Ordering::SeqCst);
    println!("read_range has been called {} times", count);

    // println!("all_uids {:?}", all_uids);

    Ok((1, all_uids))
}
