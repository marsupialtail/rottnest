use log::{info, warn};
use parquet::{column::reader, format::DictionaryPageHeader};
use rand::Rng;
use tokio::{task::JoinSet, time::sleep};

use crate::{
    formats::readers::{
        get_file_size_and_reader, get_file_sizes_and_readers, get_reader, AsyncReader, ClonableAsyncReader, ReaderType,
    },
    lava::{
        build::_build_lava_substring_char,
        error::LavaError,
        logcloud_plist::{PListChunk, PlistSize},
        search::search_lava_substring_char,
    },
};
use serde::de::DeserializeOwned;
use std::io::{BufRead, BufReader, Write};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, read_dir, File},
    io::{self, Read},
    path::Path,
    time::{Duration, Instant},
};
use zstd::stream::{encode_all, read::Decoder};

async fn read_and_decompress<T>(reader: &mut AsyncReader, start: u64, size: u64) -> Result<T, LavaError>
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

const ROW_GROUP_SIZE: usize = 100000;
const DICT_RATIO_THRESHOLD: f64 = 0.5;

fn merge_files(
    input_filenames: &[String],
    input_filenames_linenumbers: &[String],
    output_filename: &str,
    output_filename_linenumbers: &str,
    num_row_groups: usize,
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
        current_linenumbers[i] = line.split_whitespace().filter_map(|n| n.parse::<i32>().ok()).collect();
    }

    while current_lines.iter().any(|s| !s.is_empty()) {
        // Find the smallest string in `current_lines` without holding a reference to it
        let it = current_lines.iter().filter(|s| !s.is_empty()).min().cloned().unwrap_or_else(|| String::new());

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

        if it_linenumbers.len() > (num_row_groups as f64 * DICT_RATIO_THRESHOLD) as usize {
            write!(dict_file, "{}", it)?;
        } else {
            write!(output_file, "{}", it)?;
            for num in it_linenumbers {
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
                    current_linenumbers[i] =
                        lineno_line.split_whitespace().filter_map(|n| n.parse::<i32>().ok()).collect();
                }
            }
        }
    }

    Ok(())
}

fn compact(num_groups: usize) -> io::Result<()> {
    // Read the total number of lines
    let filename = format!("compressed/{}/current_line_number", num_groups - 1);
    let file = File::open(&filename)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let total_lines: usize = line.trim().parse().unwrap();
    let num_row_groups = total_lines / ROW_GROUP_SIZE + 1;

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
            num_row_groups,
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
            input_filenames_linenumbers.push(format!("compressed/{}/compacted_type_{}_lineno", i, type_));
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
            num_row_groups,
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
    let compressed_dictionary = encode_all(&bincode::serialize(&dictionary_str.as_bytes()).unwrap()[..], 10)?;
    fp.write_all(&compressed_dictionary)?;
    byte_offsets.push(fp.metadata()?.len());

    let mut templates = Vec::new();
    let mut template_posting_lists: Vec<Vec<PlistSize>> = Vec::new();
    let mut outliers = Vec::new();
    let mut outlier_linenos: Vec<Vec<PlistSize>> = Vec::new();

    let mut lineno: PlistSize = 0;

    println!("Reading templates...");

    for group_number in 0..num_groups {
        let mut group_template_idx = HashMap::new();
        let template_file = BufReader::new(File::open(format!("compressed/{}_{}.templates", filename, group_number))?);

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
            .take_while(|&chunk| Path::new(&format!("compressed/{}/chunk{:04}.eid", group_number, chunk)).exists())
            .count();

        for chunk in 0..total_chunks {
            let eid_file = File::open(format!("compressed/{}/chunk{:04}.eid", group_number, chunk))?;
            // let mut outlier_file = File::open(format!("compressed/{}/chunk{:04}.outlier", group_number, chunk))?;
            let mut outlier_file: Option<BufReader<File>> =
                if File::open(format!("compressed/{}/chunk{:04}.outlier", group_number, chunk)).is_ok() {
                    Some(BufReader::new(File::open(format!("compressed/{}/chunk{:04}.outlier", group_number, chunk))?))
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
                    outlier_linenos.push(vec![lineno / ROW_GROUP_SIZE as u32]);
                } else {
                    let idx = if group_template_idx.contains_key(&(eid as usize)) {
                        group_template_idx[&(eid as usize)]
                    } else {
                        panic!("Template not found for eid: {}, {:?}", eid, group_template_idx);
                    };
                    if template_posting_lists[idx].is_empty()
                        || template_posting_lists[idx].last() != Some(&(lineno / ROW_GROUP_SIZE as u32))
                    {
                        template_posting_lists[idx].push(lineno / ROW_GROUP_SIZE as u32);
                    }
                }
                lineno = lineno.wrapping_add(1);
                if lineno == 0 {
                    return Err(std::io::Error::new(std::io::ErrorKind::Other, "overflow"));
                }
            }
        }
    }

    // Write templates and template_posting_lists to files
    {
        let mut template_fp = File::create("compressed/template")?;
        let mut template_lineno_fp = File::create("compressed/template_lineno")?;
        for (template, posting_list) in templates.iter().zip(&template_posting_lists) {
            writeln!(template_fp, "{}", template)?;
            for &num in posting_list {
                write!(template_lineno_fp, "{} ", num)?;
            }
            writeln!(template_lineno_fp)?;
        }
    }

    // Remove templates with empty posting lists
    let templates = templates
        .iter()
        .enumerate()
        .filter(|&(i, _)| !template_posting_lists[i].is_empty())
        .map(|(_, template)| template.clone())
        .collect::<Vec<_>>();

    template_posting_lists.retain(|list| !list.is_empty());

    let template_str = templates.join("\n") + "\n";
    let compressed_template_str = encode_all(&bincode::serialize(&template_str).unwrap()[..], 0)?;
    fp.write_all(&compressed_template_str)?;
    byte_offsets.push(fp.metadata()?.len());

    let serialized2 = encode_all(&bincode::serialize(&template_posting_lists).unwrap()[..], 10).unwrap();
    fp.write_all(&serialized2)?;
    byte_offsets.push(fp.metadata()?.len());

    let outlier_str = outliers.join("") + "\n";
    let compressed_outlier_str = encode_all(&bincode::serialize(&outlier_str).unwrap()[..], 0)?;
    fp.write_all(&compressed_outlier_str)?;
    byte_offsets.push(fp.metadata()?.len());

    let serialized = encode_all(&bincode::serialize(&outlier_linenos).unwrap()[..], 10).unwrap();
    fp.write_all(&serialized)?;
    byte_offsets.push(fp.metadata()?.len());

    let mut outlier_type_str = String::new();
    let mut outlier_type_linenos = Vec::new();
    let outlier_file = File::open("compressed/outlier")?;
    let outlier_lineno_file = File::open("compressed/outlier_lineno")?;
    for (line, outlier_type_line) in
        BufReader::new(outlier_lineno_file).lines().zip(BufReader::new(outlier_file).lines())
    {
        let line = line?;
        let outlier_type_line = outlier_type_line?;
        outlier_type_str.push_str(&outlier_type_line);
        outlier_type_str.push('\n');
        let numbers: Vec<PlistSize> =
            line.split_whitespace().filter_map(|s| s.parse().ok()).collect::<HashSet<_>>().into_iter().collect();
        outlier_type_linenos.push(numbers);
    }

    let compressed_outlier_type_str = encode_all(&bincode::serialize(&outlier_type_str).unwrap()[..], 0)?;
    fp.write_all(&compressed_outlier_type_str)?;
    byte_offsets.push(fp.metadata()?.len());

    let serialized3 = encode_all(&bincode::serialize(&outlier_type_linenos).unwrap()[..], 10).unwrap();
    fp.write_all(&serialized3)?;

    for offset in byte_offsets {
        fp.write_all(&offset.to_le_bytes())?;
    }

    fp.flush()?;

    Ok(())
}

// std::pair<int, std::vector<plist_size_t>> search_kauai(VirtualFileRegion * vfr, std::string query, int k) {

async fn search_kauai(
    reader: &mut AsyncReader,
    file_size: usize,
    query: &str,
    k: u32,
) -> Result<(u32, Vec<PlistSize>), LavaError> {
    let byte_offsets = reader.read_usize_from_end(6).await?;

    println!("byte offsets: {:?}", byte_offsets);

    let dictionary: String = read_and_decompress(reader, 0, byte_offsets[0]).await?;
    let dictionary: String = "".to_string();
    let template: String = read_and_decompress(reader, byte_offsets[0], byte_offsets[1] - byte_offsets[0]).await?;
    let template_plist: Vec<Vec<PlistSize>> =
        read_and_decompress(reader, byte_offsets[1], byte_offsets[2] - byte_offsets[1]).await?;
    let outlier: String = read_and_decompress(reader, byte_offsets[2], byte_offsets[3] - byte_offsets[2]).await?;
    let outlier_plist: Vec<Vec<PlistSize>> =
        read_and_decompress(reader, byte_offsets[3], byte_offsets[4] - byte_offsets[3]).await?;
    let outlier_type: String = read_and_decompress(reader, byte_offsets[4], byte_offsets[5] - byte_offsets[4]).await?;
    let outlier_type_pl_size = file_size as u64 - byte_offsets[5] - 6 * std::mem::size_of::<usize>() as u64;
    let outlier_type_plist: Vec<Vec<PlistSize>> =
        read_and_decompress(reader, byte_offsets[5], outlier_type_pl_size).await?;

    for (_, line) in dictionary.lines().enumerate() {
        if line.contains(query) {
            println!("query matched dictionary item, brute force {}", query);
            return Ok((0, Vec::new()));
        }
    }

    let mut matched_row_groups = Vec::new();

    let search_text = |query: &str,
                       source_str: &str,
                       plists: &[Vec<PlistSize>],
                       matched_row_groups: &mut Vec<PlistSize>,
                       write: bool| {
        if write {
            println!("{}", source_str);
        }
        for (line_no, line) in source_str.lines().enumerate() {
            if let Some(_) = line.find(query) {
                println!("{} {}", line, line_no);
                let posting_list = &plists[line_no];
                for &row_group in posting_list {
                    print!("{} ", row_group);
                    matched_row_groups.push(row_group);
                }
                println!();
            }
        }
    };

    search_text(query, &template, &template_plist, &mut matched_row_groups, false);

    // Print matched row groups
    for &row_group in &matched_row_groups {
        print!("{} ", row_group);
    }
    println!("MADE IT HERE");

    search_text(query, &outlier, &outlier_plist, &mut matched_row_groups, false);

    if matched_row_groups.len() >= k.try_into().unwrap() {
        println!("inexact query for top K satisfied by template and outlier {}", query);
        return Ok((1, matched_row_groups));
    }

    println!("MADE IT HERE");

    // Search in outlier types
    search_text(query, &outlier_type, &outlier_type_plist, &mut matched_row_groups, false);

    if matched_row_groups.len() >= k.try_into().unwrap() {
        println!("inexact query for top K satisfied by template, outlier and outlier types {}", query);
        Ok((1, matched_row_groups))
    } else {
        println!("inexact query for top K not satisfied by template, outlier and outlier types {}", query);
        Ok((2, matched_row_groups))
    }
}

async fn search_oahu(
    reader: &mut AsyncReader,
    file_size: usize,
    query_type: i32,
    chunks: Option<Vec<usize>>,
    query_str: &str,
) -> Result<Vec<PlistSize>, LavaError> {
    // Read the metadata page length
    let metadata_page_length = reader.read_usize_from_end(1).await?[0];
    // Read the metadata page
    let metadata_page: (Vec<i32>, Vec<usize>, Vec<usize>) =
        read_and_decompress(reader, file_size as u64 - metadata_page_length as u64 - 8, metadata_page_length as u64)
            .await?;
    let (types, type_offsets, byte_offsets) = metadata_page;
    // Find query_type in type_order
    println!("type_order: {:?} {}", types, query_type);
    let type_index = types.iter().position(|&x| x == query_type);
    if type_index.is_none() {
        return Ok(Vec::new());
    }
    let type_index = type_index.unwrap();

    let type_offset = type_offsets[type_index];
    let num_chunks = type_offsets[type_index + 1] - type_offset;

    // Process blocks using JoinSet, if chunks is specified make sure it's shorter than num_chunks, otherwise it is 0 .. num_chunks

    let chunks = match chunks {
        Some(chunks) => {
            if chunks.len() <= num_chunks {
                chunks
            } else {
                return Err(LavaError::Parse("Invalid chunks specified".to_string()));
            }
        }
        None => (0..num_chunks).collect(),
    };

    let mut set = JoinSet::new();

    for chunk in chunks {
        let block_offset = byte_offsets[type_offset + chunk] as u64;
        let next_block_offset = byte_offsets[type_offset + chunk + 1] as u64;
        let block_size = next_block_offset - block_offset;

        let mut reader_clone = reader.clone(); // Assuming AsyncReader implements Clone
        let query_str_clone = query_str.to_string();

        set.spawn(async move {
            let block = reader_clone.read_range(block_offset, block_offset + block_size).await.unwrap();

            let compressed_strings_length = u64::from_le_bytes(block[0..8].try_into().unwrap()) as usize;
            let compressed_strings = &block[8..8 + compressed_strings_length];

            let mut decompressor = Decoder::new(compressed_strings).unwrap();
            let mut decompressed_strings: Vec<u8> = Vec::with_capacity(compressed_strings.len() as usize);
            decompressor.read_to_end(&mut decompressed_strings).unwrap();

            let compressed_plist = &block[8 + compressed_strings_length..];
            let plist = PListChunk::from_compressed(compressed_plist).unwrap();

            let mut row_groups = Vec::new();
            for (line_number, line) in String::from_utf8_lossy(&decompressed_strings).lines().enumerate() {
                if format!("\n{}\n", line).contains(&query_str_clone) {
                    row_groups.extend(plist.lookup(line_number).unwrap());
                }
            }

            row_groups
        });
    }

    let mut all_row_groups = Vec::new();
    while let Some(result) = set.join_next().await {
        let result = result.unwrap();
        all_row_groups.extend(result);
    }

    Ok(all_row_groups)
}

fn write_block(fp: &mut File, buffer: &str, lineno_buffer: &[Vec<PlistSize>], byte_offsets: &mut Vec<usize>) {
    let compressed_buffer = zstd::encode_all(buffer.as_bytes(), 0).unwrap();
    let plist = PListChunk::new(lineno_buffer.to_vec());
    let serialized = plist.serialize().unwrap();

    fp.write_all(&(compressed_buffer.len() as u64).to_le_bytes()).unwrap();
    fp.write_all(&compressed_buffer).unwrap();
    fp.write_all(&serialized).unwrap();

    byte_offsets.push(byte_offsets.last().unwrap() + compressed_buffer.len() + serialized.len() + 8);
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
            let type_num = path_str.split("compacted_type_").nth(1).unwrap().parse::<i32>().unwrap();
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

    for &type_num in &types {
        println!("Processing type: {}", type_num);

        let string_file_path = format!("compressed/compacted_type_{}", type_num);
        let lineno_file_path = format!("compressed/compacted_type_{}_lineno", type_num);

        let string_file = File::open(string_file_path).unwrap();
        let lineno_file = File::open(lineno_file_path).unwrap();

        let mut buffer = String::new();
        let mut lineno_buffer = Vec::new();

        let mut uncompressed_lines_in_block = 0;
        let mut blocks_written = 0;
        let mut lines_in_buffer = 0;

        let mut this_for_hawaii: Vec<(u64, String)> = vec![];

        for (str_line, lineno_line) in BufReader::new(string_file).lines().zip(BufReader::new(lineno_file).lines()) {
            let str_line = str_line.unwrap();
            let lineno_line = lineno_line.unwrap();

            buffer.push_str(&str_line);
            buffer.push('\n');
            lines_in_buffer += 1;
            this_for_hawaii.push((byte_offsets.len() as u64, str_line));

            let numbers: Vec<u32> = lineno_line.split_whitespace().map(|n| n.parse().unwrap()).collect();
            lineno_buffer.push(numbers);

            if uncompressed_lines_in_block == 0 && buffer.len() > BLOCK_BYTE_LIMIT / 2 {
                let compressed_buffer = zstd::encode_all(buffer.as_bytes(), 0).unwrap();
                uncompressed_lines_in_block =
                    ((BLOCK_BYTE_LIMIT as f32 / compressed_buffer.len() as f32) * lines_in_buffer as f32) as usize;
            }

            if uncompressed_lines_in_block > 0 && lines_in_buffer == uncompressed_lines_in_block {
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
        } else {
            this_for_hawaii.clear();
        }

        type_chunks.insert(type_num, blocks_written);

        type_offsets.push(byte_offsets.len() - 1);
        type_uncompressed_lines_in_block.insert(type_num, uncompressed_lines_in_block);
    }

    println!("type_chunks: {:?}", type_chunks);
    println!("type_uncompressed_lines_in_block: {:?}", type_uncompressed_lines_in_block);

    let metadata_page: (Vec<i32>, Vec<usize>, Vec<usize>) = (types, type_offsets, byte_offsets);
    let compressed_metadata = encode_all(&bincode::serialize(&metadata_page).unwrap()[..], 10).unwrap();
    fp.write_all(&compressed_metadata).unwrap();
    fp.write_all(&(compressed_metadata.len() as u64).to_le_bytes()).unwrap();

    println!("{:?}", for_hawaii.len());

    for_hawaii
}

const SYMBOL_TY: i32 = 32;
const BRUTE_THRESHOLD: usize = 5;
const CHAR_TABLE: [i32; 128] = [
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 32, 32,
    32, 32, 32, 32, 32, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 32, 32, 32, 32,
    32, 32, 4, 4, 4, 4, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32,
    32, 32, 32,
];

fn get_type(query: &str) -> i32 {
    query.bytes().fold(0, |type_acc, c| type_acc | if c >= 128 { SYMBOL_TY } else { CHAR_TABLE[c as usize] })
}

fn get_all_types(type_: i32) -> Vec<i32> {
    (1..=63).filter(|&i| (type_ & i) == type_).collect()
}

pub async fn search_hawaii_oahu(
    reader_hawaii: &mut AsyncReader,
    hawaii_size: usize,
    reader_oahu: &mut AsyncReader,
    oahu_size: usize,
    query: &str,
    limit: usize,
) -> Result<HashSet<PlistSize>, LavaError> {
    let processed_query: String = query.chars().filter(|&c| c != '\n').collect();

    info!("query: {}", processed_query);

    let query_type = get_type(&processed_query);
    info!("deduced type: {}", query_type);

    let types_to_search = get_all_types(query_type);
    for &type_to_search in &types_to_search {
        info!("type to search: {}", type_to_search);
    }

    let mut results = HashSet::new();

    for &type_to_search in &types_to_search {
        let mut found = search_oahu(reader_oahu, oahu_size, type_to_search, None, &processed_query).await?;
        results.extend(found.drain(..));
    }

    println!("results: {:?}", results);

    // #[tokio::main]
    // pub async fn search_lava_substring_char(
    //     files: Vec<String>,
    //     query: String,
    //     k: usize,
    //     reader_type: ReaderType,
    //     token_viable_limit: Option<usize>,
    //     sample_factor: Option<usize>,
    // )

    // for (&type_, chunks) in &result {
    //     info!("searching type {}", type_);
    //     for &chunk in chunks {
    //         info!("chunk {}", chunk);
    //     }
    // }

    // for (&type_, chunks) in &result {
    //     info!("searching type {}", type_);

    //     let found = if chunks == &HashSet::from([usize::MAX]) {
    //         info!("type not found, brute forcing Oahu");
    //         let chunks_to_search: Vec<usize> = (0..BRUTE_THRESHOLD).collect();
    //         search_oahu(reader_oahu, oahu_size, type_, chunks_to_search, &processed_query).await?
    //     } else {
    //         let mut chunks_vec: Vec<usize> = chunks.iter().cloned().collect();
    //         chunks_vec.truncate(limit);
    //         search_oahu(reader_oahu, oahu_size, type_, &chunks_vec, &processed_query).await?
    //     };

    //     results.extend(found);
    // }

    Ok(results)
}

#[tokio::main]
pub async fn index_logcloud(index_name: &str, num_groups: usize) {
    let _ = compact(num_groups);
    let _ = write_kauai(index_name, num_groups).unwrap();
    let texts = write_oahu(index_name);
    let _ = _build_lava_substring_char(format!("{}.hawaii", index_name), texts, 1).await.unwrap();
}

#[tokio::main]
pub async fn search_logcloud(
    split_index_prefix: String,
    query: String,
    limit: usize,
    reader_type: ReaderType,
) -> Result<Vec<usize>, LavaError> {
    /*
    Expects a split_index_prefix of the form s3://bucket/index-name/indices/split_id or path/index-name/indices/split_id
    */
    info!("split_index_prefix: {}", split_index_prefix);

    let (kauai_size, mut reader_kauai) =
        get_file_size_and_reader(format!("{}.kauai", split_index_prefix), reader_type.clone()).await?;

    let result = search_kauai(&mut reader_kauai, kauai_size, &query, limit.try_into().unwrap()).await?;
    println!("result: {:?}", result);

    let (oahu_size, mut reader_oahu) =
        get_file_size_and_reader(format!("{}.oahu", split_index_prefix), reader_type.clone()).await?;
    // let (hawaii_size, mut reader_hawaii) =
    //     get_file_size_and_reader(format!("{}.hawaii", split_index_prefix), reader_type).await?;
    let hawaii_size = oahu_size;
    let mut reader_hawaii = reader_oahu.clone();
    let mut return_results = Vec::new();

    match result.0 {
        0 => {
            // you have to brute force
            return_results.push(usize::MAX); // Using usize::MAX instead of -1
        }
        1 => {
            return_results.extend(result.1.iter().map(|&x| x as usize));
        }
        2 => {
            let current_results: Vec<usize> = result.1.into_iter().map(|x| x as usize).collect();
            let next_results =
                search_hawaii_oahu(&mut reader_hawaii, hawaii_size, &mut reader_oahu, oahu_size, &query, limit).await?;
            return_results.extend(current_results);
            return_results.extend(next_results.iter().map(|&x| x as usize));
        }
        _ => return Err(LavaError::Parse("Unexpected result from search_kauai".to_string())),
    }

    Ok(return_results)
}
