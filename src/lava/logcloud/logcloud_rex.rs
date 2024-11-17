use arrow::array::{make_array, Array, ArrayData, LargeStringArray, UInt64Array};
use chrono::NaiveDateTime;
use libc::{c_char, c_int};
use pyo3::prelude::*;
use rand::Rng;
use rayon::slice::ParallelSliceMut;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::{fs, panic};

use super::logcloud_common::{get_all_types, get_type};
use crate::lava::error::LavaError;

const CHUNK_SIZE: usize = 67108864;
// const CHUNK_SIZE: usize = 268435456;
// const TOTAL_SAMPLE_LINES: usize = 3000000;
const TOTAL_SAMPLE_SIZE: usize = 1_600_000_000;
const OUTLIER_THRESHOLD: usize = 1000;

extern "C" {
    fn trainer_wrapper(sample_str: *const c_char, output_path: *const c_char) -> c_int;
    fn compressor_wrapper(
        chunk: *const c_char,
        output_path: *const c_char,
        template_path: *const c_char,
        prefix: c_int,
    ) -> c_int;
}

fn trainer_wrapper_rust(sample_str: &str, output_path: &str) -> PyResult<()> {
    let sample_str = remove_null_bytes(&sample_str);
    let sample_str_c = CString::new(sample_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        .unwrap();
    // strip blackslashes from the sample_str
    let output_path_c = CString::new(output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        .unwrap();

    let result = panic::catch_unwind(|| unsafe {
        let result = trainer_wrapper(sample_str_c.as_ptr(), output_path_c.as_ptr());
        if result != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "trainer_wrapper_c failed",
            ));
        }
        Ok(())
    });

    match result {
        Ok(_) => println!("Function completed successfully"),
        Err(_) => println!("An exception occurred"),
    }
    Ok(())
}

fn remove_null_bytes(s: &str) -> String {
    if s.as_bytes().contains(&0) {
        s.chars().filter(|&c| c != '\0').collect()
    } else {
        s.to_string()
    }
}

fn compressor_wrapper_rust(
    chunk: &str,
    output_path: &str,
    template_path: &str,
    prefix: i32,
) -> PyResult<()> {
    let chunk_c = CString::new(remove_null_bytes(chunk))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        .unwrap();
    let output_path_c = CString::new(output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        .unwrap();
    let template_path_c = CString::new(template_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        .unwrap();

    unsafe {
        let result = compressor_wrapper(
            chunk_c.as_ptr(),
            output_path_c.as_ptr(),
            template_path_c.as_ptr(),
            prefix as c_int,
        );
        if result != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "compressor_wrapper_c failed",
            ));
        }
    }
    println!("compressor_wrapper_rust done");
    Ok(())
}

fn get_variable_info(
    total_chunks: usize,
    group_number: usize,
) -> PyResult<(
    HashMap<usize, HashSet<(i32, i32)>>,
    HashMap<i32, Vec<(i32, i32)>>,
)> {
    let mut variable_to_type = HashMap::new();
    let mut chunk_variables: HashMap<usize, HashSet<(i32, i32)>> = HashMap::new();
    let mut eid_to_variables: HashMap<i32, HashSet<(i32, i32)>> = HashMap::new();

    for chunk in 0..total_chunks {
        let variable_tag_file = format!("compressed/{}/variable_{}_tag.txt", group_number, chunk);
        let file = File::open(variable_tag_file).unwrap();
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let variable_str = parts.next().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid variable string")
            })?;
            let tag = parts
                .next()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid tag"))?
                .parse::<i32>()?;

            let mut var_parts = variable_str.split('_');

            let a_part = var_parts.next().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid variable format")
            })?;
            let a = a_part
                .chars()
                .skip_while(|c| !c.is_digit(10))
                .collect::<String>()
                .parse::<i32>()
                .map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Invalid integer in variable format",
                    )
                })?;

            let b_part = var_parts.next().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid variable format")
            })?;
            let b = b_part
                .chars()
                .skip_while(|c| !c.is_digit(10))
                .collect::<String>()
                .parse::<i32>()
                .map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Invalid integer in variable format",
                    )
                })?;

            let variable = (a, b);
            variable_to_type.insert(variable, tag);
            chunk_variables.entry(chunk).or_default().insert(variable);
            eid_to_variables.entry(a).or_default().insert(variable);
        }
    }

    let eid_to_variables = eid_to_variables
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect();

    Ok((chunk_variables, eid_to_variables))
}

fn compress_chunk(
    chunk_file_counter: usize,
    current_chunk: &str,
    template_name: &str,
    group_number: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir_name = format!("variable_{}", chunk_file_counter);
    let tag_name = format!("variable_{}_tag.txt", chunk_file_counter);

    // Remove existing directory and file
    let dir_path = Path::new(&dir_name);
    if dir_path.exists() {
        std::fs::remove_dir_all(&dir_path)?;
    }
    let tag_path = Path::new(&tag_name);
    if tag_path.exists() {
        std::fs::remove_file(&tag_path)?;
    }

    // Create the directory
    std::fs::create_dir_all(&dir_path)?;

    println!("compressing chunk");

    let chunk_filename = format!("compressed/{}/chunk{:04}", group_number, chunk_file_counter);
    compressor_wrapper_rust(
        current_chunk,
        &chunk_filename,
        template_name,
        chunk_file_counter as i32,
    )?;

    // Rename files
    let source_dir = dir_path;
    let target_dir = Path::new("compressed")
        .join(group_number.to_string())
        .join(format!("variable_{}", chunk_file_counter));
    println!("source_dir: {:?}", source_dir);
    println!("target_dir: {:?}", target_dir);
    std::fs::rename(&source_dir, &target_dir)?;

    let source_tag = tag_path;
    let target_tag = Path::new("compressed")
        .join(group_number.to_string())
        .join(format!("variable_{}_tag.txt", chunk_file_counter));

    if !source_tag.exists() {
        return Err(format!("Source tag file does not exist: {:?}", source_tag).into());
    }

    if target_tag.exists() {
        println!("Target tag file already exists: {:?}", target_tag);
    }
    println!("source_tag: {:?}", source_tag);
    println!("target_tag: {:?}", target_tag);
    std::fs::rename(&source_tag, &target_tag)?;

    println!("compress_chunk done");
    Ok(())
}

fn is_valid_timestamp(timestamp: u64) -> bool {
    let min_valid_timestamp: u64 = 946684800; // January 1, 2000, 00:00:00 UTC
    let max_valid_timestamp: u64 = 2524608000; // January 1, 2050, 00:00:00 UTC
    timestamp >= min_valid_timestamp && timestamp < max_valid_timestamp
}

#[tokio::main]
pub async fn compress_logs(
    array: ArrayData,
    uid: ArrayData,
    index_name: String,
    group_number: usize,
    timestamp_bytes: Option<usize>,
    timestamp_format: Option<String>,
) -> Result<(), LavaError> {
    let extract_timestamp = timestamp_bytes.is_some() && timestamp_format.is_some();

    let array = make_array(array);
    // let uid = make_array(ArrayData::from_pyarrow(uid)?);
    let uid = make_array(uid);
    let array: &arrow_array::GenericByteArray<arrow_array::types::GenericStringType<i64>> = array
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or(LavaError::Parse(
            "Expects string array as first argument".to_string(),
        ))?;

    let uid: &arrow_array::PrimitiveArray<arrow::datatypes::UInt64Type> = uid
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(LavaError::Parse(
            "Expects uint64 array as second argument".to_string(),
        ))?;

    if array.len() != uid.len() {
        return Err(LavaError::Parse(
            "The length of the array and the uid array must be the same".to_string(),
        ));
    }

    let mut logs1 = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        let log_line = array.value(i).trim().replace("\\", "");
        logs1.push(log_line);
    }
    let logs: Vec<&str> = logs1.iter().map(|s| s.as_str()).collect();

    let mut inds = Vec::with_capacity(array.len());
    for i in 0..uid.len() {
        inds.push(uid.value(i) as usize);
    }

    let template_prefix = format!("compressed/{}_{}", index_name, group_number);
    let mut samples = Vec::new();
    let mut sample_total = 0;
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_uids = Vec::new();
    let mut current_uids = Vec::new();

    let mut global_line_count = 0;
    let mut rng = rand::thread_rng();

    std::fs::create_dir_all(format!("compressed/{}", group_number))?;

    let mut epoch_ts_vector = Vec::new();

    let mut last_timestamp = 0;

    for (line, ind) in logs.into_iter().zip(inds.into_iter()) {
        if line.is_empty() {
            continue;
        }
        if extract_timestamp {
            let timestamp_bytes = timestamp_bytes.unwrap();
            let timestamp_format = timestamp_format.clone().unwrap();
            // Attempt to parse the timestamp
            let mut epoch_ts = if line.len() >= timestamp_bytes {
                let extract_timestamp_from_this_line = &line[..timestamp_bytes];
                match NaiveDateTime::parse_from_str(
                    extract_timestamp_from_this_line.trim(),
                    &timestamp_format,
                ) {
                    Ok(dt) => dt.timestamp() as u64,
                    Err(_) => last_timestamp,
                }
            } else {
                last_timestamp
            };

            // Check if the timestamp is valid
            if !is_valid_timestamp(epoch_ts) {
                if last_timestamp == 0 {
                    eprintln!("Unable to backfill timestamp for a log line, most likely because the start of a file does not contain valid timestamp");
                    eprintln!("This will lead to wrong extracted timestamps");
                    eprintln!(
                        "Attempted to parse '{}' with '{}'",
                        &line[..std::cmp::min(timestamp_bytes, line.len())],
                        timestamp_format
                    );
                }
                // Use last_timestamp even if it's 0
                epoch_ts = last_timestamp;
            } else {
                // Update last_timestamp with the valid timestamp
                last_timestamp = epoch_ts;
            }
            epoch_ts_vector.push(epoch_ts);
        }
        if sample_total < TOTAL_SAMPLE_SIZE {
            samples.push(line);
            sample_total += line.len();
        } else {
            let j = rng.gen_range(0..global_line_count);
            if j < samples.len() {
                sample_total -= samples[j].len();
                sample_total += line.len();
                samples[j] = line;
            }
        }

        current_chunk.push_str(&line);
        current_chunk.push('\n');
        current_uids.push(ind);

        // Check if the current chunk has reached the maximum size
        if current_chunk.len() >= CHUNK_SIZE {
            chunks.push(std::mem::take(&mut current_chunk));
            chunk_uids.push(std::mem::take(&mut current_uids));
        }

        global_line_count += 1;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
        chunk_uids.push(current_uids);
    }

    let samples_str = samples.join("\n");
    //write out samples_str to debug file
    // let mut samples_file = File::create(format!("compressed/{}_samples.txt", group_number))?;
    // samples_file.write_all(samples_str.as_bytes())?;
    // samples_file.flush()?;

    let _ = trainer_wrapper_rust(&samples_str, &template_prefix).unwrap();

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        compress_chunk(chunk_index, chunk, &template_prefix, group_number).unwrap();
    }
    println!("Finished compressing chunks");

    /*
    Now go compress them chunks
    */

    let total_chunks = chunk_uids.len();
    let (chunk_variables, eid_to_variables) =
        get_variable_info(total_chunks, group_number).unwrap();

    let mut touched_types = std::collections::HashSet::new();

    let mut expanded_items: std::collections::HashMap<i32, Vec<String>> =
        std::collections::HashMap::new();
    let mut expanded_lineno: std::collections::HashMap<i32, Vec<usize>> =
        std::collections::HashMap::new();

    println!("total_chunks: {}", total_chunks);

    for chunk in 0..total_chunks {
        let mut variable_files = std::collections::HashMap::new();
        let mut variable_idx = std::collections::HashMap::new();
        for &variable in chunk_variables
            .get(&chunk)
            .unwrap_or(&std::collections::HashSet::new())
        {
            let file_path = format!(
                "compressed/{}/variable_{}/E{}_V{}",
                group_number, chunk, variable.0, variable.1
            );
            let file_content = fs::read_to_string(file_path).unwrap();
            let lines = file_content
                .lines()
                .map(String::from)
                .collect::<Vec<String>>();
            variable_files.insert(variable, lines);
            variable_idx.insert(variable, 0);
        }

        let chunk_filename = format!("compressed/{}/chunk{:04}.eid", group_number, chunk);
        let eid_file = std::fs::File::open(chunk_filename).unwrap();
        let eid_reader = std::io::BufReader::new(eid_file);

        for (idx, line) in eid_reader.lines().enumerate() {
            let eid = line.unwrap().parse::<i32>().unwrap();
            if eid < 0 || !eid_to_variables.contains_key(&eid) {
                continue;
            }
            let this_variables = eid_to_variables.get(&eid).unwrap();
            let mut type_vars = std::collections::HashMap::new();

            for &variable in this_variables {
                let item =
                    variable_files.get_mut(&variable).unwrap()[variable_idx[&variable]].to_string();
                variable_idx.entry(variable).and_modify(|v| *v += 1);
                let t = get_type(&item);
                if t == 0 {
                    eprintln!(
                        "WARNING, null variable detected in LogCrisp. {} {} {} This variable is not indexed.",
                        chunk, variable.0, variable.1
                    );
                    continue;
                }
                touched_types.insert(t);
                type_vars.entry(t).or_insert_with(Vec::new).push(item);
            }

            for (&t, items) in &type_vars {
                // println!("{} {} {}", chunk, t, items.len());
                expanded_items
                    .entry(t)
                    .or_default()
                    .extend(items.iter().cloned());
                expanded_lineno
                    .entry(t)
                    .or_default()
                    .extend(std::iter::repeat(chunk_uids[chunk][idx]).take(items.len()));
            }
        }
    }

    // Process and write compacted types and outliers
    let mut compacted_type_files = std::collections::HashMap::new();
    let mut compacted_lineno_files = std::collections::HashMap::new();
    let mut outlier_file =
        std::fs::File::create(format!("compressed/{}/outlier", group_number)).unwrap();
    let mut outlier_lineno_file =
        std::fs::File::create(format!("compressed/{}/outlier_lineno", group_number)).unwrap();
    let mut outlier_items = Vec::new();
    let mut outlier_lineno = Vec::new();

    for &t in &touched_types {
        if expanded_items[&t].is_empty() {
            panic!(
                "Error in variable extraction. No items detected for type {}",
                t
            );
        }

        let mut paired: Vec<_> = expanded_items[&t]
            .iter()
            .zip(expanded_lineno[&t].iter())
            .collect();
        paired.par_sort_unstable_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(b.1)));

        let mut compacted_items = Vec::new();
        let mut compacted_lineno = Vec::new();
        let mut last_item = String::new();

        for (item, &lineno) in paired {
            if item != &last_item {
                compacted_items.push(item.clone());
                compacted_lineno.push(vec![lineno]);
                last_item = item.clone();
            } else if lineno != *compacted_lineno.last().unwrap().last().unwrap() {
                compacted_lineno.last_mut().unwrap().push(lineno);
            }
        }

        if compacted_items.len() > OUTLIER_THRESHOLD {
            let type_file = compacted_type_files.entry(t).or_insert_with(|| {
                std::fs::File::create(format!("compressed/{}/compacted_type_{}", group_number, t))
                    .unwrap()
            });
            let lineno_file = compacted_lineno_files.entry(t).or_insert_with(|| {
                std::fs::File::create(format!(
                    "compressed/{}/compacted_type_{}_lineno",
                    group_number, t
                ))
                .unwrap()
            });

            for (item, linenos) in compacted_items.iter().zip(compacted_lineno.iter()) {
                writeln!(type_file, "{}", item).unwrap();
                writeln!(
                    lineno_file,
                    "{}",
                    linenos
                        .iter()
                        .map(|&n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                )
                .unwrap();
            }
        } else {
            outlier_items.extend(compacted_items);
            outlier_lineno.extend(compacted_lineno);
        }
    }

    // Sort and write outliers
    let mut paired: Vec<_> = outlier_items
        .into_iter()
        .zip(outlier_lineno.into_iter())
        .collect();
    paired.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (item, linenos) in paired {
        writeln!(outlier_file, "{}", item).unwrap();
        writeln!(
            outlier_lineno_file,
            "{}",
            linenos
                .iter()
                .map(|&n| n.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )
        .unwrap();
    }

    // flush the files
    for mut file in compacted_type_files.values() {
        file.flush().unwrap();
    }
    for mut file in compacted_lineno_files.values() {
        file.flush().unwrap();
    }
    outlier_file.flush().unwrap();
    outlier_lineno_file.flush().unwrap();

    Ok(())
}
