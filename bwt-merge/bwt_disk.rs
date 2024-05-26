use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Result};
use bit_vec::BitVec;
use opendal::{raw::oio::ReadExt, services::Fs, Error, Operator, Reader};
use rand::seq::SliceRandom;

use crate::bwt::run_bwt;

// generate subsets of input file of certain sizes using naive algorithm
// and calculate bwt and write to file
const SIZES: &[usize] = &[
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    2097152,
    usize::MAX,
];
pub fn generate_test_files(input_file: &str, output_path: &str) {
    let line_count = std::fs::read_to_string(input_file).unwrap().lines().count();
    let mut rng = rand::thread_rng();

    for xsize in SIZES {
        // time
        let start = std::time::Instant::now();

        let mut size = *xsize;
        if size == usize::MAX {
            size = line_count / 2;
        }

        // generate random sample indices
        let mut sample_indices: Vec<usize> = (0..line_count).collect();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(size * 2);

        // create two output files
        for ind in 0..2 {
            let mut indices = sample_indices[ind * size..(ind + 1) * size].to_vec();
            indices.sort();

            let mut strs: Vec<u8> = Vec::new();
            let mut indices_i = 0;
            for (i, line) in std::fs::read_to_string(input_file)
                .unwrap()
                .lines()
                .enumerate()
            {
                if indices_i < indices.len() && i == indices[indices_i] {
                    strs.extend_from_slice(line.as_bytes());
                    strs.push(b'\n');
                    indices_i += 1;
                }
            }

            let bwt = run_bwt(&strs);

            let output_file = format!("{}/{}_{}.bwt", output_path, size, ind);
            let output_index_file = format!("{}/{}_{}.index", output_path, size, ind);
            let output_counts_file = format!("{}/{}_{}.counts", output_path, size, ind);

            std::fs::write(output_file, bwt.0).unwrap();
            let index_str = bwt
                .1
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join("\n")
                + "\n";
            std::fs::write(output_index_file, index_str).unwrap();
            let counts_str = bwt
                .2
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join("\n")
                + "\n";
            std::fs::write(output_counts_file, counts_str).unwrap();
        }

        // generate full sampled file
        sample_indices.sort();

        let mut strs: Vec<u8> = Vec::new();
        let mut indices_i = 0;
        for (i, line) in std::fs::read_to_string(input_file)
            .unwrap()
            .lines()
            .enumerate()
        {
            if indices_i < sample_indices.len() && i == sample_indices[indices_i] {
                strs.extend_from_slice(line.as_bytes());
                strs.push(b'\n');
                indices_i += 1;
            }
        }

        let output_file = format!("{}/{}_full.txt", output_path, size);
        std::fs::write(output_file, strs).unwrap();
        println!("generated test files for size {}", size);

        let duration = start.elapsed();
        println!("time to generate for size {}: {:?}", size, duration);
    }
}

// Size of buffer for reading files
const BUFFER_SIZE: usize = 1024 * 1024;

// Compute the interleave of two BWTs, using opendal Readers
async fn compute_interleave(
    bwt0_reader: &mut Reader,
    bwt1_reader: &mut Reader,
    lens: (usize, usize),
    counts: &[usize; 256],
) -> Result<BitVec, Error> {
    let (bwt0_len, bwt1_len) = lens;

    // construct character starts array
    let mut starts: [usize; 256] = [0; 256];
    let mut sum = 0;
    for i in 0..256 {
        starts[i] = sum;
        sum += counts[i];
    }

    let mut interleave = BitVec::from_elem(bwt0_len + bwt1_len, true);
    for i in 0..bwt0_len {
        interleave.set(i, false);
    }

    let mut interleave_iterations = 0;

    loop {
        let mut ind: [usize; 2] = [0, 0];

        // reset readers
        bwt0_reader.seek(std::io::SeekFrom::Start(0)).await?;
        bwt1_reader.seek(std::io::SeekFrom::Start(0)).await?;

        let mut bwt0 = vec![0u8; BUFFER_SIZE];
        let mut bwt1 = vec![0u8; BUFFER_SIZE];
        bwt0_reader.read(&mut bwt0).await?;
        bwt1_reader.read(&mut bwt1).await?;

        let mut offsets = starts;
        let mut new_interleave = BitVec::from_elem(interleave.len(), false);
        for i in 0..interleave.len() {
            if interleave[i] {
                new_interleave.set(offsets[bwt1[ind[1]] as usize], true);
                offsets[bwt1[ind[1]] as usize] += 1;
                ind[1] += 1;

                if ind[1] == BUFFER_SIZE {
                    bwt1_reader.read(&mut bwt1).await?;
                    ind[1] = 0;
                }
            } else {
                offsets[bwt0[ind[0]] as usize] += 1;
                ind[0] += 1;

                if ind[0] == BUFFER_SIZE {
                    bwt0_reader.read(&mut bwt0).await?;
                    ind[0] = 0;
                }
            }
        }

        interleave_iterations += 1;

        if new_interleave == interleave {
            break;
        }
        interleave = new_interleave;
    }

    println!("interleave iterations: {}", interleave_iterations);
    Ok(interleave)
}

// get operator and readers for filesystem
fn get_operator() -> std::io::Result<Operator> {
    let mut builder = Fs::default();
    let current_path = std::env::current_dir()?;
    builder.root(current_path.to_str().expect("Current path not found"));
    Ok(Operator::new(builder)?.finish())
}

async fn get_file_reader(
    path: &str,
    operator: &Operator,
) -> Result<opendal::Reader, opendal::Error> {
    operator.clone().reader(path).await
}

async fn read_file_size(path: &str, operator: &Operator) -> Result<usize> {
    let len = operator.stat(path).await?.content_length() as usize;
    Ok(len)
}

// Read integers from a Reader
// extra_num is the number formed by the last digits read
async fn read_ints(reader: &mut Reader, extra_num: usize) -> Result<(Vec<usize>, usize)> {
    let mut buf = vec![0u8; BUFFER_SIZE];
    reader.read(&mut buf).await?;

    let mut ints = Vec::new();
    let mut cur_num: usize = extra_num;

    for chr in buf {
        if chr == b'\n' {
            ints.push(cur_num);
            cur_num = 0;
        } else if chr.is_ascii_digit() {
            cur_num = cur_num * 10 + (chr - b'0') as usize;
        } else if chr == 0 {
            break;
        } else {
            return Err(anyhow!("Invalid character in index file"));
        }
    }

    Ok((ints, cur_num))
}

// Merge two BWTs using our algorithm.
// Paths should be the paths to the extensionless files
pub async fn bwt_merge_disk(bwt0_path: &str, bwt1_path: &str, output_path: &str) -> Result<()> {
    // construct character counts array
    let mut counts: [usize; 256] = [0; 256];

    let operator = get_operator()?;
    let mut counts0_reader =
        get_file_reader(format!("{}.counts", bwt0_path).as_str(), &operator).await?;
    let mut counts1_reader =
        get_file_reader(format!("{}.counts", bwt1_path).as_str(), &operator).await?;
    let mut buf: Vec<u8> = Vec::new();
    counts0_reader.read_to_end(&mut buf).await?;
    let counts0 = buf
        .split(|&x| x == b'\n')
        .filter(|x| !x.is_empty())
        .map(|x| std::str::from_utf8(x).unwrap().parse().unwrap())
        .collect::<Vec<usize>>();
    buf.clear();
    counts1_reader.read_to_end(&mut buf).await?;
    let counts1 = buf
        .split(|&x| x == b'\n')
        .filter(|x| !x.is_empty())
        .map(|x| std::str::from_utf8(x).unwrap().parse().unwrap())
        .collect::<Vec<usize>>();

    assert!(
        counts0.len() == 256 && counts1.len() == 256,
        "Invalid counts file"
    );
    for i in 0..256 {
        counts[i] = counts0[i] + counts1[i];
    }
    let num_newlines = counts0[b'\n' as usize];

    // get bwt file paths
    let bwt0_file_path = format!("{}.bwt", bwt0_path);
    let bwt1_file_path = format!("{}.bwt", bwt1_path);

    let mut bwt0_reader = get_file_reader(bwt0_file_path.as_str(), &operator).await?;
    let mut bwt1_reader = get_file_reader(bwt1_file_path.as_str(), &operator).await?;
    let bwt0_len = read_file_size(bwt0_file_path.as_str(), &operator).await?;
    let bwt1_len = read_file_size(bwt1_file_path.as_str(), &operator).await?;

    let start = std::time::Instant::now();
    let interleave = compute_interleave(
        &mut bwt0_reader,
        &mut bwt1_reader,
        (bwt0_len, bwt1_len),
        &counts,
    )
    .await?;
    let duration = start.elapsed();
    println!("interleave time: {:?}", duration);

    // construct bwt
    bwt0_reader.seek(std::io::SeekFrom::Start(0)).await?;
    bwt1_reader.seek(std::io::SeekFrom::Start(0)).await?;
    let mut bwt0 = vec![0u8; BUFFER_SIZE];
    let mut bwt1 = vec![0u8; BUFFER_SIZE];
    bwt0_reader.read(&mut bwt0).await?;
    bwt1_reader.read(&mut bwt1).await?;

    // read line index
    let line_ind0_path = format!("{}.index", bwt0_path);
    let line_ind1_path = format!("{}.index", bwt1_path);
    let mut line_ind0_reader = get_file_reader(line_ind0_path.as_str(), &operator).await?;
    let mut line_ind1_reader = get_file_reader(line_ind1_path.as_str(), &operator).await?;
    let (mut line_ind0, mut extra_num0) = read_ints(&mut line_ind0_reader, 0).await?;
    let (mut line_ind1, mut extra_num1) = read_ints(&mut line_ind1_reader, 0).await?;
    let mut line_ind0_iter = line_ind0.iter();
    let mut line_ind1_iter = line_ind1.iter();

    let output_bwt_path = format!("{}.bwt", output_path);
    let output_index_path = format!("{}.index", output_path);
    let mut bwt_writer = BufWriter::new(File::create(output_bwt_path.as_str())?);
    let mut index_writer = BufWriter::new(File::create(output_index_path.as_str())?);

    let mut ind0 = 0;
    let mut ind1 = 0;

    for i in 0..interleave.len() {
        if interleave[i] {
            bwt_writer.write_all(&[bwt1[ind1]])?;

            let line_ind_opt = line_ind1_iter.next();
            let line_ind: usize;
            if line_ind_opt.is_none() {
                (line_ind1, extra_num1) = read_ints(&mut line_ind1_reader, extra_num1).await?;
                line_ind1_iter = line_ind1.iter();
                line_ind = *line_ind1_iter.next().expect("Line index is too short");
            } else {
                line_ind = *line_ind_opt.unwrap();
            }
            writeln!(index_writer, "{}", line_ind + num_newlines)?;

            ind1 += 1;
            if ind1 == BUFFER_SIZE {
                bwt1_reader.read(&mut bwt1).await?;
                ind1 = 0;
            }
        } else {
            bwt_writer.write_all(&[bwt0[ind0]])?;

            let line_ind_opt = line_ind0_iter.next();
            let line_ind: usize;
            if line_ind_opt.is_none() {
                (line_ind0, extra_num0) = read_ints(&mut line_ind0_reader, extra_num0).await?;
                line_ind0_iter = line_ind0.iter();
                line_ind = *line_ind0_iter.next().unwrap();
            } else {
                line_ind = *line_ind_opt.unwrap();
            }
            writeln!(index_writer, "{}", line_ind)?;

            ind0 += 1;
            if ind0 == BUFFER_SIZE {
                bwt0_reader.read(&mut bwt0).await?;
                ind0 = 0;
            }
        }
    }

    // write counts
    let output_counts_path = format!("{}.counts", output_path);
    let counts_data = counts
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join("\n")
        + "\n";
    std::fs::write(output_counts_path, counts_data)?;

    Ok(())
}

pub async fn test_merge_disk(input_path: &str, output_path: &str, test_rebuild: bool) {
    let mut test_sizes: Vec<usize> = SIZES[0..SIZES.len() - 1].to_vec();
    test_sizes.push(3719388); // full size

    let operator = get_operator().unwrap();

    for size in test_sizes.iter() {
        let bwt0_path = format!("{}/{}_0", input_path, size);
        let bwt1_path = format!("{}/{}_1", input_path, size);
        let output_path_n = format!("{}/{}_merged", output_path, size);

        // time merge
        let merge_start = std::time::Instant::now();
        bwt_merge_disk(&bwt0_path, &bwt1_path, &output_path_n)
            .await
            .unwrap();
        let merge_duration = merge_start.elapsed();
        println!("merge time for size {}: {:?}", size, merge_duration);

        if test_rebuild {
            // time full rebuild, including i/o times
            let rebuild_start = std::time::Instant::now();
            let path = format!("{}.bwt", output_path_n);
            let mut reader = get_file_reader(path.as_str(), &operator).await.unwrap();
            let mut full_text = Vec::new();
            reader.read_to_end(&mut full_text).await.unwrap();
            let full_bwt = run_bwt(&full_text);

            let output_path = format!("{}/{}_merged_naive", output_path, size);
            std::fs::write(format!("{}.bwt", output_path), full_bwt.0).unwrap();
            std::fs::write(
                format!("{}.index", output_path),
                full_bwt
                    .1
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join("\n"),
            )
            .unwrap();
            std::fs::write(
                format!("{}.counts", output_path),
                full_bwt
                    .2
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join("\n"),
            )
            .unwrap();
            let rebuild_duration = rebuild_start.elapsed();
            println!("rebuild time for size {}: {:?}", size, rebuild_duration);
        }
    }
}
