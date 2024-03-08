use bwt_merge::bwt::{bwt_merge, fm_index, get_matching_lines, run_bwt};
use clap::Parser;
use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Cli {
    /// Mode to generate several BWT files
    #[arg(short, long)]
    generate: bool,

    /// Mode to test BWT merge on disk
    #[arg(short, long)]
    test_disk: bool,

    /// Whether to test rebuild
    #[arg(short, long)]
    rebuild: bool,

    /// Input file
    #[arg(short, long, value_name = "FILE")]
    input_file: Option<PathBuf>,

    /// Whether to print BWT
    /// Default: false
    #[arg(short, long, default_value_t = false)]
    print_bwt: bool,

    /// Query string
    #[arg(short, long, value_name = "STRING")]
    query: Option<String>,

    /// Mode to generate trie
    #[arg(long)]
    create_trie: bool,

    /// Mode to compare zstd size
    #[arg(long)]
    compare_zstd: bool,

    #[arg(long, value_name = "FILE")]
    trie_file: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if cli.generate {
        let input_file = cli.input_file.unwrap();
        let output_path = "./data/tests";
        bwt_merge::bwt_disk::generate_test_files(input_file.to_str().unwrap(), output_path);
        return;
    }

    if cli.test_disk {
        let input_path = "./data/tests";
        let output_path = "./data/test_out_new";
        bwt_merge::bwt_disk::test_merge_disk(input_path, output_path, cli.rebuild).await;
        return;
    }

    let mut input_lines: Vec<Vec<u8>>;
    if let Some(input_file) = cli.input_file {
        // read from file
        let input = std::fs::read(input_file).unwrap();
        input_lines = input.split(|&x| x == b'\n').map(|x| x.to_vec()).collect();
    } else {
        // read from stdin
        let stdin = io::stdin();
        input_lines = Vec::new();
        for line in stdin.lock().lines() {
            input_lines.push(line.unwrap().as_bytes().to_vec());
        }
    }

    if cli.create_trie || cli.compare_zstd || cli.trie_file.is_some() {
        let trie;
        if let Some(trie_file) = cli.trie_file {
            println!("reading file");
            let trie_data = std::fs::read(trie_file).unwrap();
            // decompress
            println!("decompressing");
            let mut decoder = zstd::stream::Decoder::new(&trie_data[..]).unwrap();
            let mut trie_data = Vec::new();
            decoder.read_to_end(&mut trie_data).unwrap();
            println!("deserializing");
            trie = postcard::from_bytes(&trie_data).unwrap();
        } else {
            // build trie
            let input_inds: Vec<Vec<usize>> = (0..input_lines.len()).map(|x| vec![x]).collect();

            let start_time = Instant::now();
            trie = bwt_merge::trie::build_binary_trie(&input_lines, &input_inds);
            let trie_duration = start_time.elapsed();
            println!("trie build time: {:?}", trie_duration);
        }

        if cli.query.is_some() {
            // try a query
            let query_str = cli.query.unwrap();
            let query = query_str.as_bytes().to_vec();
            let res = bwt_merge::trie::query_string(&trie, &query);
            println!("trie res: {:?}", res);
        }

        if cli.compare_zstd {
            // serialize trie
            let serialized = postcard::to_stdvec(&trie).unwrap();

            // compress
            println!("original trie size: {}", serialized.len());
            let mut encoder = zstd::stream::Encoder::new(Vec::new(), 0).unwrap();
            encoder.write_all(&serialized).unwrap();
            let compressed = encoder.finish().unwrap();
            println!("compressed size: {}", compressed.len());
            std::fs::write("data/trie.zstd", compressed).unwrap();

            // find compressed size of bwt
            let mut bwt_manual = input_lines.join(&b'\n');
            bwt_manual.push(b'\n');

            println!("starting bwt build");
            let start_time = Instant::now();
            let data = run_bwt(&bwt_manual);
            let bwt_duration = start_time.elapsed();
            println!("bwt build time: {:?}", bwt_duration);

            // exclude indices
            let serialized_bwt = postcard::to_stdvec(&data.0).unwrap();
            println!("original bwt size: {}", serialized_bwt.len());
            let mut encoder = zstd::stream::Encoder::new(Vec::new(), 0).unwrap();
            encoder.write_all(&serialized_bwt).unwrap();
            let compressed_bwt = encoder.finish().unwrap();
            println!("compressed bwt size: {}", compressed_bwt.len());

            let counts_vec = data.2.to_vec();
            let serialize_data = (data.0, data.1, counts_vec);
            let serialized_bwt = postcard::to_stdvec(&serialize_data).unwrap();
            println!("original bwt size, with indices: {}", serialized_bwt.len());
            let mut encoder = zstd::stream::Encoder::new(Vec::new(), 0).unwrap();
            encoder.write_all(&serialized_bwt).unwrap();
            let compressed_bwt = encoder.finish().unwrap();
            println!(
                "compressed bwt size, with indices: {}",
                compressed_bwt.len()
            );
        }

        return;
    }

    // split input into two lists
    let mut input0 = Vec::new();
    let mut input1 = Vec::new();
    for (i, line) in input_lines.iter().enumerate() {
        if i < input_lines.len() / 2 {
            input0.push(line.clone());
        } else {
            input1.push(line.clone());
        }
    }

    // concatenate inputs separated by newline
    let mut input0_concat = input0.join(&b'\n');
    input0_concat.push(b'\n');
    let mut input1_concat = input1.join(&b'\n');
    input1_concat.push(b'\n');

    let data0 = run_bwt(&input0_concat);
    let data1 = run_bwt(&input1_concat);

    // custom bwt merge
    // note: bwt build time not included
    let bwt_merge_start = Instant::now();
    let data_merge = bwt_merge(&data0, &data1);
    let bwt_merge_duration = bwt_merge_start.elapsed();
    let bwt_str = String::from_utf8(data_merge.0.clone()).unwrap();
    if cli.print_bwt {
        println!("{}", bwt_str);
    }
    println!("bwt merge time: {:?}", bwt_merge_duration);

    let mut bwt_manual = input_lines.join(&b'\n');
    bwt_manual.push(b'\n');

    // lib bwt construction
    let bwt_lib_start = Instant::now();
    let test_data = run_bwt(&bwt_manual);
    let bwt_lib_duration = bwt_lib_start.elapsed();
    let test_bwt_str = String::from_utf8(test_data.0.clone()).unwrap();
    if cli.print_bwt {
        println!("{}", test_bwt_str);
    }
    println!("bwt lib time: {:?}", bwt_lib_duration);

    // build fm indices
    let merge_index = fm_index(&data_merge);
    let test_index = fm_index(&test_data);

    // try some queries
    let query_str = cli.query.unwrap_or("a".to_string());
    println!("Querying for string '{}'", query_str);
    let query = query_str.as_bytes().to_vec();
    let merge_res = get_matching_lines(&data_merge, &merge_index, &query);
    let test_res = get_matching_lines(&test_data, &test_index, &query);
    println!("merge res: {:?}", merge_res);
    println!("test res: {:?}", test_res);
}
