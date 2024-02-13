use anyhow::anyhow;
use arrow::datatypes::ToByteSlice;
use arrow::error::ArrowError;
use arrow_array::{Array, StringArray};
use parquet::{
    arrow::array_reader::make_byte_array_reader, basic::{Encoding, Type}, column::page::Page, compression::{create_codec, Codec, CodecOptionsBuilder}, data_type::AsBytes, errors::ParquetError, file::{
        footer::{decode_footer, decode_metadata},
        metadata::ParquetMetaData,
        reader::*,
        statistics, FOOTER_SIZE,
    }, format::{PageHeader, PageType}, thrift::TSerializable, util::InMemoryPageIterator
};
use thrift::protocol::TCompactInputProtocol;

use opendal::raw::oio::ReadExt;
use opendal::services::{Fs, S3};
use opendal::{Operator, Reader};

use bytes::{Bytes, BytesMut};
use std::{convert::TryFrom, time::Instant};
use std::{
    fmt::Display,
    io::{self, Read, SeekFrom},
};

use futures::stream::{self, StreamExt};
use itertools::{izip, Itertools};
use std::collections::HashMap;
use std::{env, usize};
use tokio::{self};
use std::fmt;
use regex::Regex;

#[derive(Debug)]
pub enum MyError {
    ParquetError(ParquetError),
    OpendalError(opendal::Error),
    ThriftError(thrift::Error),
    // Add more variants for other errors or general cases
}

impl Display for MyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyError::ParquetError(err) => write!(f, "Parquet error: {}", err),
            MyError::OpendalError(err) => write!(f, "Opendal error: {}", err),
            MyError::ThriftError(err) => write!(f, "Thrift error: {}", err),
        }
    }
}

impl From<ParquetError> for MyError {

    fn from(e: ParquetError) -> Self {
        MyError::ParquetError(e)
    }
}

impl From<opendal::Error> for MyError {
    fn from(e: opendal::Error) -> Self {
        MyError::OpendalError(e)
    }
}

impl From<thrift::Error> for MyError {
    fn from(e: thrift::Error) -> Self {
        MyError::ThriftError(e)
    }
}

//dupilcate code for now
struct S3Builder(S3);

impl From<&str> for S3Builder {
    fn from(file: &str) -> Self {
        let mut builder = S3::default();
        let mut iter = file[5..].split("/");

        builder.bucket(iter.next().expect("malformed path"));
        // Set the region. This is required for some services, if you don't care about it, for example Minio service, just set it to "auto", it will be ignored.
        builder.region("us-west-2");
        builder.enable_virtual_host_style();
        builder.endpoint("https://tos-s3-cn-beijing.volces.com");
        S3Builder(builder)
    }
}

struct FsBuilder(Fs);

impl From<&str> for FsBuilder {
    fn from(folder: &str) -> Self {
        let mut builder = Fs::default();
        // let current_path = env::current_dir().expect("no path");
        builder.root(folder);
        FsBuilder(builder)
    }
}

struct Operators(Operator);

impl From<S3Builder> for Operators {
    fn from(builder: S3Builder) -> Self {
        Operators(
            Operator::new(builder.0)
                .expect("S3 builder construction error")
                .finish(),
        )
    }
}

impl Operators {
    fn into_inner(self) -> Operator {
        self.0
    }
}

impl From<FsBuilder> for Operators {
    fn from(builder: FsBuilder) -> Self {
        Operators(
            Operator::new(builder.0)
                .expect("Fs Builder construction error")
                .finish(),
        )
    }
}

async fn get_reader_and_size_from_file(file: &str) -> Result<(usize, Reader), MyError> {
    let mut file_name = file.to_string();
    let operator = if file.starts_with("s3://") {
        file_name = file_name.replace("s3://", "");
        let mut iter = file_name.split("/");
        let bucket = iter.next().expect("malformed s3 path");
        file_name = file_name[bucket.len() + 1 ..].to_string();

        Operators::from(S3Builder::from(file)).into_inner()
    } else {
        let current_path = env::current_dir().unwrap();
        Operators::from(FsBuilder::from(current_path.to_str().expect("no path"))).into_inner()
    };

    let file_size: usize = operator.stat(&file_name).await?.content_length() as usize;
    let reader: Reader = operator.clone().reader(&file_name).await?;

    Ok((file_size, reader))
}

async fn parse_metadata(reader: &mut Reader, file_size: usize) -> Result<ParquetMetaData, MyError> {
    // check file is large enough to hold footer

    let mut footer = [0_u8; 8];

    reader.seek(SeekFrom::End(-8)).await.unwrap();
    reader.read(&mut footer).await.unwrap();

    let metadata_len = decode_footer(&footer)?;
    let footer_metadata_len = FOOTER_SIZE + metadata_len;

    if footer_metadata_len > file_size as usize {
        return Err(MyError::from(ParquetError::General(
            "Invalid Parquet file. Size is smaller than footer".to_string(),
        )));
    }

    let start = file_size as u64 - footer_metadata_len as u64;
    let mut bytes = vec![0_u8; metadata_len];
    reader.seek(SeekFrom::Start(start)).await.unwrap();
    reader.read(&mut bytes).await.unwrap();

    decode_metadata(bytes.to_byte_slice()).map_err(|e| MyError::from(e))
}

pub(crate) fn decode_page(
    page_header: PageHeader,
    buffer: Bytes,
    physical_type: Type,
    decompressor: Option<&mut Box<dyn Codec>>,
) -> Result<Page, MyError> {
    let mut offset: usize = 0;
    let mut can_decompress = true;

    if let Some(ref header_v2) = page_header.data_page_header_v2 {
        offset = (header_v2.definition_levels_byte_length + header_v2.repetition_levels_byte_length)
            as usize;
        // When is_compressed flag is missing the page is considered compressed
        can_decompress = header_v2.is_compressed.unwrap_or(true);
    }

    // TODO: page header could be huge because of statistics. We should set a
    // maximum page header size and abort if that is exceeded.
    let buffer = match decompressor {
        Some(decompressor) if can_decompress => {
            let uncompressed_size = page_header.uncompressed_page_size as usize;
            let mut decompressed = Vec::with_capacity(uncompressed_size);
            let compressed = &buffer.as_ref()[offset..];
            decompressed.extend_from_slice(&buffer.as_ref()[..offset]);
            decompressor.decompress(
                compressed,
                &mut decompressed,
                Some(uncompressed_size - offset),
            )?;

            if decompressed.len() != uncompressed_size {
                return Err(MyError::from(ParquetError::General(
                    "messed decompression".to_string(),
                )));
            }

            Bytes::from(decompressed)
        }
        _ => buffer,
    };

    let result = match page_header.type_ {
        PageType::DICTIONARY_PAGE => {
            let dict_header = page_header.dictionary_page_header.as_ref().ok_or_else(|| {
                ParquetError::General("Missing dictionary page header".to_string())
            })?;
            let is_sorted = dict_header.is_sorted.unwrap_or(false);
            Page::DictionaryPage {
                buf: buffer,
                num_values: dict_header.num_values as u32,
                encoding: Encoding::try_from(dict_header.encoding)?,
                is_sorted,
            }
        }
        PageType::DATA_PAGE => {
            let header = page_header
                .data_page_header
                .ok_or_else(|| ParquetError::General("Missing V1 data page header".to_string()))?;
            Page::DataPage {
                buf: buffer,
                num_values: header.num_values as u32,
                encoding: Encoding::try_from(header.encoding)?,
                def_level_encoding: Encoding::try_from(header.definition_level_encoding)?,
                rep_level_encoding: Encoding::try_from(header.repetition_level_encoding)?,
                statistics: statistics::from_thrift(physical_type, header.statistics)?,
            }
        }
        PageType::DATA_PAGE_V2 => {
            let header = page_header
                .data_page_header_v2
                .ok_or_else(|| ParquetError::General("Missing V2 data page header".to_string()))?;
            let is_compressed = header.is_compressed.unwrap_or(true);
            Page::DataPageV2 {
                buf: buffer,
                num_values: header.num_values as u32,
                encoding: Encoding::try_from(header.encoding)?,
                num_nulls: header.num_nulls as u32,
                num_rows: header.num_rows as u32,
                def_levels_byte_len: header.definition_levels_byte_length as u32,
                rep_levels_byte_len: header.repetition_levels_byte_length as u32,
                is_compressed,
                statistics: statistics::from_thrift(physical_type, header.statistics)?,
            }
        }
        _ => {
            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
            unimplemented!("Page type {:?} is not supported", page_header.type_)
        }
    };

    Ok(result)
}

fn read_page_header<C: ChunkReader>(
    reader: &C,
    offset: u64,
) -> Result<(usize, PageHeader), MyError> {
    struct TrackedRead<R>(R, usize);

    impl<R: Read> Read for TrackedRead<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let v = self.0.read(buf)?;
            self.1 += v;
            Ok(v)
        }
    }

    let input = reader.get_read(offset)?;
    let mut tracked = TrackedRead(input, 0);
    let mut prot = TCompactInputProtocol::new(&mut tracked);
    let header = PageHeader::read_from_in_protocol(&mut prot)?;
    Ok((tracked.1, header))
}

async fn parse_metadatas(
    file_paths: &Vec<String>,
) -> HashMap<String, ParquetMetaData> {
    let iter = file_paths.iter().dedup();

    let handles = stream::iter(iter)
        .map(|file_path: &String| {

            let file_path = file_path.clone();
            
            tokio::spawn(async move {
                let (file_size, mut reader) = get_reader_and_size_from_file(&file_path).await.unwrap();
                let metadata = parse_metadata(&mut reader, file_size as usize)
                    .await
                    .unwrap();

                (file_path, metadata)
            })
        })
        .collect::<Vec<_>>()
        .await;
    let res = futures::future::join_all(handles).await;

    let mut metadatas = HashMap::new();

    for elem in res {
        let _ = match elem {
            Ok((k, v)) => metadatas.insert(k, v),
            Err(_) => None,
        };
    }

    metadatas
}

#[derive(Debug, Clone)]
pub struct ParquetLayout {
    pub num_row_groups: usize,
    pub dictionary_page_sizes: Vec<usize>, // 0 means no dict page
    pub data_page_sizes: Vec<usize>,
    pub data_page_offsets: Vec<usize>,
    pub data_page_num_rows: Vec<usize>,
    pub row_group_data_pages: Vec<usize>,
}

#[tokio::main]
pub async fn get_parquet_layout(
    column_name: &str,
    file_path: &str,
) -> Result<(arrow::array::ArrayData, ParquetLayout), MyError> {
    let (file_size, mut reader) = get_reader_and_size_from_file(file_path).await?;
    let metadata = parse_metadata(&mut reader, file_size as usize).await?;

    let codec_options = CodecOptionsBuilder::default()
        .set_backward_compatible_lz4(false)
        .build();

    let mut parquet_layout = ParquetLayout {
        num_row_groups: metadata.num_row_groups(),
        dictionary_page_sizes: vec![],
        data_page_sizes: vec![],
        data_page_offsets: vec![],
        data_page_num_rows: vec![],
        row_group_data_pages: vec![],
    };

    let mut pages: Vec<Vec<parquet::column::page::Page>> = Vec::new();
    let mut total_values = 0;

    let column_index = metadata.file_metadata().schema_descr().columns().iter()
        .position(|column| column.name() == column_name)
        .expect(&format!("column {} not found in parquet file {}", column_name, file_path));
    
    // @rain we should parallelize this across row groups using tokio

    for row_group in 0..metadata.num_row_groups() {
        let column = metadata.row_group(row_group).column(column_index);
        let mut start = column
            .dictionary_page_offset()
            .unwrap_or_else(|| column.data_page_offset()) as u64;
        let end = start + column.compressed_size() as u64;

        let compression_scheme = column.compression();
        let mut codec = create_codec(compression_scheme, &codec_options)
            .unwrap()
            .unwrap();

        let mut total_data_pages: usize = 0;

        // let mut column_chunk_bytes = BytesMut::with_capacity((end - start) as usize);
        let mut column_chunk_bytes = vec![0u8; (end - start) as usize];
        reader.seek(SeekFrom::Start(start as u64)).await.unwrap();
        // parallelize this please @Rain
        let mut total_read: usize = 0; 
        while total_read < (end - start) as usize {
            let read = reader.read(&mut column_chunk_bytes[total_read..]).await?;
            if read == 0 {
                // If read returns 0, it means EOF is reached
                break;
            }
            total_read += read; // Update total bytes read
        }
        let column_chunk_bytes = Bytes::from(column_chunk_bytes);

        let mut column_chunk_pages: Vec<parquet::column::page::Page> = Vec::new();

        let end = end - start;
        start = 0;

        while start != end {

            // this takes a slice of the entire thing for each page, granted it won't read the entire thing,
            // the thrift will terminate after reading the necessary things. @Rain the alternative is to feed it 
            // chunks at a time in a loop until a valid header is returned, like before how we are using the reader in rust-test

            let (header_len, header) = read_page_header(&column_chunk_bytes, start)?;
            // println!("{} {} {:?}", start, header_len, header);
            
            let page_header = header.clone();

            let mut dictionary_page_size: usize = 0;

            let page: Page = match page_header.type_ {
                PageType::DICTIONARY_PAGE => {
                    dictionary_page_size = page_header.compressed_page_size as usize + header_len;
                    let page: Page = decode_page(
                        page_header,
                        column_chunk_bytes.slice((start as usize + header_len)  .. (start as usize + dictionary_page_size as usize)),
                        Type::BYTE_ARRAY,
                        Some(&mut codec),
                    )
                    .unwrap();
                    start += dictionary_page_size as u64;
                    page
                }
                PageType::DATA_PAGE | PageType::DATA_PAGE_V2 => {
                    let compressed_page_size = page_header.compressed_page_size;
                    parquet_layout
                        .data_page_sizes
                        .push(compressed_page_size as usize + header_len);
                    parquet_layout.data_page_offsets.push(start as usize);
                    
                    parquet_layout
                        .dictionary_page_sizes
                        .push(dictionary_page_size);
                    total_data_pages += 1;
                    
                    let page = decode_page(
                            page_header,
                            column_chunk_bytes.slice((start as usize + header_len)  .. (start as usize + header_len + compressed_page_size as usize)),
                            Type::BYTE_ARRAY,
                            Some(&mut codec),
                        )
                        .unwrap();
                    
                    parquet_layout
                        .data_page_num_rows
                        .push(page.num_values() as usize);
                    total_values += page.num_values() as usize;

                    start += compressed_page_size as u64 + header_len as u64;
                    page
                }
                _ => {
                    // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                    unimplemented!("Page type {:?} is not supported", page_header.type_)
                }
            };

            column_chunk_pages.push(page);

        }

        pages.push(column_chunk_pages);
        parquet_layout.row_group_data_pages.push(total_data_pages);
    }

    let page_iterator = InMemoryPageIterator::new(pages);
    let mut array_reader = make_byte_array_reader(
        Box::new(page_iterator),
        metadata.row_group(0)
                .schema_descr()
                .column(column_index),
        None,
    )
    .unwrap();
    let array = array_reader.next_batch(total_values as usize).unwrap();

    let new_array: &arrow_array::GenericByteArray<arrow_array::types::GenericStringType<i32>> = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            ArrowError::ParseError(
                "Expects string array as first argument".to_string(),
            )
        })
        .unwrap();

    let data: arrow::array::ArrayData = new_array.to_data();
    
    Ok((data, parquet_layout))
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub file_path: String,
    pub column_index: usize,
    pub row_group: usize,
    pub offset_in_row_group: usize,
    pub matched: String,
}

#[tokio::main]
pub async fn search_indexed_pages(
    query: String,
    column_name: &str,
    file_paths: Vec<String>,
    row_groups: Vec<usize>,
    page_offsets: Vec<u64>,
    page_sizes: Vec<usize>,
    dict_page_sizes: Vec<usize>, // 0 means no dict page
) -> Result<Vec<MatchResult>, MyError> {
    // current implementation might re-read dictionary pages, this should be optimized
    // we are assuming that all the files are either on disk or cloud.
    
    let codec_options = CodecOptionsBuilder::default()
        .set_backward_compatible_lz4(false)
        .build();
    let re = Regex::new(&query).unwrap();

    let metadatas = parse_metadatas(&file_paths).await;

    let iter = izip!(
        file_paths,
        row_groups,
        page_offsets,
        page_sizes,
        dict_page_sizes
    );

    let iter: Vec<tokio::task::JoinHandle<Vec<MatchResult>>> = stream::iter(iter)
        .map(
            |(file_path, row_group, page_offset, page_size, dict_page_size)| {

                let column_index = metadatas[&file_path].file_metadata().schema_descr().columns().iter()
                    .position(|column| column.name() == column_name)
                    .expect(&format!("column {} not found in parquet file {}", column_name, file_path));
                let column_descriptor = metadatas[&file_path]
                    .row_group(row_group)
                    .schema_descr()
                    .column(column_index);
                let compression_scheme = metadatas[&file_path]
                    .row_group(row_group)
                    .column(column_index)
                    .compression();
                let dict_page_offset = metadatas[&file_path]
                    .row_group(row_group)
                    .column(column_index)
                    .dictionary_page_offset();
                let mut codec = create_codec(compression_scheme, &codec_options)
                    .unwrap()
                    .unwrap();

                
                let re = re.clone();

                let handle = tokio::spawn(async move {

                    let (file_size, mut reader) = get_reader_and_size_from_file(&file_path).await.unwrap();
                    let mut pages: Vec<parquet::column::page::Page> = Vec::new();
                    if dict_page_size > 0 {
                        let start = dict_page_offset.unwrap();
                        let mut dict_page_bytes = vec![0; dict_page_size];
                        reader.seek(SeekFrom::Start(start as u64)).await.unwrap();
                        reader.read(&mut dict_page_bytes).await.unwrap();
                        let dict_page_bytes = Bytes::from(dict_page_bytes);
                        let (dict_header_len, dict_header) =
                            read_page_header(&dict_page_bytes, 0).unwrap();
                        let dict_page = decode_page(
                            dict_header,
                            dict_page_bytes.slice(dict_header_len..dict_page_size),
                            Type::BYTE_ARRAY,
                            Some(&mut codec),
                        )
                        .unwrap();
                        pages.push(dict_page);
                    }

                    let mut page_bytes = vec![0; page_size];
                    reader.seek(SeekFrom::Start(page_offset)).await.unwrap();
                    reader.read(&mut page_bytes).await.unwrap();
                    let page_bytes = Bytes::from(page_bytes);
                    let (header_len, header) = read_page_header(&page_bytes, 0).unwrap();
                    let page: Page = decode_page(
                        header,
                        page_bytes.slice(header_len..page_size),
                        Type::BYTE_ARRAY,
                        Some(&mut codec),
                    )
                    .unwrap();
                    let num_values = page.num_values();

                    pages.push(page);
                    let page_iterator = InMemoryPageIterator::new(vec![pages]);
                    let mut array_reader = make_byte_array_reader(
                        Box::new(page_iterator),
                        column_descriptor.clone(),
                        None,
                    )
                    .unwrap();
                    let array = array_reader.next_batch(num_values as usize).unwrap();

                    let new_array = array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            ArrowError::ParseError(
                                "Expects string array as first argument".to_string(),
                            )
                        })
                        .unwrap();

                    let mut match_results: Vec<MatchResult> = vec![];

                    for i in 0..new_array.len() {
                        if re.is_match(new_array.value(i)) {
                            match_results.push(MatchResult {
                                file_path: file_path.clone(),
                                column_index: column_index,
                                row_group: row_group,
                                offset_in_row_group: i,
                                matched: new_array.value(i).to_string(),
                            })
                        }
                    }

                    match_results
                });

                handle
            },
        )
        .collect::<Vec<_>>()
        .await;

    let _res: Vec<std::prelude::v1::Result<Vec<MatchResult>, tokio::task::JoinError>> =
        futures::future::join_all(iter).await;
    let result: Result<Vec<MatchResult>, tokio::task::JoinError> =
        _res.into_iter().try_fold(Vec::new(), |mut acc, r| {
            r.map(|inner_vec| {
                acc.extend(inner_vec);
                acc
            })
        });

    result.map_err(|e| {
        // Here, you can convert `e` (a JoinError) into your custom error type.
        MyError::from(ParquetError::General(e.to_string()))
    })
}
