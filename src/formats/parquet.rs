use arrow::array::ArrayData;
use arrow::datatypes::ToByteSlice;
use arrow::error::ArrowError;
use arrow_array::{Array, BinaryArray, StringArray};

use log::debug;
use parquet::{
    arrow::array_reader::make_byte_array_reader,
    basic::{Encoding, Type},
    column::page::Page,
    compression::{create_codec, Codec, CodecOptionsBuilder},
    errors::ParquetError,
    file::{
        footer::{decode_footer, decode_metadata},
        metadata::ParquetMetaData,
        reader::*,
        statistics, FOOTER_SIZE,
    },
    format::{PageHeader, PageType},
    thrift::TSerializable,
    util::InMemoryPageIterator,
};
use thrift::protocol::TCompactInputProtocol;

use bytes::Bytes;
use std::{collections::BTreeMap, hash::Hash, io::Read};
use std::{convert::TryFrom, sync::Arc};

use futures::stream::{self, StreamExt};
use itertools::{izip, Itertools};
use std::collections::HashMap;

use tokio::{self};

use crate::{
    formats::readers::{get_file_size_and_reader, get_reader, AsyncReader},
    lava::error::LavaError,
};

use super::readers::ReaderType;
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

async fn get_metadata_bytes(
    reader: &mut AsyncReader,
    file_size: usize,
) -> Result<Bytes, LavaError> {
    // check file is large enough to hold footer

    let footer: [u8; 8] = reader
        .read_range(file_size as u64 - 8, file_size as u64)
        .await?
        .to_byte_slice()
        .try_into()
        .unwrap();

    let metadata_len = decode_footer(&footer)?;
    let footer_metadata_len = FOOTER_SIZE + metadata_len;

    if footer_metadata_len > file_size as usize {
        return Err(LavaError::from(ParquetError::General(
            "Invalid Parquet file. Size is smaller than footer".to_string(),
        )));
    }

    let start = file_size as u64 - footer_metadata_len as u64;
    let bytes = reader
        .read_range(start, start + metadata_len as u64)
        .await?;

    Ok(bytes)
}

pub(crate) fn decode_page(
    page_header: PageHeader,
    buffer: Bytes,
    physical_type: Type,
    decompressor: Option<&mut Box<dyn Codec>>,
) -> Result<Page, LavaError> {
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
                return Err(LavaError::from(ParquetError::General(
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
) -> Result<(usize, PageHeader), LavaError> {
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
    reader_type: ReaderType,
) -> HashMap<String, ParquetMetaData> {
    let iter = file_paths.iter().dedup();

    let handles = stream::iter(iter)
        .map(|file_path: &String| {
            let file_path = file_path.clone();
            let reader_type = reader_type.clone();

            tokio::spawn(async move {
                let (file_size, mut reader) =
                    get_file_size_and_reader(file_path.clone(), reader_type)
                        .await
                        .unwrap();

                let metadata_bytes = get_metadata_bytes(&mut reader, file_size as usize)
                    .await
                    .unwrap();

                let metadata = decode_metadata(metadata_bytes.to_byte_slice())
                    .map_err(LavaError::from)
                    .unwrap();
                (file_path, metadata)
            })
        })
        .collect::<Vec<_>>()
        .await;
    let res: Vec<Result<(String, ParquetMetaData), tokio::task::JoinError>> =
        futures::future::join_all(handles).await;

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
    pub metadata_bytes: Bytes,
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
    reader_type: ReaderType,
) -> Result<(Vec<arrow::array::ArrayData>, ParquetLayout), LavaError> {
    let (file_size, mut reader) =
        get_file_size_and_reader(file_path.to_string(), reader_type).await?;
    let metadata_bytes = get_metadata_bytes(&mut reader, file_size as usize).await?;
    let metadata = decode_metadata(metadata_bytes.to_byte_slice()).map_err(LavaError::from)?;

    let codec_options = CodecOptionsBuilder::default()
        .set_backward_compatible_lz4(false)
        .build();

    let mut parquet_layout = ParquetLayout {
        num_row_groups: metadata.num_row_groups(),
        metadata_bytes: metadata_bytes,
        dictionary_page_sizes: vec![],
        data_page_sizes: vec![],
        data_page_offsets: vec![],
        data_page_num_rows: vec![],
        row_group_data_pages: vec![],
    };

    let mut pages: Vec<Vec<parquet::column::page::Page>> = Vec::new();
    let mut total_values = 0;

    let column_index = metadata
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|column| column.name() == column_name)
        .expect(&format!(
            "column {} not found in parquet file {}",
            column_name, file_path
        ));

    //TODO: @rain we should parallelize this across row groups using tokio
    // this need to refactor the ParquetLayout data structure, since it won't cost too much time, postpone for now.

    for row_group in 0..metadata.num_row_groups() {
        let column = metadata.row_group(row_group).column(column_index);
        let mut start = column
            .dictionary_page_offset()
            .unwrap_or_else(|| column.data_page_offset()) as u64;
        let end = start + column.compressed_size() as u64;

        let compression_scheme = column.compression();
        let mut codec = create_codec(compression_scheme, &codec_options).unwrap();
        //.unwrap();

        let mut total_data_pages: usize = 0;

        let column_chunk_bytes = reader.read_range(start, end).await?;

        let mut column_chunk_pages: Vec<parquet::column::page::Page> = Vec::new();

        let end = end - start;
        let column_chunk_offset = start;
        start = 0;

        let mut dictionary_page_size: usize = 0;

        while start != end {
            // this takes a slice of the entire thing for each page, granted it won't read the entire thing,
            // the thrift will terminate after reading the necessary things. @Rain the alternative is to feed it
            // chunks at a time in a loop until a valid header is returned, like before how we are using the reader in rust-test

            let (header_len, header) = read_page_header(&column_chunk_bytes, start)?;
            // println!("{} {} {:?}", start, header_len, header);

            let page_header = header.clone();

            let page: Page = match page_header.type_ {
                PageType::DICTIONARY_PAGE => {
                    dictionary_page_size = page_header.compressed_page_size as usize + header_len;
                    let page: Page = decode_page(
                        page_header,
                        column_chunk_bytes.slice(
                            (start as usize + header_len)
                                ..(start as usize + dictionary_page_size as usize),
                        ),
                        Type::BYTE_ARRAY,
                        codec.as_mut(),
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
                    parquet_layout
                        .data_page_offsets
                        .push((column_chunk_offset + start) as usize);

                    parquet_layout
                        .dictionary_page_sizes
                        .push(dictionary_page_size);
                    total_data_pages += 1;

                    let page = decode_page(
                        page_header,
                        column_chunk_bytes.slice(
                            (start as usize + header_len)
                                ..(start as usize + header_len + compressed_page_size as usize),
                        ),
                        Type::BYTE_ARRAY,
                        codec.as_mut(),
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
        metadata.row_group(0).schema_descr().column(column_index),
        None,
    )
    .unwrap();
    // let array = array_reader.next_batch(total_values as usize).unwrap();

    // instead of reading in total_values at once, we need to read 10_000 at a time and collect results into a Vec<Arc<dyn Array>>

    let mut arrays: Vec<ArrayData> = Vec::new();

    for _ in (0..total_values).step_by(10_000) {
        let array = array_reader.next_batch(10_000).unwrap();
        let new_array: Result<
            &arrow_array::GenericByteArray<arrow::datatypes::GenericStringType<i32>>,
            ArrowError,
        > = array.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
            ArrowError::ParseError("Expects string array as first argument".to_string())
        });

        let data = match new_array {
            Ok(_) => new_array.unwrap().to_data(),
            Err(_) => array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| {
                    ArrowError::ParseError(
                        "Expects string or binary array as first argument".to_string(),
                    )
                })
                .unwrap()
                .to_data(),
        };
        arrays.push(data);
    }

    Ok((arrays, parquet_layout))
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub file_path: String,
    pub column_index: usize,
    pub row_group: usize,
    pub offset_in_row_group: usize,
    pub matched: String,
}

pub async fn read_indexed_pages_async(
    column_name: String,
    file_paths: Vec<String>,
    row_groups: Vec<usize>,
    page_offsets: Vec<u64>,
    page_sizes: Vec<usize>,
    dict_page_sizes: Vec<usize>, // 0 means no dict page
    reader_type: ReaderType,
    file_metadatas: Option<HashMap<String, Bytes>>,
    in_order: Option<bool>,
) -> Result<Vec<ArrayData>, LavaError> {
    // current implementation might re-read dictionary pages, this should be optimized
    // we are assuming that all the files are either on disk or cloud.

    let codec_options = CodecOptionsBuilder::default()
        .set_backward_compatible_lz4(false)
        .build();

    let metadatas = match file_metadatas {
        Some(file_metadatas) => {
            println!("Using provided file metadatas");
            let mut metadatas: HashMap<String, ParquetMetaData> = HashMap::new();
            for (key, value) in file_metadatas.into_iter() {
                metadatas.insert(
                    key,
                    decode_metadata(value.to_byte_slice())
                        .map_err(LavaError::from)
                        .unwrap(),
                );
            }
            metadatas
        }
        None => parse_metadatas(&file_paths, reader_type.clone()).await,
    };

    let in_order: bool = in_order.unwrap_or(true);

    let mut reader = get_reader(file_paths[0].clone(), reader_type.clone())
        .await
        .unwrap();

    let iter = izip!(
        file_paths,
        row_groups,
        page_offsets,
        page_sizes,
        dict_page_sizes
    );

    let start = std::time::Instant::now();

    let mut future_handles: Vec<tokio::task::JoinHandle<ArrayData>> = vec![];
    let mut join_set = JoinSet::new();

    let iter: Vec<_> = stream::iter(iter)
        .map(
            |(file_path, row_group, page_offset, page_size, dict_page_size)| {
                let column_index = metadatas[&file_path]
                    .file_metadata()
                    .schema_descr()
                    .columns()
                    .iter()
                    .position(|column| column.name() == column_name)
                    .expect(&format!(
                        "column {} not found in parquet file {}",
                        column_name, file_path
                    ));
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

                let mut reader_c = reader.clone();
                reader_c.update_filename(file_path).unwrap();

                let future = async move {
                    let mut pages: Vec<parquet::column::page::Page> = Vec::new();
                    if dict_page_size > 0 {
                        let start = dict_page_offset.unwrap() as u64;
                        let dict_page_bytes = reader_c
                            .read_range(start, start + dict_page_size as u64)
                            .await
                            .unwrap();
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

                    let page_bytes = reader_c
                        .read_range(page_offset, page_offset + page_size as u64)
                        .await
                        .unwrap();
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

                    let new_array: Result<
                        &arrow_array::GenericByteArray<arrow::datatypes::GenericStringType<i32>>,
                        ArrowError,
                    > = array.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                        ArrowError::ParseError("Expects string array as first argument".to_string())
                    });

                    let data = match new_array {
                        Ok(_) => new_array.unwrap().to_data(),
                        Err(_) => array
                            .as_any()
                            .downcast_ref::<BinaryArray>()
                            .ok_or_else(|| {
                                ArrowError::ParseError(
                                    "Expects string or binary array as first argument".to_string(),
                                )
                            })
                            .unwrap()
                            .to_data(),
                    };

                    data
                };

                if in_order {
                    let handle = tokio::spawn(future);
                    future_handles.push(handle);
                } else {
                    join_set.spawn(future);
                }
            },
        )
        .collect::<Vec<_>>()
        .await;

    // it is absolutely crucial to collect results in the same order.

    let result: Vec<ArrayData> = if in_order {
        let res: Vec<std::prelude::v1::Result<ArrayData, tokio::task::JoinError>> =
            futures::future::join_all(future_handles).await;
        res.into_iter().map(|res| res.unwrap()).collect()
    } else {
        let mut result_inner: Vec<ArrayData> = vec![];
        while let Some(res) = join_set.join_next().await {
            result_inner.push(res.unwrap());
        }
        result_inner
    };

    join_set.shutdown().await;

    let end = std::time::Instant::now();
    println!("read_indexed_pages_async took {:?}", end - start);

    Ok(result)
}

pub fn read_indexed_pages(
    column_name: String,
    file_paths: Vec<String>,
    row_groups: Vec<usize>,
    page_offsets: Vec<u64>,
    page_sizes: Vec<usize>,
    dict_page_sizes: Vec<usize>, // 0 means no dict page
    reader_type: ReaderType,
    file_metadatas: Option<HashMap<String, Bytes>>,
    in_order: Option<bool>,
) -> Result<Vec<ArrayData>, LavaError> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let res = rt.block_on(read_indexed_pages_async(
        column_name,
        file_paths,
        row_groups,
        page_offsets,
        page_sizes,
        dict_page_sizes,
        reader_type,
        file_metadatas,
        in_order,
    ));
    rt.shutdown_background();
    res
}
