use crate::{
    formats::readers::{
        get_file_size_and_reader, get_file_sizes_and_readers, get_reader, get_readers, AsyncReader,
        ReaderType,
    },
    lava::error::LavaError,
};

use std::io::Read;
use tokenizers::tokenizer::Tokenizer;
use zstd::stream::encode_all;
use zstd::stream::read::Decoder;

pub(crate) fn get_tokenizer(
    tokenizer_file: Option<String>,
) -> Result<(Tokenizer, Vec<u8>), LavaError> {
    // if the tokenizer file is provided, check if the file exists. If it does not exist, raise an Error
    let tokenizer = if let Some(tokenizer_file) = tokenizer_file {
        if !std::path::Path::new(&tokenizer_file).exists() {
            return Err(LavaError::Parse(
                "Tokenizer file does not exist".to_string(),
            ));
        }
        println!("Tokenizer file: {}", tokenizer_file);
        Tokenizer::from_file(tokenizer_file).unwrap()
    } else {
        Tokenizer::from_pretrained("bert-base-uncased", None).unwrap()
    };

    let serialized_tokenizer = serde_json::to_string(&tokenizer).unwrap();
    let compressed_tokenizer =
        encode_all(serialized_tokenizer.as_bytes(), 0).expect("Compression failed");
    Ok((tokenizer, compressed_tokenizer))
}

pub(crate) async fn get_tokenizer_async(
    mut readers: Vec<AsyncReader>,
) -> Result<(Tokenizer, Vec<String>), LavaError> {
    let mut compressed_tokenizer: Option<Vec<u8>> = None;

    for i in 0..readers.len() {
        // now interpret this as a usize
        // readers[i].seek(SeekFrom::Start(0)).await?;
        let compressed_tokenizer_size = readers[i].read_usize_from_start(0, 1).await?[0];
        let this_compressed_tokenizer: bytes::Bytes = readers[i]
            .read_range(8, 8 + compressed_tokenizer_size)
            .await?;
        match &compressed_tokenizer {
            Some(value) => assert!(
                this_compressed_tokenizer == value,
                "detected different tokenizers between different lava files, can't search across them."
            ),
            None => compressed_tokenizer = Some(this_compressed_tokenizer.to_vec()),
        }
    }

    let slice = &compressed_tokenizer.unwrap()[..];
    let mut decompressor = Decoder::new(slice)?;
    let mut decompressed_serialized_tokenizer: Vec<u8> = Vec::with_capacity(slice.len() as usize);
    decompressor.read_to_end(&mut decompressed_serialized_tokenizer)?;

    let mut result: Vec<String> = Vec::new();
    let tokenizer = Tokenizer::from_bytes(decompressed_serialized_tokenizer).unwrap();

    for i in 0..tokenizer.get_vocab_size(false) {
        let tok = tokenizer.decode(&vec![i as u32], false).unwrap();
        result.push(tok);
    }

    Ok((tokenizer, result))
}

#[tokio::main]
pub async fn get_tokenizer_vocab(
    files: Vec<String>,
    reader_type: ReaderType,
) -> Result<Vec<String>, LavaError> {
    let (_file_sizes, readers) = get_file_sizes_and_readers(&files, reader_type).await?;
    Ok(get_tokenizer_async(readers).await?.1)
}
