use crate::{formats::readers::get_file_size_and_reader, lava::error::LavaError};

use super::trie::{BinaryTrieNode, FastTrie};
use arrow::array::{make_array, Array, ArrayData, LargeStringArray, UInt64Array};

#[tokio::main]
pub async fn build_lava_uuid(
    output_file_name: String,
    array: ArrayData,
    uid: ArrayData,
) -> Result<Vec<(usize, usize)>, LavaError> {
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

    let mut texts = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        let text = array.value(i);
        texts.push(text.as_bytes().to_vec());
    }
    let mut inds = Vec::with_capacity(array.len());
    for i in 0..uid.len() {
        inds.push(vec![uid.value(i) as usize]);
    }

    let root = BinaryTrieNode::build(&texts, &inds);
    let fast_trie = FastTrie::new(root, Some(16));
    let (serialized_fast_trie, (cache_start, cache_end)) = fast_trie.serialize();
    std::fs::write(output_file_name, serialized_fast_trie).unwrap();

    Ok(vec![(cache_start, cache_end)])
}

pub(crate) async fn merge_lava_uuid(
    condensed_lava_file: &str,
    lava_files: Vec<String>,
    uid_offsets: Vec<u64>,
    reader_type: ReaderType,
) -> Result<Vec<(usize, usize)>, LavaError> {
    // currently only support merging two files, but can support more in the future.
    assert_eq!(lava_files.len(), 2);
    assert_eq!(uid_offsets.len(), 2);

    let (file_size1, mut reader1) =
        get_file_size_and_reader(lava_files[0].clone(), reader_type.clone()).await?;
    let (file_size2, mut reader2) =
        get_file_size_and_reader(lava_files[1].clone(), reader_type.clone()).await?;

    // let buffer: bytes::Bytes = reader1.read_range(0, file_size1 as u64).await?;
    // let mut fast_trie1 = FastTrie::deserialize(buffer.to_vec());
    // let buffer: bytes::Bytes = reader2.read_range(0, file_size2 as u64).await?;
    // let mut fast_trie2 = FastTrie::deserialize(buffer.to_vec());

    // fast_trie1.extend(
    //     &mut fast_trie2,
    //     uid_offsets[0] as usize,
    //     uid_offsets[1] as usize,
    // );
    // let (serialized, (cache_start, cache_end)) = fast_trie1.serialize();
    // let mut output_file = File::create(condensed_lava_file)?;
    // output_file.write(&serialized)?;

    let (cache_start, cache_end) = FastTrie::extend_with_readers_into_file(
        file_size1,
        &mut reader1,
        file_size2,
        &mut reader2,
        condensed_lava_file,
        uid_offsets[0] as usize,
        uid_offsets[1] as usize,
    )
    .await?;

    Ok(vec![(cache_start, cache_end)])
}
