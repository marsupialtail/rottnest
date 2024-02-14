use arrow::array::{make_array, Array, ArrayData, StringArray, UInt64Array};

use tantivy::tokenizer::*;
use tantivy_jieba::JiebaTokenizer;

use bincode;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use zstd::stream::encode_all;

use whatlang::{detect, Lang};

use lazy_static::lazy_static;

use crate::lava::error::LavaError;
use crate::lava::plist::PList;

#[derive(Clone)]
enum TokenizerEnum {
    Simple(SimpleTokenizer),
    Jieba(JiebaTokenizer),
    English(TextAnalyzer),
}

lazy_static! {
    static ref DEFAULT_TOKENIZER: TokenizerEnum = TokenizerEnum::Simple(SimpleTokenizer::default());
    static ref TOKENIZERS: HashMap<Lang, TokenizerEnum> = {
        let mut tokenizers = HashMap::new();
        tokenizers.insert(Lang::Cmn, TokenizerEnum::Jieba(JiebaTokenizer {}));

        let en_stem = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(Stemmer::new(Language::English))
            .build();
        tokenizers.insert(Lang::Eng, TokenizerEnum::English(en_stem));

        tokenizers
    };
}

impl TokenizerEnum {
    fn token_stream<'a>(&'a mut self, text: &'a str) -> tantivy::tokenizer::BoxTokenStream<'a> {
        match self {
            TokenizerEnum::Simple(tokenizer) => BoxTokenStream::new(tokenizer.token_stream(text)),
            TokenizerEnum::Jieba(tokenizer) => BoxTokenStream::new(tokenizer.token_stream(text)),
            TokenizerEnum::English(tokenizer) => tokenizer.token_stream(text),
        }
    }
}

/*
Structure of the lava file
It is important to put the posting lists first. Just trust me bro.
| compressed posting lists line by line | compressed term dictionary | compressed posting list offsets|
8 bytes = offsets of compressed term dict | 8 bytes = offset of compressed posting list offsets
*/

/// Function that tokenizes the input text and returns a list of tokens.

pub fn build_lava_natural_language(
    output_file_name: Cow<str>,
    array: ArrayData,
    uid: ArrayData,
    language: Option<ArrayData>,
) -> Result<(), LavaError> {
    let array = make_array(array);
    // let uid = make_array(ArrayData::from_pyarrow(uid)?);
    let uid = make_array(uid);

    let array: &arrow_array::GenericByteArray<arrow_array::types::GenericStringType<i32>> = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(LavaError::Parse(
            "Expects string array as first argument".to_string(),
        ))?;

    let uid = uid
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(LavaError::Parse(
            "Expects uint64 array as second argument".to_string(),
        ))?;

    let mut unique_uids: HashSet<u64> = HashSet::new();
    for i in 0..uid.len() {
        unique_uids.insert(uid.value(i));
    }
    let num_unique_uids = unique_uids.len() as u64;

    if array.len() != uid.len() {
        return Err(LavaError::Parse(
            "The length of the array and the uid array must be the same".to_string(),
        ));
    }

    let language = match language {
        Some(x) => {
            let array = make_array(x);

            let test = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(LavaError::Parse(
                    "Expects string array as optional third argument".to_string(),
                ))?;
            Some(test.clone())
        }
        None => None,
    };

    // let mut tokens: Vec<Vec<String>> = Vec::new();
    let mut inverted_index: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();

    for i in 0..array.len() {
        let text = array.value(i);
        // let lang = if let Some(ref language) = language {
        //     Lang::from_code(language.value(i))
        // } else {
        //     detect(text).map(|info| info.lang())
        // }
        // .unwrap_or(Lang::Eng);

        // let mut tokenizer = TOKENIZERS.get(&lang).unwrap_or(&DEFAULT_TOKENIZER).clone();

        let mut tokenizer = DEFAULT_TOKENIZER.clone();
        // println!("text: {} {}", text, detect(text).unwrap_or(Info::new(Script::Latin, Lang::Eng, 0.0)).lang());

        // The following code can be optimized as multiple threads https://docs.rs/futures/0.3.30/futures/executor/struct.ThreadPool.html
        let mut token_stream = tokenizer.token_stream(text);
        // let mut this_tokens = Vec::new();
        while let Some(token) = token_stream.next() {
            // this_tokens.push(token.text.to_string());
            inverted_index
                .entry(format!("{}\n", token.text))
                .or_insert_with(BTreeSet::new)
                .insert(uid.value(i));
        }
        // tokens.push(this_tokens);
    }

    let total_length: usize = inverted_index.keys().map(|k| k.len()).sum();
    let mut term_dictionary = String::with_capacity(total_length);
    for key in inverted_index.keys() {
        term_dictionary.push_str(key);
    }

    let mut file = File::create(output_file_name.as_ref())?;

    let bytes = term_dictionary.as_bytes();
    let compressed_term_dictionary = encode_all(bytes, 0).expect("Compression failed");

    // Handle the compressed data (for example, saving to a file or sending over a network)
    println!(
        "Compressed term dictionary length: {}",
        compressed_term_dictionary.len()
    );

    let mut plist_offsets: Vec<u64> = vec![0];
    let mut plist_elems: Vec<u64> = vec![0];
    let mut plist = PList::new()?;
    let mut counter: u64 = 0;

    for (_, value) in inverted_index.iter() {
        // this usually saves around 20% of the space. Don't remember things that happen more than 1/4 of the time.
        
        let mut value_all = BTreeSet::new();
        value_all.insert(u64::MAX);

        let value_vec = if value.len() <= (num_unique_uids / 4) as usize {
            //@Rain can we get rid of this clone
            value
        } else {
            &value_all
        };

        // let value_vec = value;

        counter += 1;

        // value_vec.sort();
        // println!("{}", key);
        let written = plist.add_plist(value_vec)?;
        if written > 1024 * 1024 || counter == inverted_index.len() as u64 {
            let bytes = plist.finalize_compression()?;
            file.write_all(&bytes)?;
            plist_offsets.push(plist_offsets[plist_offsets.len() - 1] + bytes.len() as u64);
            plist_elems.push(counter);
            plist = PList::new()?;
        }
    }


    plist_offsets.append(&mut plist_elems);

    let compressed_term_dict_offset = file.seek(SeekFrom::Current(0))?;
    file.write_all(&compressed_term_dictionary)?;

    let compressed_plist_offsets_offset = file.seek(SeekFrom::Current(0))?;
    let serialized = bincode::serialize(&plist_offsets).unwrap();
    let compressed_plist_offsets =
        encode_all(&serialized[..], 0).expect("Compression of plist offsets failed");
    file.write_all(&compressed_plist_offsets)?;

    file.write_all(&(compressed_term_dict_offset as u64).to_le_bytes())?;
    file.write_all(&(compressed_plist_offsets_offset as u64).to_le_bytes())?;

    Ok(())
}
