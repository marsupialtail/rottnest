pub mod error;

mod bm25;
#[cfg(feature = "logcloud")]
mod logcloud;
mod merge;
mod plist;
mod substring;
mod tokenizer_utils;
mod uuid;
mod vector;

pub(crate) use tokenizer_utils::get_tokenizer;
pub(crate) use tokenizer_utils::get_tokenizer_async;
pub use tokenizer_utils::get_tokenizer_vocab;

pub use bm25::build_lava_bm25;
pub use bm25::search_lava_bm25;

pub use substring::build_lava_substring;
pub use substring::build_lava_substring_char;
pub use substring::search_lava_substring;
pub use substring::search_lava_substring_char;

pub use merge::parallel_merge_files;

pub use vector::search_lava_vector;

pub use uuid::build_lava_uuid;
pub use uuid::search_lava_uuid;

#[cfg(feature = "logcloud")]
pub use logcloud::compress_logs;
#[cfg(feature = "logcloud")]
pub use logcloud::index_analysis;
#[cfg(feature = "logcloud")]
pub use logcloud::index_logcloud;
#[cfg(feature = "logcloud")]
pub use logcloud::search_logcloud;
