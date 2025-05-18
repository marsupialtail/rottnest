mod build;
mod constants;
pub(crate) mod fm_chunk;
pub(crate) mod merge;
mod search;
pub(crate) mod wavelet_tree;
pub(crate) use merge::merge_lava_substring;
pub(crate) use merge::merge_lava_substring_char;

pub(crate) use build::_build_lava_substring_char;
pub(crate) use build::_build_lava_substring_char_wavelet;
pub use build::build_lava_substring;
pub use build::build_lava_substring_char;
pub(crate) use search::_search_lava_substring_char;
pub use search::search_lava_substring;
pub use search::search_lava_substring_char;
