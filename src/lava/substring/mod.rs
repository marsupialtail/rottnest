mod constants;
pub(crate) mod fm_chunk;
pub(crate) mod merge;
mod substring;
pub(crate) mod wavelet_tree;
pub(crate) use merge::merge_lava_substring;

pub(crate) use substring::_build_lava_substring_char;
pub(crate) use substring::_build_lava_substring_char_wavelet;
pub use substring::build_lava_substring;
pub use substring::build_lava_substring_char;
