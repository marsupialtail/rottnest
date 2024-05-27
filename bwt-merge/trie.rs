use std::{
    cmp::{max, min},
    num::ParseIntError,
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct BinaryTrieNode<T: Clone> {
    pub left: Option<Box<BinaryTrieNode<T>>>,
    pub right: Option<Box<BinaryTrieNode<T>>>,
    pub data: Vec<T>,
}

impl<T: Clone> BinaryTrieNode<T> {
    pub fn new() -> BinaryTrieNode<T> {
        BinaryTrieNode {
            left: None,
            right: None,
            data: Vec::new(),
        }
    }

    /// Builds a binary trie from a list of strings and their corresponding indices.
    /// String list should be sorted.
    /// Default extra bits is 8.
    pub fn build(strs: &[Vec<u8>], str_data: &[Vec<T>]) -> BinaryTrieNode<T> {
        BinaryTrieNode::build_extra(strs, str_data, 8)
    }

    /// Build, specifying extra bits
    pub fn build_extra(
        strs: &[Vec<u8>],
        str_data: &[Vec<T>],
        extra_bits: usize,
    ) -> BinaryTrieNode<T> {
        // big endian
        let get_bit = |stri: usize, i: usize| -> bool {
            let chr = i / 8;
            let bit = 7 - (i % 8);
            (strs[stri][chr] >> bit) & 1 == 1
        };

        // calculate LCPs
        let mut lcp = vec![0; strs.len() + 1];
        // lcp[i] contains lcp of strs[i-1] and strs[i]
        // lcp[0] := lcp[n] := 0
        for i in 0..strs.len() - 1 {
            let mut j = 0;
            while j < strs[i].len() * 8
                && j < strs[i + 1].len() * 8
                && get_bit(i, j) == get_bit(i + 1, j)
            {
                j += 1;
            }
            lcp[i + 1] = j;
        }

        // build trie
        let mut root = BinaryTrieNode::new();
        for i in 0..strs.len() {
            let node_depth = min(max(lcp[i], lcp[i + 1]) + 1 + extra_bits, strs[i].len() * 8);
            let mut node = &mut root;
            for j in 0..node_depth {
                if !get_bit(i, j) {
                    if node.left.is_none() {
                        node.left = Some(Box::new(BinaryTrieNode::new()));
                    }
                    node = node.left.as_mut().unwrap();
                } else {
                    if node.right.is_none() {
                        node.right = Some(Box::new(BinaryTrieNode::new()));
                    }
                    node = node.right.as_mut().unwrap();
                }
            }
            node.data.extend(str_data[i].clone());
        }

        root
    }

    // Merge trie t2 into trie t1, consuming t2 (faster than merge_tries since no cloning)
    pub fn extend(&mut self, t2: BinaryTrieNode<T>) {
        self.data.extend(t2.data);
        if let Some(self_left) = self.left.as_mut() {
            if let Some(t2_left) = t2.left {
                self_left.extend(*t2_left);
            }
        } else {
            self.left = t2.left;
        }

        if let Some(self_right) = self.right.as_mut() {
            if let Some(t2_right) = t2.right {
                self_right.extend(*t2_right);
            }
        } else {
            self.right = t2.right;
        }
    }

    // Query the trie for matching indices
    // Note that if string does not exist it may return results that don't match,
    // but only a few, so you can check manually
    // Collects all results seen on the way, in order to support merging
    pub fn query(&self, query: &[u8]) -> Vec<T> {
        let get_bit = |i: usize| -> bool {
            let chr = i / 8;
            let bit = 7 - (i % 8);
            (query[chr] >> bit) & 1 == 1
        };

        let mut node = self;
        let mut results = Vec::new();
        for i in 0..query.len() * 8 {
            if !get_bit(i) {
                if node.left.is_none() {
                    break;
                }
                results.extend(node.data.clone());
                node = node.left.as_ref().unwrap();
            } else {
                if node.right.is_none() {
                    break;
                }
                results.extend(node.data.clone());
                node = node.right.as_ref().unwrap();
            }
        }

        results.extend(node.data.clone());
        results
    }
}

impl<T: Clone> Default for BinaryTrieNode<T> {
    fn default() -> Self {
        Self::new()
    }
}

// note: it's difficult to support insert on this trie,
// since we stop storing anything past the LCP
// so we can't differentiate between two strings that share a prefix after build

/// Merges two tries into a new trie.
/// Indices are kept as-is.
pub fn merge_tries<T: Clone>(t1: &BinaryTrieNode<T>, t2: &BinaryTrieNode<T>) -> BinaryTrieNode<T> {
    let mut output = BinaryTrieNode::new();
    output.data.extend(t1.data.clone());
    output.data.extend(t2.data.clone());

    if t1.left.is_none() {
        output.left = t2.left.clone();
    } else if t2.left.is_none() {
        output.left = t1.left.clone();
    } else {
        output.left = Some(Box::new(merge_tries(
            t1.left.as_ref().unwrap(),
            t2.left.as_ref().unwrap(),
        )));
    }

    if t1.right.is_none() {
        output.right = t2.right.clone();
    } else if t2.right.is_none() {
        output.right = t1.right.clone();
    } else {
        output.right = Some(Box::new(merge_tries(
            t1.right.as_ref().unwrap(),
            t2.right.as_ref().unwrap(),
        )));
    }

    output
}

pub fn hex_to_u8(hex: &str) -> Result<Vec<u8>, ParseIntError> {
    let mut bytes = Vec::new();
    for i in 0..hex.len() / 2 {
        let byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)?;
        bytes.push(byte);
    }
    if hex.len() % 2 == 1 {
        let byte = u8::from_str_radix(&hex[hex.len() - 1..hex.len()], 16)? * 16;
        bytes.push(byte);
    }
    Ok(bytes)
}

pub fn compress_hex_strs(strs: &[&str]) -> Result<Vec<Vec<u8>>, ParseIntError> {
    strs.iter().map(|s| hex_to_u8(s)).collect()
}
