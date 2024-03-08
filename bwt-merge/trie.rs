use std::cmp::{max, min};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct BinaryTrieNode {
    pub left: Option<Box<BinaryTrieNode>>,
    pub right: Option<Box<BinaryTrieNode>>,
    pub end: bool,
    pub data: Option<Vec<usize>>,
}

impl BinaryTrieNode {
    pub fn new() -> BinaryTrieNode {
        BinaryTrieNode {
            left: None,
            right: None,
            end: false,
            data: None,
        }
    }
}

// note: it's difficult to support insert on this trie,
// since we stop storing anything past the LCP
// so we can't differentiate between two strings that share a prefix after build

pub fn build_binary_trie(strs: &Vec<Vec<u8>>, inds: &Vec<Vec<usize>>) -> BinaryTrieNode {
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
        let first_diff_char = min(max(lcp[i], lcp[i + 1]) + 1, strs[i].len() * 8);
        let mut node = &mut root;
        for j in 0..first_diff_char {
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
        if node.data.is_none() {
            node.data = Some(Vec::new());
        }
        node.data.as_mut().unwrap().extend(&inds[i]);
        node.end = true;
    }

    root
}

// Query the trie for matching indices
// Note that if string does not exist it may return results that don't match,
// but at most one, so you can check manually
pub fn query_string(root: &BinaryTrieNode, query: &Vec<u8>) -> Vec<usize> {
    let get_bit = |i: usize| -> bool {
        let chr = i / 8;
        let bit = 7 - (i % 8);
        (query[chr] >> bit) & 1 == 1
    };

    let mut node = root;
    for i in 0..query.len() * 8 {
        if !get_bit(i) {
            if node.left.is_none() {
                if node.end {
                    return node.data.as_ref().unwrap().clone();
                }
                return Vec::new();
            }
            node = node.left.as_ref().unwrap();
        } else {
            if node.right.is_none() {
                if node.end {
                    return node.data.as_ref().unwrap().clone();
                }
                return Vec::new();
            }
            node = node.right.as_ref().unwrap();
        }
    }
    if node.end {
        node.data.as_ref().unwrap().clone()
    } else {
        Vec::new()
    }
}
