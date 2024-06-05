use std::{
    cmp::{max, min}, collections::{BTreeMap, BTreeSet}, num::ParseIntError, ops::AddAssign
};

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use bitvec::prelude::*;
use crate::{
    formats::readers::{get_file_size_and_reader, get_file_sizes_and_readers, AsyncReader, ClonableAsyncReader, ReaderType},
    lava::error::LavaError,
};
use zstd::stream::{encode_all, read::Decoder};
use std::io::Read;

#[derive(Serialize, Deserialize, Clone)]
pub struct BinaryTrieNode<T: Clone + AddAssign> {
    pub left: Option<Box<BinaryTrieNode<T>>>,
    pub right: Option<Box<BinaryTrieNode<T>>>,
    pub data: Vec<T>,
}

pub struct FastTrie{
    pub root_lut: BTreeMap<BitVec, (Vec<usize>, Option<usize>)>,
    pub leaf_tree_roots: Vec<Box<BinaryTrieNode<usize>>>,
    pub root_levels: usize
}

// Helper function to take a node and replace it with None
fn take_leaf_node(node: &mut Box<BinaryTrieNode<usize>>) -> Box<BinaryTrieNode<usize>> {
    std::mem::replace(node, Box::new(BinaryTrieNode { data: Vec::new(), left: None, right: None }))
}


impl FastTrie {
    pub fn new(node: BinaryTrieNode<usize>, root_levels: Option<usize>) -> Self {
        let root_levels = root_levels.unwrap_or(8);

        // currently due to dumb way we are handling the query_with_reader, this is a requirement.
        assert_eq!(root_levels % 8, 0);

        let mut root_lut: BTreeMap<BitVec, (Vec<usize>, Option<usize>)> = BTreeMap::new();

        // walk the tree up to levels
        let mut leaf_tree_roots= Vec::new();
        let mut queue: VecDeque<(&mut Box<BinaryTrieNode<usize>>, usize, BitVec)> = VecDeque::new();
        let mut current = Box::new(node);
        queue.push_back((&mut current, 0, BitVec::new()));

        while let Some((current, level, path)) = queue.pop_front() {

            if level == root_levels { // Check if next level is within the limit
                let idx = leaf_tree_roots.len();
                root_lut.insert(path.clone(), (current.data.clone(), Some(idx)));
                leaf_tree_roots.push(take_leaf_node(current));
                
            } else {
                root_lut.insert(path.clone(), (current.data.clone(), None));
                if ! current.left.is_none() {
                    let mut new_path = path.clone();
                    new_path.push(false);
                    queue.push_back((current.left.as_mut().unwrap(), level + 1, new_path));
                }
                if ! current.right.is_none() {
                    let mut new_path = path.clone();
                    new_path.push(true);
                    queue.push_back((current.right.as_mut().unwrap(), level + 1, new_path));
                }
            }
            
        }

        // print the length of all the bitvectors

        for (k, v) in &root_lut {
            println!("{} {:?} {:?}", k.len(), k, v.1);
        }

        FastTrie {
            root_lut,
            leaf_tree_roots,
            root_levels
        }
    }

    // structure is serialized trie | serialized trie | ... | serialized (lut, offsets) | metadata page offset

    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut offsets: Vec<usize> = vec![0];

        // now go serialize all the roots

        for node in &self.leaf_tree_roots {
            let serialized_node = bincode::serialize(&node).unwrap();
            bytes.extend(encode_all(&serialized_node[..], 10).unwrap());
            offsets.push(bytes.len());
        }

        let metadata_page_offset = offsets[offsets.len() - 1];

        let metadata: (&BTreeMap<BitVec, (Vec<usize>, Option<usize>)>, &Vec<usize>, usize) = (&self.root_lut, &offsets, self.root_levels);

        let serialized_metadata = bincode::serialize(&metadata).unwrap();
        let compressed = encode_all(&serialized_metadata[..], 10).unwrap();

        let rehydrate_metadata: (BTreeMap<BitVec, (Vec<usize>, Option<usize>)>, Vec<usize>, usize) = bincode::deserialize(&serialized_metadata[..]).unwrap();
        println!("{} {}", rehydrate_metadata.0.len(), rehydrate_metadata.1.len());

        bytes.extend(compressed);
        bytes.extend(&(metadata_page_offset as u64).to_le_bytes());

        bytes
    }

    pub fn deserialize(bytes:Vec<u8>) -> Self {
        let metadata_page_offset = u64::from_le_bytes(bytes[(bytes.len() - 8)..].try_into().unwrap());
        
        let metadata_page_bytes = &bytes[(metadata_page_offset as usize) .. bytes.len() -8];
        let mut decompressor = Decoder::new(&metadata_page_bytes[..]).unwrap();
        let mut serialized_metadata: Vec<u8> = Vec::with_capacity(metadata_page_bytes.len() as usize);
        decompressor.read_to_end(&mut serialized_metadata).unwrap();

        let metadata: (BTreeMap<BitVec, (Vec<usize>, Option<usize>)>, Vec<usize>, usize) = bincode::deserialize(&serialized_metadata[..]).unwrap();

        let lut: BTreeMap<BitVec, (Vec<usize>, Option<usize>)> = metadata.0;
        let offsets: Vec<usize> = metadata.1;

        let mut leaf_tree_roots = Vec::new();

        for i in 0..offsets.len() - 1 {
            let start = offsets[i];
            let end = offsets[i + 1];
            let compressed_node = &bytes[start..end];
            let mut decompressor = Decoder::new(&compressed_node[..]).unwrap();
            let mut serialized_node: Vec<u8> = Vec::with_capacity(compressed_node.len() as usize);
            decompressor.read_to_end(&mut serialized_node).unwrap();
            let node: Box<BinaryTrieNode<usize>> = bincode::deserialize(&serialized_node).unwrap();
            leaf_tree_roots.push(node);
        }

        FastTrie {
            root_lut: lut,
            leaf_tree_roots: leaf_tree_roots,
            root_levels: metadata.2
        }
    }

    pub async fn query_with_reader(file_size: usize, reader: &mut AsyncReader, query: &str) -> Result<Vec<usize>, LavaError> {

        let query = query.as_bytes();

        let get_bit = |i: usize| -> bool {
            let chr = i / 8;
            let bit = 7 - (i % 8);
            (query[chr] >> bit) & 1 == 1
        };

        let metadata_page_offset = reader.read_usize_from_end(1).await?[0];
        let metadata_page_bytes = reader.read_range(metadata_page_offset, file_size as u64 - 8).await?;
        let mut decompressor = Decoder::new(&metadata_page_bytes[..]).unwrap();
        let mut serialized_metadata: Vec<u8> = Vec::with_capacity(metadata_page_bytes.len() as usize);
        decompressor.read_to_end(&mut serialized_metadata).unwrap();

        let metadata: (BTreeMap<BitVec, (Vec<usize>, Option<usize>)>, Vec<usize>, usize) = bincode::deserialize(&serialized_metadata[..]).unwrap();
        let lut: BTreeMap<BitVec, (Vec<usize>, Option<usize>)> = metadata.0;
        let offsets: Vec<usize> = metadata.1;
        let root_levels = metadata.2;

        // take a short cut here and assume you have more bits than 
        let mut bitvec: BitVec<usize, Lsb0> = BitVec::new();
        for i in 0..root_levels {
            bitvec.push(get_bit(i));
        }

        match lut.get(&bitvec) { 
            Some((values, offset)) => {

                let offset = offset.as_ref().unwrap();
                let start = offsets[*offset];
                let end = offsets[*offset + 1];
                let compressed_trie_bytes = reader.read_range(start as u64, end as u64).await?;
                
                let mut decompressor = Decoder::new(&compressed_trie_bytes[..]).unwrap();
                let mut serialized_trie: Vec<u8> = Vec::with_capacity(compressed_trie_bytes.len() as usize);
                decompressor.read_to_end(&mut serialized_trie).unwrap();
                let trie: Box<BinaryTrieNode<usize>> = bincode::deserialize(&serialized_trie[..]).unwrap();
                let result = trie.query(&query[root_levels / 8 ..]);
                return Ok(result);

            }, 
            None => {return Ok(Vec::new());}
        }; 

    }

    // extend and consume the second FastTrie
    pub fn extend(&mut self, t2: &mut FastTrie, uid_offset_0: usize, uid_offset_1: usize) {

        assert_eq!(self.root_levels, t2.root_levels);

        // first increment all the values by uid_offsets in the lut as well as the leaf_tree_roots
        for (_, v) in self.root_lut.iter_mut() {
            v.0.iter_mut().for_each(|x| *x += uid_offset_0);
        }
        for (_, v) in t2.root_lut.iter_mut() {
            v.0.iter_mut().for_each(|x| *x += uid_offset_1);
        }

        for v in self.leaf_tree_roots.iter_mut() {
            v.increment_values(uid_offset_0);
        }
        for v in t2.leaf_tree_roots.iter_mut() {
            v.increment_values(uid_offset_1);
        }

        // first insert keys in t2.root_lut into self.root_lut if key doesn't exist in self.root_lut
        
        let mut t2_to_remove: BTreeSet<BitVec> = BTreeSet::new();
        for (k, v) in t2.root_lut.iter() {
            if !self.root_lut.contains_key(k) {

                let (data, offset) = v;
                let new_offset = match offset {
                    Some(x) => {
                        let idx = self.leaf_tree_roots.len();
                        self.leaf_tree_roots.push(take_leaf_node(&mut t2.leaf_tree_roots[*x]));
                        Some(idx)
                    }
                    None => {
                        None
                    }
                };
                self.root_lut.insert(k.clone(), (data.clone(), new_offset));
                t2_to_remove.insert(k.clone());
            }
        }

        // now remove all the things in t2_to_remove
        for k in t2_to_remove.iter() {
            t2.root_lut.remove(k);
        }

        // now merge existing keys
        for (k, v) in self.root_lut.iter_mut() {
            match t2.root_lut.remove_entry(k) {
                Some((_, v2)) => {
                    v.0.extend(v2.0);
                    v.1 = match (v.1, v2.1) {
                        (Some(x), Some(y)) => {
                            println!("merging leaf tree roots {:?} {:?}", x, y);
                            let owned_t2_leaf_tree_root = take_leaf_node(&mut t2.leaf_tree_roots[y]);
                            self.leaf_tree_roots[x].extend(*owned_t2_leaf_tree_root);
                            Some(x)
                        },
                        (Some(x), None) => {Some(x)},
                        (None, Some(y)) => {
                            let idx = self.leaf_tree_roots.len();
                            self.leaf_tree_roots.push(take_leaf_node(&mut t2.leaf_tree_roots[y]));
                            Some(idx)
                        },
                        (None, None) => {None}
                    }
                },
                None => {println!("{:?} not found in t2", k);},
            }
        }

        // assert!(t2.leaf_tree_roots.is_empty());
        // assert!(t2.root_lut.is_empty());
        
    }

}


impl<T: Clone + AddAssign> BinaryTrieNode<T> {
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
            // println!("node depth: {}", node_depth);
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

    pub fn increment_values(&mut self, offset: T) {
        self.data.iter_mut().for_each(|x| *x += offset.clone());
        if let Some(self_left) = self.left.as_mut() {
            self_left.increment_values(offset.clone());
        }
        if let Some(self_right) = self.right.as_mut() {
            self_right.increment_values(offset.clone());
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
            // println!("{}", i);
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

impl<T: Clone + AddAssign> Default for BinaryTrieNode<T> {
    fn default() -> Self {
        Self::new()
    }
}

// note: it's difficult to support insert on this trie,
// since we stop storing anything past the LCP
// so we can't differentiate between two strings that share a prefix after build

/// Merges two tries into a new trie.
/// Indices are kept as-is.
pub fn merge_tries<T: Clone + AddAssign>(t1: &BinaryTrieNode<T>, t2: &BinaryTrieNode<T>) -> BinaryTrieNode<T> {
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
