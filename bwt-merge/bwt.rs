use std::collections::BTreeSet;

use bit_vec::BitVec;
use libdivsufsort_rs::divsufsort64;

type BWT = Vec<u8>;
// bwt, line index, character counts
type BWTData = (BWT, Vec<usize>, [usize; 256]);

// Compute the BWT of a string, using the divsufsort crate.
// Returns the BWT and the line index
pub fn run_bwt(input: &Vec<u8>) -> BWTData {
    // find newline indices
    let newlines = input
        .iter()
        .enumerate()
        .filter(|(_, &x)| x == b'\n')
        .map(|(i, _)| i)
        .collect::<Vec<usize>>();

    let sa = divsufsort64(&input).unwrap();

    let mut bwt = Vec::with_capacity(input.len());
    let mut line_index = Vec::with_capacity(input.len());
    let mut counts: [usize; 256] = [0; 256];
    for i in 0..sa.len() {
        if sa[i] == 0 {
            bwt.push(input[input.len() - 1]);
        } else {
            bwt.push(input[(sa[i] - 1) as usize]);
        }

        // binary search for newline index
        let res = newlines.binary_search_by(|nl| nl.cmp(&(sa[i] as usize)));
        match res {
            Ok(ind) => line_index.push(ind),
            Err(ind) => line_index.push(ind),
        }

        counts[bwt[i] as usize] += 1;
    }

    return (bwt, line_index, counts);
}

// Compute the interleave of two BWTs.
fn compute_interleave(bwt0: &BWT, bwt1: &BWT, counts: &[usize; 256]) -> BitVec {
    // construct character starts array
    let mut starts: [usize; 256] = [0; 256];
    let mut sum = 0;
    for i in 0..256 {
        starts[i] = sum;
        sum += counts[i];
    }

    let mut interleave = BitVec::from_elem(bwt0.len() + bwt1.len(), true);
    for i in 0..bwt0.len() {
        interleave.set(i, false);
    }

    loop {
        let mut ind: [usize; 2] = [0, 0];

        let mut offsets = starts.clone();
        let mut new_interleave = BitVec::from_elem(interleave.len(), false);
        for i in 0..interleave.len() {
            if interleave[i] {
                new_interleave.set(offsets[bwt1[ind[1]] as usize], true);
                offsets[bwt1[ind[1]] as usize] += 1;
                ind[1] += 1;
            } else {
                offsets[bwt0[ind[0]] as usize] += 1;
                ind[0] += 1;
            }
        }

        if new_interleave == interleave {
            break;
        }
        interleave = new_interleave;
    }
    interleave
}

// Merge two BWTs using our algorithm.
pub fn bwt_merge(bwt0_d: &BWTData, bwt1_d: &BWTData) -> BWTData {
    let (bwt0, line_ind0, counts0) = bwt0_d;
    let (bwt1, line_ind1, counts1) = bwt1_d;

    // construct character counts array
    let mut counts: [usize; 256] = [0; 256];
    for i in 0..256 {
        counts[i] = counts0[i] + counts1[i];
    }

    let interleave = compute_interleave(bwt0, bwt1, &counts);

    // construct bwt
    let mut bwt = Vec::with_capacity(interleave.len());
    let mut line_index = Vec::with_capacity(interleave.len());
    let mut ind0 = 0;
    let mut ind1 = 0;

    // assumes the number of lines in bwt0 is the number of newlines
    let num_newlines = counts0[b'\n' as usize];
    
    for i in 0..interleave.len() {
        if interleave[i] {
            bwt.push(bwt1[ind1]);
            line_index.push(line_ind1[ind1] + num_newlines);
            ind1 += 1;
        } else {
            bwt.push(bwt0[ind0]);
            line_index.push(line_ind0[ind0]);
            ind0 += 1;
        }
    }
    return (bwt, line_index, counts);
}

const BLOCK_SIZE: usize = 1024;
pub struct FMBlock {
    bwt_slice: Vec<u8>,
    c_arr: [usize; 256],
    offsets: [usize; 256],
}

// Compute the FM-index of a BWT.
pub fn fm_index(data: &BWTData) -> Vec<FMBlock> {
    let (bwt, _, all_counts) = data;
    let num_blocks = (bwt.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut blocks: Vec<FMBlock> = Vec::with_capacity(num_blocks);

    // calculate C array
    let mut c_arr: [usize; 256] = [0; 256];
    let mut sum = 0;
    for i in 0..256 {
        c_arr[i] = sum;
        sum += all_counts[i];
    }

    let mut counts: [usize; 256] = [0; 256];
    for i in 0..num_blocks {
        let start = i * BLOCK_SIZE;
        let end = (i + 1) * BLOCK_SIZE;
        if end > bwt.len() {
            blocks.push(FMBlock {
                bwt_slice: bwt[start..].to_vec(),
                c_arr: c_arr.clone(),
                offsets: counts.clone(),
            });
        } else {
            blocks.push(FMBlock {
                bwt_slice: bwt[start..end].to_vec(),
                c_arr: c_arr.clone(),
                offsets: counts.clone(),
            });

            for j in start..end {
                counts[bwt[j] as usize] += 1;
            }
        }
    }
    blocks
}

// Run LF-mapping on the index
fn lf_map(blocks: &Vec<FMBlock>, ind: usize, chr: u8) -> usize {
    let block_ind = ind / BLOCK_SIZE;
    let ind = ind % BLOCK_SIZE;

    let mut offset = blocks[block_ind].offsets[chr as usize];
    for i in 0..ind {
        if blocks[block_ind].bwt_slice[i] == chr {
            offset += 1;
        }
    }

    return blocks[block_ind].c_arr[chr as usize] + offset;
}

// Search FM-index for a pattern
// Returns (start, end) indices of the pattern in the BWT, end is exclusive
pub fn substring_search(blocks: &Vec<FMBlock>, pattern: &Vec<u8>, n: usize) -> Option<(usize, usize)> {
    let mut start = 0;
    let mut end = n;
    for i in (0..pattern.len()).rev() {
        let chr = pattern[i];
        start = lf_map(blocks, start, chr);
        end = lf_map(blocks, end, chr);
        if start > end {
            return None;
        }
    }

    Some((start, end))
}

// Get all matching line indices from the BWT
pub fn get_matching_lines(bwt_data: &BWTData, blocks: &Vec<FMBlock>, pattern: &Vec<u8>) -> BTreeSet<usize> {
    let (bwt, line_ind, _) = bwt_data;
    let n = bwt.len();
    let res = substring_search(blocks, pattern, n);
    if res.is_none() {
        return BTreeSet::new();
    }

    let (start, end) = res.unwrap();
    let mut lines: BTreeSet<usize> = BTreeSet::new();
    for i in start..end {
        lines.insert(line_ind[i]);
    }
    lines
}
