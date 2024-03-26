use std::collections::HashMap;

use crate::formats::parquet::read_indexed_pages_async;
use crate::vamana::vamana::{
    build_index_par, Distance, IndexParams, Indexable, VectorAccessMethod,
};
use arrow::array::BinaryArray;
use arrow::compute::binary;
use arrow::datatypes::ToByteSlice;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2};
use opendal::Reader;
use rayon::iter::empty;

pub struct Euclidean<T: Indexable> {
    t: std::marker::PhantomData<T>,
}

impl<T: Indexable> Distance<T> for Euclidean<T>
where
    T: std::ops::Sub<Output = T>
        + std::ops::Mul<T, Output = T>
        + std::clone::Clone
        + std::iter::Sum<<T as std::ops::Mul>::Output>
        + num_traits::cast::AsPrimitive<f64>,
{
    #[inline(always)]
    fn calculate(a: &[T], b: &[T]) -> f64 {
        let mut result = T::zero();
        for (x, y) in a.iter().zip(b.iter()) {
            result += (*x - *y) * (*x - *y);
        }
        result.as_()
    }
}

pub struct InMemoryAccessMethodF32 {
    pub data: Array2<f32>,
}

pub struct ReaderAccessMethodF32<'a> {
    pub dim: usize,
    pub num_points: usize,
    pub column_name: String,
    pub uid_nrows: &'a Vec<usize>,
    // uid to (file_path, row_group, page_offset, page_size, dict_page_size)
    pub uid_to_metadata: &'a Vec<(String, usize, usize, usize, usize)>,
}

impl VectorAccessMethod<f32> for ReaderAccessMethodF32<'_> {
    fn get_vec<'a>(&'a self, idx: usize) -> &'a [f32] {
        // self.data.slice(s![idx, ..]).reborrow().to_slice().unwrap()

        // the uid_nrows will look something like 0, 300, 600, 900 etc.
        // we want to find the idx that is smaller than idx, say it's x
        // then uid = x + 1 and offset = idx - uid_nrows[x]

        let x = match self.uid_nrows.binary_search(&idx) {
            Ok(index) => index,
            Err(index) => index - 1,
        };
        let uid = x;
        let offset = idx - self.uid_nrows[x];

        let (file_path, row_group, page_offset, page_size, dict_page_size) =
            self.uid_to_metadata[uid].clone();

        let array_data = read_indexed_pages_async(
            self.column_name.clone(),
            vec![file_path],
            vec![row_group],
            vec![page_offset as u64],
            vec![page_size],
            vec![dict_page_size], // 0 means no dict page
        )
        .await
        .unwrap()
        .remove(0);

        let binary_array = BinaryArray::from(array_data);
        let fetched_slice = binary_array.value(offset);
        // now we need to interpret the u8 binary slice as f32

        let mut result = Box::new(Vec::with_capacity(self.dim));
        for i in 0..self.dim {
            result.push(f32::from_le_bytes(
                fetched_slice[i * 4..(i + 1) * 4]
                    .to_vec()
                    .as_slice()
                    .try_into()
                    .unwrap(),
            ));
        }
        Box::leak(result)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn num_points(&self) -> usize {
        self.num_points
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a [f32]> {
        std::iter::empty()
    }

    fn par_iter<'a>(&'a self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'a [f32]> {
        rayon::iter::empty::<&'a [f32]>()
    }
}

pub struct EuclideanF32;

impl Distance<f32> for EuclideanF32 {
    #[inline(always)]
    fn calculate(a: &[f32], b: &[f32]) -> f64 {
        let mut result = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            result += (*x - *y) * (*x - *y);
        }
        result as f64
    }
}

impl VectorAccessMethod<f32> for InMemoryAccessMethodF32 {
    fn get_vec<'a>(&'a self, idx: usize) -> &'a [f32] {
        self.data.slice(s![idx, ..]).reborrow().to_slice().unwrap()
    }

    fn dim(&self) -> usize {
        self.data.shape()[1]
    }

    fn num_points(&self) -> usize {
        self.data.shape()[0]
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a [f32]> {
        self.data
            .outer_iter()
            .map(|x| x.reborrow().to_slice().unwrap())
    }

    fn par_iter<'a>(&'a self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'a [f32]> {
        self.data
            .outer_iter()
            .into_par_iter()
            .map(|x| x.reborrow().to_slice().unwrap())
    }
}
