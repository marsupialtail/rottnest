use ndarray::{Array2, s};
use ndarray::parallel::prelude::*;
use crate::vamana::{Distance, build_index_par, IndexParams, VectorAccessMethod, Indexable};

mod vamana;
mod kmeans;

struct Euclidean<T : Indexable>
{
    t : std::marker::PhantomData<T>,
}

impl<T : Indexable> Distance<T> for Euclidean<T> where
    T : std::ops::Sub<Output = T> + std::ops::Mul<T, Output = T> + std::clone::Clone + std::iter::Sum<<T as std::ops::Mul>::Output> + num_traits::cast::AsPrimitive<f64>
{
    #[inline(always)]
    fn calculate(a : &[T], b : &[T]) -> f64
    {
        let mut result = T::zero();
        for (x, y) in a.iter().zip(b.iter())
        {
            result += (*x - *y) * (*x - *y);
        }
        result.as_()
    }
}

struct InMemoryAccessMethodF32
{
    data : Array2<f32>,
}

struct EuclideanF32;

impl Distance<f32> for EuclideanF32
{
    #[inline(always)]
    fn calculate(a : &[f32], b : &[f32]) -> f64
    {
        let mut result = 0.0;
        for (x, y) in a.iter().zip(b.iter())
        {
            result += (*x - *y) * (*x - *y);
        }
        result as f64
    }
}

impl VectorAccessMethod<f32> for InMemoryAccessMethodF32 
{
    fn get_vec<'a>(&'a self, idx : usize) -> &'a [f32]
    {
        self.data.slice(s![idx, ..]).reborrow().to_slice().unwrap()
    }

    fn dim(&self) -> usize
    {
        self.data.shape()[1]
    }

    fn num_points(&self) -> usize
    {
        self.data.shape()[0]
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a [f32]>
    {
        self.data.outer_iter().map(|x| x.reborrow().to_slice().unwrap())
    }

    fn par_iter<'a>(&'a self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'a [f32]>
    {
        self.data.outer_iter().into_par_iter().map(|x| x.reborrow().to_slice().unwrap())
    }
}

fn read_dataset(file_path : &str, dataset_name : &str) -> hdf5::Result<Array2<f32>>
{
    let file = hdf5::File::open(file_path)?;
    let dataset = file.dataset(dataset_name)?;
    let data : Array2<f32> = dataset.read()?;
    Ok(data)
}

fn read_neighbors(file_path : &str) -> hdf5::Result<Array2<i32>>
{
    let file = hdf5::File::open(file_path)?;
    let dataset = file.dataset("neighbors")?;
    let data : Array2<i32> = dataset.read()?;
    Ok(data)
}

fn main()
{
    let file = "/home/sasha/Downloads/mnist-784-euclidean.hdf5";
    let train = read_dataset(file, "train").unwrap();
    let test = read_dataset(file, "test").unwrap();
    let start = std::time::Instant::now();
    let index = build_index_par::<f32, EuclideanF32, _>(
        InMemoryAccessMethodF32 { data : train },
        IndexParams
        {
            num_neighbors : 32,
            search_frontier_size : 32,
            pruning_threshold : 2.0,
        }
    );
    println!("Success! Build took {} sec", start.elapsed().as_secs());
    let mut ctx = index.get_search_context();
    let neighbors = read_neighbors(file).unwrap();
    for (itest, (example, expected_vec)) in test.outer_iter().zip(neighbors.outer_iter()).enumerate()
    {
        index.search(&mut ctx, example.as_slice().unwrap());
        let mut expected : Vec<_> = expected_vec.to_slice().unwrap().iter().map(|x| *x).collect();
        expected.truncate(index.max_num_neighbors());
        let actual = ctx.frontier.iter().map(|(v, _d)| *v as i32);
        let num_in_common = actual.filter(|x| expected.contains(x)).count() as f64;
        let recall = num_in_common / (expected.len() as f64);
        println!("Test {} recall: {}", itest, recall);
    }
}
