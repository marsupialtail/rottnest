use crate::{build_index_par, IndexParams};
use crate::{EuclideanF32, InMemoryAccessMethodF32};
// mod vamana;

fn read_dataset(file_path: &str, dataset_name: &str) -> hdf5::Result<Array2<f32>> {
    let file = hdf5::File::open(file_path)?;
    let dataset = file.dataset(dataset_name)?;
    let data: Array2<f32> = dataset.read()?;
    Ok(data)
}

fn read_neighbors(file_path: &str) -> hdf5::Result<Array2<i32>> {
    let file = hdf5::File::open(file_path)?;
    let dataset = file.dataset("neighbors")?;
    let data: Array2<i32> = dataset.read()?;
    Ok(data)
}

fn main() {
    let file = "/home/sasha/Downloads/mnist-784-euclidean.hdf5";
    let train = read_dataset(file, "train").unwrap();
    let test = read_dataset(file, "test").unwrap();
    let start = std::time::Instant::now();
    let index: VamanaIndex<f32, EuclideanF32, _> = build_index_par::<f32, EuclideanF32, _>(
        InMemoryAccessMethodF32 { data: train },
        IndexParams {
            num_neighbors: 32,
            search_frontier_size: 32,
            pruning_threshold: 2.0,
        },
    );
    println!("Success! Build took {} sec", start.elapsed().as_secs());
    let mut ctx = index.get_search_context();
    let neighbors = read_neighbors(file).unwrap();
    for (itest, (example, expected_vec)) in
        test.outer_iter().zip(neighbors.outer_iter()).enumerate()
    {
        index.search(&mut ctx, example.as_slice().unwrap());
        let mut expected: Vec<_> = expected_vec
            .to_slice()
            .unwrap()
            .iter()
            .map(|x| *x)
            .collect();
        expected.truncate(index.max_num_neighbors());
        let actual = ctx.frontier.iter().map(|(v, _d)| *v as i32);
        let num_in_common = actual.filter(|x| expected.contains(x)).count() as f64;
        let recall = num_in_common / (expected.len() as f64);
        println!("Test {} recall: {}", itest, recall);
    }
}
