use bitvector::BitVector;
use ndarray::{s, Array2};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::vamana::kmeans::{kmeans, KMeansAssignment};
use crate::lava::error::LavaError;

use super::{access, InMemoryAccessMethodF32};

pub trait Distance<T: Indexable>: std::marker::Send + std::marker::Sync {
    fn calculate(a: &[T], b: &[T]) -> f64;
}

pub trait VectorAccessMethod<T: Indexable>: std::marker::Sync {
    fn get_vec<'a>(&'a self, idx: usize) -> &'a [T];
    fn dim(&self) -> usize;
    fn num_points(&self) -> usize;
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a [T]>;
    fn par_iter<'a>(&'a self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'a [T]>;

    fn get_vec_async(&self, idx: usize) -> impl std::future::Future<Output = Result<Vec<T>, LavaError>> {
        async move {
            Ok(self.get_vec(idx).to_vec())
        }
    }
}

pub trait Indexable:
    std::clone::Clone
    + num_traits::Zero
    + num_traits::FromPrimitive
    + std::ops::Div<Output = Self>
    + ndarray::ScalarOperand
    + std::ops::DivAssign
    + std::ops::AddAssign
    + std::marker::Sync
    + std::marker::Send
{
}
impl<T> Indexable for T where
    T: std::clone::Clone
        + num_traits::Zero
        + num_traits::FromPrimitive
        + std::ops::Div<Output = T>
        + ndarray::ScalarOperand
        + std::ops::DivAssign
        + std::ops::AddAssign
        + std::marker::Sync
        + std::marker::Send
{
}

#[derive(Debug)]
pub struct IndexParams {
    pub num_neighbors: usize,
    pub search_frontier_size: usize,
    pub pruning_threshold: f64,
}

pub struct VamanaIndex<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>> {
    params: IndexParams,
    pub neighbors: Array2<usize>,
    access_method: V,
    pub start: usize,
    metric: std::marker::PhantomData<D>,
    t: std::marker::PhantomData<T>,
}

// TODO: Try this sparse set: https://research.swtch.com/sparse
struct VisitedSet {
    visited: BitVector,
}

pub struct SearchContext {
    visited: VisitedSet,
    pub frontier: Vec<(usize, f64)>,
}

impl SearchContext {
    pub fn reset(&mut self) {
        self.visited.clear();
        self.frontier.clear();
    }
}

struct BuildContext {
    search_ctx: SearchContext,
    cached_distances: Vec<(usize, f64)>,
}

impl<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>> VamanaIndex<T, D, V> {
    fn new(access_method: V, params: IndexParams) -> VamanaIndex<T, D, V> {
        let num_points = access_method.num_points();
        let max_num_neighbors = params.num_neighbors;
        let neighbors = Array2::zeros((num_points, max_num_neighbors + 1));

        VamanaIndex {
            params: params,
            neighbors: neighbors,
            access_method: access_method,
            start: 0,
            metric: std::marker::PhantomData,
            t: std::marker::PhantomData,
        }
    }

    pub fn hydrate(
        access_method: V,
        params: IndexParams,
        neighbors: ndarray::prelude::ArrayBase<
            ndarray::OwnedRepr<usize>,
            ndarray::prelude::Dim<[usize; 2]>,
        >,
        start: usize,
    ) -> VamanaIndex<T, D, V> {
        let num_points = access_method.num_points();
        let max_num_neighbors = params.num_neighbors;

        VamanaIndex {
            params: params,
            neighbors: neighbors,
            access_method: access_method,
            start: start,
            metric: std::marker::PhantomData,
            t: std::marker::PhantomData,
        }
    }

    pub fn get_search_context(&self) -> SearchContext {
        let frontier_capacity = self.max_num_neighbors() + self.search_frontier_size();
        SearchContext {
            visited: VisitedSet::new(self.num_points()),
            frontier: Vec::with_capacity(frontier_capacity),
        }
    }

    pub async fn search(&self, ctx: &mut SearchContext, query: &[T]) -> Result<(), LavaError> {
        ctx.reset();
        let start_vector = self.get_vector_async(self.start).await?;
        let start_distance = D::calculate(query, &start_vector);
        let mut closest_unvisited_vertex = 0;
        ctx.frontier.push((self.start, start_distance));
        while closest_unvisited_vertex < ctx.frontier.len() {
            let closest = ctx.frontier[closest_unvisited_vertex];
            ctx.visited.visit(closest.0);
            for n in self.neighbors(closest.0) {
                let neighbor_vector = self.get_vector_async(*n).await?;
                let distance = D::calculate(query, &neighbor_vector);
                ctx.frontier.push((*n, distance));
            }
            ctx.frontier
                .sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
            let new_frontier_size =
                dedup_frontier(&mut ctx.frontier).min(self.search_frontier_size());
            ctx.frontier.truncate(new_frontier_size);
            closest_unvisited_vertex = ctx.frontier.len();
            for i in 0..ctx.frontier.len() {
                let v = ctx.frontier[i].0;
                if !ctx.visited.is_visited(v) {
                    closest_unvisited_vertex = i;
                    break;
                }
            }
        }
        Ok(())
    }

    pub fn max_num_neighbors(&self) -> usize {
        self.params.num_neighbors
    }

    pub fn search_frontier_size(&self) -> usize {
        self.params.search_frontier_size
    }

    pub fn num_points(&self) -> usize {
        self.access_method.num_points()
    }

    pub fn num_neighbors(&self, idx: usize) -> usize {
        self.neighbors[[idx, 0]]
    }

    fn clear_neighbors(&mut self, idx: usize) {
        self.neighbors[[idx, 0]] = 0
    }

    pub fn neighbors<'a>(&'a self, idx: usize) -> &'a [usize] {
        let num_neighbors = self.num_neighbors(idx);
        self.neighbors
            .slice(s![idx, 1..(num_neighbors + 1)])
            .reborrow()
            .to_slice()
            .unwrap()
    }

    pub fn get_vector<'a>(&'a self, idx: usize) -> &'a [T] {
        self.access_method.get_vec(idx)
    }

    pub async fn get_vector_async(&self, idx: usize) -> Result<Vec<T>, LavaError> {
        self.access_method.get_vec_async(idx).await
    }

    fn push_neighbor(&mut self, idx: usize, new_neighbor: usize) {
        let cur_num_neighbors = self.num_neighbors(idx);
        self.neighbors[[idx, cur_num_neighbors + 1]] = new_neighbor;
        self.neighbors[[idx, 0]] += 1;
    }
}

impl VisitedSet {
    pub fn new(num_points: usize) -> Self {
        VisitedSet {
            visited: BitVector::new(num_points),
        }
    }

    pub fn clear(&mut self) {
        self.visited.clear();
    }

    pub fn visit(&mut self, id: usize) {
        self.visited.insert(id);
    }

    pub fn mark_unvisited(&mut self, id: usize) {
        self.visited.remove(id);
    }

    pub fn is_visited(&self, id: usize) -> bool {
        self.visited.contains(id)
    }
}

impl BuildContext {
    pub fn new<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
        index: &VamanaIndex<T, D, V>,
    ) -> BuildContext {
        let distance_cache_size = std::cmp::max(
            index.num_points(),
            index.search_frontier_size() + index.max_num_neighbors(),
        );
        BuildContext {
            search_ctx: index.get_search_context(),
            cached_distances: Vec::with_capacity(distance_cache_size),
        }
    }

    pub fn search<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
        &mut self,
        index: &VamanaIndex<T, D, V>,
        query: usize,
    ) {
        self.reset();
        let query_vector = index.get_vector(query);
        let start_vector = index.get_vector(index.start);
        let start_distance = D::calculate(query_vector, start_vector);
        let mut closest_unvisited_vertex = 0;
        self.search_ctx.frontier.push((index.start, start_distance));
        assert!(
            self.search_ctx.frontier.len()
                <= index.max_num_neighbors() + index.search_frontier_size()
        );
        while closest_unvisited_vertex < self.search_ctx.frontier.len() {
            let closest = self.search_ctx.frontier[closest_unvisited_vertex];
            self.cached_distances.push(closest);
            self.search_ctx.visited.visit(closest.0);
            for n in index.neighbors(closest.0) {
                let neighbor_vector = index.get_vector(*n);
                let distance = D::calculate(query_vector, neighbor_vector);
                self.search_ctx.frontier.push((*n, distance));
                assert!(
                    self.search_ctx.frontier.len()
                        <= index.max_num_neighbors() + index.search_frontier_size()
                );
            }
            self.search_ctx
                .frontier
                .sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
            let new_frontier_size =
                dedup_frontier(&mut self.search_ctx.frontier).min(index.search_frontier_size());
            self.search_ctx.frontier.truncate(new_frontier_size);
            closest_unvisited_vertex = self.search_ctx.frontier.len();
            for i in 0..self.search_ctx.frontier.len() {
                let v = self.search_ctx.frontier[i].0;
                if !self.search_ctx.visited.is_visited(v) {
                    closest_unvisited_vertex = i;
                    break;
                }
            }
        }
    }

    pub fn prune_index<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
        &mut self,
        index: &mut VamanaIndex<T, D, V>,
        query: usize,
        pruning_threshold: f64,
    ) {
        let query_vector = index.get_vector(query);
        for n in index.neighbors(query) {
            self.search_ctx.visited.visit(*n);
            let neighbor_vector = index.get_vector(*n);
            let distance = D::calculate(neighbor_vector, query_vector);
            self.cached_distances.push((*n, distance));
        }
        self.search_ctx.visited.mark_unvisited(query);
        self.cached_distances
            .sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

        index.clear_neighbors(query);
        for ivisited in 0..self.cached_distances.len() {
            let (v, d) = self.cached_distances[ivisited];
            if !self.search_ctx.visited.is_visited(v) {
                continue;
            }
            index.push_neighbor(query, v);
            if index.num_neighbors(query) == index.params.num_neighbors {
                return;
            }
            let curr_vec = index.get_vector(v);
            for ielim in (ivisited + 1)..self.cached_distances.len() {
                let elim_id = self.cached_distances[ielim].0;
                if !self.search_ctx.visited.is_visited(elim_id) {
                    continue;
                }
                let elim_vec = index.get_vector(elim_id);
                let distance = D::calculate(curr_vec, elim_vec);
                if pruning_threshold * distance < d {
                    self.search_ctx.visited.mark_unvisited(elim_id);
                }
            }
        }
    }

    pub fn insert_backwards_edges<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
        &mut self,
        index: &mut VamanaIndex<T, D, V>,
        query: usize,
        pruning_threshold: f64,
    ) {
        let num_neighbors = index.num_neighbors(query);
        for ineighbor in 0..num_neighbors {
            let neighbor = index.neighbors(query)[ineighbor];
            if index.num_neighbors(neighbor) < index.params.num_neighbors {
                index.push_neighbor(neighbor, query);
            } else {
                let query_vector = index.get_vector(query);
                let neighbor_vector = index.get_vector(neighbor);
                let distance = D::calculate(query_vector, neighbor_vector);
                self.reset();
                self.search_ctx.visited.visit(query);
                self.cached_distances.push((query, distance));
                self.prune_index(index, neighbor, pruning_threshold);
            }
        }
    }

    pub fn reset(&mut self) {
        self.search_ctx.reset();
        self.cached_distances.clear();
    }
}

fn randomize_edgelist<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    index: &mut VamanaIndex<T, D, V>,
) {
    let mut rng = rand::thread_rng();
    let max_num_neighbors = index.max_num_neighbors();
    let uniform = Uniform::new(0, max_num_neighbors - 1);
    for (ivert, mut row) in index.neighbors.outer_iter_mut().enumerate() {
        for i in 0..max_num_neighbors {
            let mut neighbor_id = uniform.sample(&mut rng);
            if neighbor_id >= ivert {
                neighbor_id += 1;
            }
            row[1 + i] = neighbor_id;
        }
        let row_slice = &mut row.as_slice_mut().unwrap();
        row_slice[1..].sort_unstable();
        row_slice[0] = dedup_slice(&mut row_slice[1..]);
    }
}

fn dedup_frontier(x: &mut [(usize, f64)]) -> usize {
    let mut offset = 0;
    for i in 1..x.len() {
        if x[i].0 != x[i - 1].0 {
            offset += 1;
            x[offset] = x[i].clone();
        }
    }
    offset
}

fn dedup_slice(x: &mut [usize]) -> usize {
    let mut offset = 0;
    for i in 1..x.len() {
        if x[i] != x[i - 1] {
            offset += 1;
            x[offset] = x[i]
        }
    }
    offset
}

fn compute_index_of_vector_closest_to_mean<
    T: Indexable,
    D: Distance<T>,
    V: VectorAccessMethod<T>,
>(
    access_method: &V,
) -> usize {
    let mut mean: Vec<T> = vec![T::zero(); access_method.dim()];
    let num_points_as_t = T::from_usize(access_method.num_points()).unwrap();
    access_method.iter().for_each(|x| {
        for (mi, xi) in mean.iter_mut().zip(x.iter()) {
            *mi += xi.clone();
        }
    });
    mean.iter_mut()
        .for_each(|mi| *mi /= num_points_as_t.clone());

    let closest = access_method
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let d = D::calculate(x, mean.as_slice());
            (i, d)
        })
        .fold((0, std::f64::INFINITY), |(cur_i, cur_d), (i, d)| {
            if d < cur_d {
                (i, d)
            } else {
                (cur_i, cur_d)
            }
        });
    closest.0
}

struct PartitionedAccessMethod<'a, T: Indexable, V: VectorAccessMethod<T>> {
    partition_id: usize,
    underlying_access_method: &'a V,
    partition_assignment: &'a KMeansAssignment,
    t: std::marker::PhantomData<T>,
}

impl<'a, T: Indexable, V: VectorAccessMethod<T>> VectorAccessMethod<T>
    for PartitionedAccessMethod<'a, T, V>
{
    fn get_vec<'b>(&'b self, local_idx: usize) -> &'b [T] {
        let global_idx = self
            .partition_assignment
            .get_global_idx(self.partition_id, local_idx);
        self.underlying_access_method.get_vec(global_idx)
    }

    fn dim(&self) -> usize {
        self.underlying_access_method.dim()
    }

    fn num_points(&self) -> usize {
        self.partition_assignment
            .num_points_in_partition(self.partition_id)
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'b [T]> {
        self.partition_assignment
            .get_elements_in_partition(self.partition_id)
            .iter()
            .map(|i| self.underlying_access_method.get_vec(*i))
    }

    #[allow(unreachable_code)]
    fn par_iter<'b>(&'b self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'b [T]> {
        panic!("Calling par_iter on a partition is probably wrong!");
        self.underlying_access_method.par_iter()
    }
}

pub fn build_index<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: V,
    params: IndexParams,
) -> VamanaIndex<T, D, V> {
    println!(
        "Building index on dataset of shape {}x{} with {:?}",
        access_method.num_points(),
        access_method.dim(),
        params
    );

    let start_vector = compute_index_of_vector_closest_to_mean::<T, D, V>(&access_method);

    let mut index = VamanaIndex::new(access_method, params);
    index.start = start_vector;
    randomize_edgelist(&mut index);
    let mut build_ctx = BuildContext::new(&index);
    let mut p: Vec<_> = (0..index.num_points()).collect();
    for prune in [1.0, index.params.pruning_threshold] {
        let mut rng = rand::thread_rng();
        p.shuffle(&mut rng);
        for v in p.iter() {
            build_ctx.search(&index, *v);
            build_ctx.prune_index(&mut index, *v, prune);
            build_ctx.insert_backwards_edges(&mut index, *v, prune);
        }
    }
    index
}

pub fn build_index_par<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: V,
    params: IndexParams,
) -> VamanaIndex<T, D, V> {
    let num_partitions = 2 * rayon::current_num_threads();
    println!("Building with {} partitions", num_partitions);
    let partition_assignment = kmeans::<T, D, V>(&access_method, num_partitions);
    let partition_indexes = (0..num_partitions)
        .into_par_iter()
        .map(|partition_idx| {
            let partitioned_access_method = PartitionedAccessMethod {
                partition_id: partition_idx,
                underlying_access_method: &access_method,
                partition_assignment: &partition_assignment,
                t: std::marker::PhantomData,
            };
            let partition_params = IndexParams {
                num_neighbors: params.num_neighbors / 2, // Halve number of neighbors so we can seamlessly merge indexes
                search_frontier_size: params.search_frontier_size,
                pruning_threshold: params.pruning_threshold,
            };
            build_index::<T, D, _>(partitioned_access_method, partition_params)
        })
        .collect::<Vec<_>>();

    let mut big_graph = Array2::zeros((access_method.num_points(), params.num_neighbors + 1));
    big_graph
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(iglobal, mut edgelist_including_length)| {
            let (a1, a2) = partition_assignment.get_partitions_for_element(iglobal);
            let (ilocal_a1, ilocal_a2) = partition_assignment.get_local_idx(iglobal);
            let edges_a1 = partition_indexes[a1].neighbors(ilocal_a1);
            let edges_a2 = partition_indexes[a2].neighbors(ilocal_a2);

            let a1_end = edges_a1.len();
            let a2_end = a1_end + edges_a2.len();
            let mut edgelist = &mut edgelist_including_length.as_slice_mut().unwrap()[1..];
            edgelist[0..a1_end]
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| {
                    *x = partition_assignment.get_global_idx(a1, edges_a1[i]);
                });
            edgelist[a1_end..a2_end]
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| {
                    *x = partition_assignment.get_global_idx(a2, edges_a2[i]);
                });
            edgelist.sort_unstable();
            edgelist_including_length[0] = dedup_slice(&mut edgelist);
        });
    drop(partition_indexes);
    let big_graph_start = compute_index_of_vector_closest_to_mean::<T, D, V>(&access_method);
    VamanaIndex {
        params: params,
        neighbors: big_graph,
        access_method: access_method,
        start: big_graph_start,
        metric: std::marker::PhantomData,
        t: std::marker::PhantomData,
    }
}
