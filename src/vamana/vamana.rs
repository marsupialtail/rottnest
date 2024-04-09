use bitvector::BitVector;
use ndarray::{s, Array2};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rayon::prelude::*;

use super::{access, InMemoryAccessMethodF32};
use crate::lava::error::LavaError;
use crate::vamana::kmeans::{kmeans, KMeansAssignment};
use futures::StreamExt;
use ndarray::{concatenate, Axis};
use std::sync::{Arc, Mutex};

pub trait Distance<T: Indexable>: std::marker::Send + std::marker::Sync {
    fn calculate(a: &[T], b: &[T]) -> f64;
}

pub trait VectorAccessMethod<T: Indexable>: std::marker::Sync + Send {
    fn get_vec_sync<'a>(&'a self, idx: usize) -> &'a [T];
    async fn get_vec<'a>(&'a self, idx: usize) -> Vec<T>;
    fn dim(&self) -> usize;
    fn num_points(&self) -> usize;
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a [T]>;
    fn par_iter<'a>(&'a self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'a [T]>;
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

#[derive(Debug, Clone, Copy)]
pub struct IndexParams {
    pub num_neighbors: usize,
    pub search_frontier_size: usize,
    pub pruning_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct VamanaIndex<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>> {
    params: IndexParams,
    pub neighbors: Array2<usize>,
    access_method: V,
    pub start: usize,
    metric: std::marker::PhantomData<D>,
    t: std::marker::PhantomData<T>,
}

// TODO: Try this sparse set: https://research.swtch.com/sparse
#[derive(Clone)]
struct VisitedSet {
    visited: BitVector,
}

#[derive(Clone)]
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

#[derive(Clone)]
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
        let start_vector = self.get_vector(self.start).await;
        let start_distance = D::calculate(query, &start_vector);
        let mut closest_unvisited_vertex = 0;
        ctx.frontier.push((self.start, start_distance));
        while closest_unvisited_vertex < ctx.frontier.len() {
            let closest = ctx.frontier[closest_unvisited_vertex];
            ctx.visited.visit(closest.0);
            for n in self.neighbors(closest.0) {
                let neighbor_vector = self.get_vector(*n).await;
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

    fn neighbors_mut<'a>(&'a mut self, idx: usize) -> &'a mut [usize] {
        let num_neighbors = self.num_neighbors(idx);
        self.neighbors
            .slice_mut(s![idx, 1..(num_neighbors + 1)])
            .reborrow()
            .into_slice()
            .unwrap()
    }

    pub async fn get_vector(&self, idx: usize) -> Vec<T> {
        self.access_method.get_vec(idx).await
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

    pub async fn search<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
        &mut self,
        index: &VamanaIndex<T, D, V>,
        query: usize,
    ) {
        self.reset();
        let query_vector = index.get_vector(query).await;
        let start_vector = index.get_vector(index.start).await;
        let start_distance = D::calculate(&query_vector, &start_vector);
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
                let neighbor_vector = index.get_vector(*n).await;
                let distance = D::calculate(&query_vector, &neighbor_vector);
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

    pub async fn prune_index<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
        &mut self,
        index: &mut VamanaIndex<T, D, V>,
        query: usize,
        pruning_threshold: f64,
    ) {
        let query_vector = index.get_vector(query).await;
        for n in index.neighbors(query) {
            self.search_ctx.visited.visit(*n);
            let neighbor_vector = index.get_vector(*n).await;
            let distance = D::calculate(&neighbor_vector, &query_vector);
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
            let curr_vec = index.get_vector(v).await;
            for ielim in (ivisited + 1)..self.cached_distances.len() {
                let elim_id = self.cached_distances[ielim].0;
                if !self.search_ctx.visited.is_visited(elim_id) {
                    continue;
                }
                let elim_vec = index.get_vector(elim_id).await;
                let distance = D::calculate(&curr_vec, &elim_vec);
                if pruning_threshold * distance < d {
                    self.search_ctx.visited.mark_unvisited(elim_id);
                }
            }
        }
    }

    pub async fn insert_backwards_edges<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
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
                let query_vector = index.get_vector(query).await;
                let neighbor_vector = index.get_vector(neighbor).await;
                let distance = D::calculate(&query_vector, &neighbor_vector);
                self.reset();
                self.search_ctx.visited.visit(query);
                self.cached_distances.push((query, distance));
                self.prune_index(index, neighbor, pruning_threshold).await;
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

struct PartitionedAccessMethod<T: Indexable, V: VectorAccessMethod<T>> {
    partition_id: usize,
    underlying_access_method: Arc<V>,
    partition_assignment: Arc<KMeansAssignment>,
    t: std::marker::PhantomData<T>,
}

impl<T: Indexable, V: VectorAccessMethod<T>> VectorAccessMethod<T>
    for PartitionedAccessMethod<T, V>
{
    fn get_vec_sync<'b>(&'b self, local_idx: usize) -> &'b [T] {
        let global_idx = self
            .partition_assignment
            .get_global_idx(self.partition_id, local_idx);
        self.underlying_access_method.get_vec_sync(global_idx)
    }
    async fn get_vec<'b>(&'b self, local_idx: usize) -> Vec<T> {
        let global_idx = self
            .partition_assignment
            .get_global_idx(self.partition_id, local_idx);
        self.underlying_access_method.get_vec(global_idx).await
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
            .map(|i| self.underlying_access_method.get_vec_sync(*i))
    }

    #[allow(unreachable_code)]
    fn par_iter<'b>(&'b self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'b [T]> {
        panic!("Calling par_iter on a partition is probably wrong!");
        self.underlying_access_method.par_iter()
    }
}

pub async fn build_index<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
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
            build_ctx.search(&index, *v).await;
            build_ctx.prune_index(&mut index, *v, prune).await;
            build_ctx
                .insert_backwards_edges(&mut index, *v, prune)
                .await;
        }
    }
    index
}

#[tokio::main]
pub async fn build_index_par<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: V,
    params: IndexParams,
) -> VamanaIndex<T, D, V> {
    let num_partitions = 2 * rayon::current_num_threads();
    println!("Building with {} partitions", num_partitions);
    let partition_assignment = Arc::new(kmeans::<T, D, V>(&access_method, num_partitions));
    let access_method = Arc::new(access_method);
    let partition_handles = (0..num_partitions)
        .into_iter()
        .map(|partition_idx| {
            let partition_assignment = partition_assignment.clone();
            let params_clone = IndexParams {
                num_neighbors: params.num_neighbors / 2,
                search_frontier_size: params.search_frontier_size,
                pruning_threshold: params.pruning_threshold,
            };
            let access_method = access_method.clone();

            tokio::task::spawn(async move {
                let partitioned_access_method = PartitionedAccessMethod {
                    partition_id: partition_idx,
                    underlying_access_method: access_method,
                    partition_assignment: partition_assignment,
                    t: std::marker::PhantomData,
                };

                build_index::<T, D, _>(partitioned_access_method, params_clone).await
            })
        })
        .collect::<Vec<_>>();

    let results: Vec<VamanaIndex<T, D, _>> = futures::future::join_all(partition_handles)
        .await
        .into_iter()
        .map(|result| result.unwrap())
        .collect();

    let mut big_graph = Array2::zeros((access_method.num_points(), params.num_neighbors + 1));
    big_graph
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(iglobal, mut edgelist_including_length)| {
            let (a1, a2) = partition_assignment.get_partitions_for_element(iglobal);
            let (ilocal_a1, ilocal_a2) = partition_assignment.get_local_idx(iglobal);
            let edges_a1 = results[a1].neighbors(ilocal_a1);
            let edges_a2 = results[a2].neighbors(ilocal_a2);

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
    drop(results);
    let big_graph_start = compute_index_of_vector_closest_to_mean::<T, D, V>(&access_method);
    let access_method = match Arc::try_unwrap(access_method) {
        Ok(access_method) => access_method,
        Err(_) => panic!("access_method should be closed"),
    }
    VamanaIndex {
        params: params,
        neighbors: big_graph,
        access_method: access_method,
        start: big_graph_start,
        metric: std::marker::PhantomData,
        t: std::marker::PhantomData,
    }
}

pub struct MergedAccessMethod<T: Indexable, V: VectorAccessMethod<T>> {
    underlying_access_method: (V, V),
    t: std::marker::PhantomData<T>,
}

impl<T: Indexable, V: VectorAccessMethod<T>> VectorAccessMethod<T> for MergedAccessMethod<T, V> {
    fn get_vec_sync<'b>(&'b self, ivec: usize) -> &'b [T] {
        unimplemented!("get_vec not implemented for ReaderAccessMethodF32")
    }

    async fn get_vec<'b>(&'b self, ivec: usize) -> Vec<T> {
        let num_points_0 = self.underlying_access_method.0.num_points();
        if ivec < num_points_0 {
            self.underlying_access_method.0.get_vec(ivec).await
        } else {
            self.underlying_access_method
                .1
                .get_vec(ivec - num_points_0)
                .await
        }
    }

    fn dim(&self) -> usize {
        self.underlying_access_method.0.dim()
    }

    fn num_points(&self) -> usize {
        self.underlying_access_method.0.num_points() + self.underlying_access_method.1.num_points()
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'b [T]> {
        self.underlying_access_method
            .0
            .iter()
            .chain(self.underlying_access_method.1.iter())
    }

    #[allow(unreachable_code)]
    fn par_iter<'b>(&'b self) -> impl rayon::prelude::IndexedParallelIterator<Item = &'b [T]> {
        self.underlying_access_method
            .0
            .par_iter()
            .chain(self.underlying_access_method.1.par_iter())
    }
}

async unsafe fn search_merge<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    ctx: &mut BuildContext,
    start: usize,
    query: usize,
    access_method: &V,
    edgelist: *mut usize,
    locks: &Vec<Mutex<()>>,
    max_num_neighbors: usize,
    search_frontier_size: usize,
) {
    ctx.reset();
    let dim = max_num_neighbors + 1;
    let query_vector = access_method.get_vec(query).await;
    let start_vector = access_method.get_vec(start).await;
    let start_distance = D::calculate(&query_vector, &start_vector);
    let mut closest_unvisited_vertex = 0;
    ctx.search_ctx.frontier.push((start, start_distance));
    while closest_unvisited_vertex < ctx.search_ctx.frontier.len() {
        let closest = ctx.search_ctx.frontier[closest_unvisited_vertex];
        let _guard = locks[closest.0].lock().unwrap();
        ctx.cached_distances.push(closest);
        ctx.search_ctx.visited.visit(closest.0);
        let num_neighbors = *edgelist.add(closest.0 * dim);
        for ineighbor in 0..num_neighbors {
            let n = *edgelist.add(closest.0 * dim + ineighbor + 1);
            let neighbor_vector = access_method.get_vec(n).await;
            let distance = D::calculate(&query_vector, &neighbor_vector);
            ctx.search_ctx.frontier.push((n, distance));
        }
        ctx.search_ctx
            .frontier
            .sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let new_frontier_size =
            dedup_frontier(&mut ctx.search_ctx.frontier).min(search_frontier_size);
        ctx.search_ctx.frontier.truncate(new_frontier_size);
        closest_unvisited_vertex = ctx.search_ctx.frontier.len();
        for i in 0..ctx.search_ctx.frontier.len() {
            let v = ctx.search_ctx.frontier[i].0;
            if !ctx.search_ctx.visited.is_visited(v) {
                closest_unvisited_vertex = i;
                break;
            }
        }
    }
}

async unsafe fn prune_merge<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    ctx: &mut BuildContext,
    query: usize,
    access_method: &V,
    edgelist: *mut usize,
    locks: &Vec<Mutex<()>>,
    pruning_threshold: f64,
    max_num_neighbors: usize,
    should_lock: bool,
) {
    let dim = max_num_neighbors + 1;
    let query_vector = access_method.get_vec(query).await;
    let _guard = if should_lock {
        Some(locks[query].lock().unwrap())
    } else {
        None
    };
    let num_neighbors = *edgelist.add(query * dim);
    for ineighbor in 0..num_neighbors {
        let n = *edgelist.add(query * dim + ineighbor + 1);
        ctx.search_ctx.visited.visit(n);
        let neighbor_vector = access_method.get_vec(n).await;
        let distance = D::calculate(&neighbor_vector, &query_vector);
        ctx.cached_distances.push((n, distance));
    }
    ctx.search_ctx.visited.mark_unvisited(query);
    ctx.cached_distances
        .sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

    *edgelist.add(query * dim) = 0;
    for ivisited in 0..ctx.cached_distances.len() {
        let (v, d) = ctx.cached_distances[ivisited];
        if !ctx.search_ctx.visited.is_visited(v) {
            continue;
        }
        let new_num_neighbors = *edgelist.add(query * dim) + 1;
        *edgelist.add(query * dim) = new_num_neighbors;
        *edgelist.add(query * dim + new_num_neighbors) = v;
        if new_num_neighbors == max_num_neighbors {
            return;
        }
        let curr_vec = access_method.get_vec(v).await;
        for ielim in (ivisited + 1)..ctx.cached_distances.len() {
            let elim_id = ctx.cached_distances[ielim].0;
            if !ctx.search_ctx.visited.is_visited(elim_id) {
                continue;
            }
            let elim_vec = access_method.get_vec(elim_id).await;
            let distance = D::calculate(&curr_vec, &elim_vec);
            if pruning_threshold * distance < d {
                ctx.search_ctx.visited.mark_unvisited(elim_id);
            }
        }
    }
}

async unsafe fn insert_backwards_edges_merge<
    T: Indexable,
    D: Distance<T>,
    V: VectorAccessMethod<T>,
>(
    ctx: &mut BuildContext,
    query: usize,
    access_method: &V,
    edgelist: *mut usize,
    locks: &Vec<Mutex<()>>,
    pruning_threshold: f64,
    max_num_neighbors: usize,
) {
    let dim = max_num_neighbors + 1;
    let (num_neighbors, neighbors) = {
        let _guard = locks[query].lock().unwrap();
        let num_neighbors = *edgelist.add(query * dim);
        let neighbors = (0..num_neighbors)
            .map(|x| *edgelist.add(query * dim + x + 1))
            .collect::<Vec<_>>();
        (num_neighbors, neighbors)
    };

    for ineighbor in 0..num_neighbors {
        let neighbor = neighbors[ineighbor];
        let _neighbor_guard = locks[neighbor].lock().unwrap();
        let num_neighbor_neighbors = *edgelist.add(neighbor * dim);
        if num_neighbor_neighbors < max_num_neighbors {
            let new_num_neighbor_neighbors = num_neighbor_neighbors + 1;
            *edgelist.add(neighbor * dim) = new_num_neighbor_neighbors;
            *edgelist.add(neighbor * dim + new_num_neighbor_neighbors) = query;
        } else {
            let query_vector = access_method.get_vec(query).await;
            let neighbor_vector = access_method.get_vec(neighbor).await;
            let distance = D::calculate(&query_vector, &neighbor_vector);
            ctx.reset();
            ctx.search_ctx.visited.visit(query);
            ctx.cached_distances.push((query, distance));
            for inn in 0..num_neighbor_neighbors {
                let nn = *edgelist.add(neighbor * dim + inn + 1);
                let nn_vec = access_method.get_vec(nn).await;
                let d = D::calculate(&nn_vec, &neighbor_vector);
                ctx.search_ctx.visited.visit(nn);
                ctx.cached_distances.push((nn, d));
            }
            prune_merge::<T, D, V>(
                ctx,
                neighbor,
                access_method,
                edgelist,
                locks,
                pruning_threshold,
                max_num_neighbors,
                false, /* should_lock */
            )
            .await;
        }
    }
}

async unsafe fn insert_single<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    build_ctx: &mut BuildContext,
    start: usize,
    to_insert: usize,
    access_method: &V,
    edgelist: *mut usize,
    locks: &Vec<Mutex<()>>,
    pruning_threshold: f64,
    max_num_neighbors: usize,
    search_frontier_size: usize,
) {
    search_merge::<T, D, V>(
        build_ctx,
        start,
        to_insert,
        access_method,
        edgelist,
        locks,
        search_frontier_size,
        max_num_neighbors,
    )
    .await;
    prune_merge::<T, D, V>(
        build_ctx,
        to_insert,
        access_method,
        edgelist,
        locks,
        pruning_threshold,
        max_num_neighbors,
        true, /* should_lock */
    )
    .await;
    insert_backwards_edges_merge::<T, D, V>(
        build_ctx,
        to_insert,
        access_method,
        edgelist,
        locks,
        pruning_threshold,
        max_num_neighbors,
    )
    .await;
}

pub fn merge_indexes<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    a: VamanaIndex<T, D, V>,
    b: VamanaIndex<T, D, V>,
) -> VamanaIndex<T, D, MergedAccessMethod<T, V>> {
    let num_points = a.access_method.num_points() + b.access_method.num_points();
    let b_offset = a.access_method.num_points();
    let params = a.params;
    let access_method = MergedAccessMethod {
        underlying_access_method: (a.access_method, b.access_method),
        t: std::marker::PhantomData,
    };
    let big_graph = concatenate!(Axis(0), a.neighbors, b.neighbors);
    let start = a.start;
    let mut merged_index = VamanaIndex {
        params: params,
        neighbors: big_graph,
        access_method: access_method,
        start: start,
        metric: std::marker::PhantomData,
        t: std::marker::PhantomData,
    };

    for i in b_offset..num_points {
        merged_index
            .neighbors_mut(i)
            .iter_mut()
            .for_each(|x| *x += b_offset)
    }

    let mut build_ctx = BuildContext::new(&merged_index);
    let prune = merged_index.params.pruning_threshold;
    let num_total_points = merged_index.num_points();
    for b_vertex in (b_offset..num_total_points) {
        build_ctx.search(&merged_index, b_vertex);
        build_ctx.prune_index(&mut merged_index, b_vertex, prune);
        build_ctx.insert_backwards_edges(&mut merged_index, b_vertex, prune);
    }
    merged_index
}

struct PtrWrapper(*mut usize);
unsafe impl Sync for PtrWrapper {}

#[tokio::main]
pub async fn merge_indexes_par<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    a: VamanaIndex<T, D, V>,
    b: VamanaIndex<T, D, V>,
) -> VamanaIndex<T, D, MergedAccessMethod<T, V>> {
    let num_points = a.access_method.num_points() + b.access_method.num_points();
    let b_offset = a.access_method.num_points();
    let params = a.params;
    let access_method = MergedAccessMethod {
        underlying_access_method: (a.access_method, b.access_method),
        t: std::marker::PhantomData,
    };
    let big_graph = concatenate!(Axis(0), a.neighbors, b.neighbors);
    let start = a.start;
    let mut merged_index = VamanaIndex {
        params: params,
        neighbors: big_graph,
        access_method: access_method,
        start: start,
        metric: std::marker::PhantomData,
        t: std::marker::PhantomData,
    };

    for i in b_offset..num_points {
        merged_index
            .neighbors_mut(i)
            .iter_mut()
            .for_each(|x| *x += b_offset)
    }

    let mut build_ctx = BuildContext::new(&merged_index);
    let num_total_points = merged_index.num_points();
    let mut locks = Vec::with_capacity(num_total_points);
    for _ in 0..num_total_points {
        locks.push(Mutex::new(()));
    }
    
    let tasks: Vec<_> = (b_offset..num_total_points)
        .into_iter()
        .map(|b_vertex| {
            let tl_ctx = build_ctx.clone();
            let edgelist_ptr = PtrWrapper(merged_index.neighbors.as_mut_ptr());
            // let locks = locks.clone();
            let prune = merged_index.params.pruning_threshold;
            let num_neighbors = merged_index.params.num_neighbors;
            let search_frontier_size = merged_index.params.search_frontier_size;
            let _ = &edgelist_ptr;
            tokio::spawn(async move {
                unsafe {
                    insert_single::<T, D, MergedAccessMethod<T, V>>(
                        &mut tl_ctx,
                        start,
                        b_vertex,
                        &merged_index.access_method,
                        edgelist_ptr.0,
                        &locks,
                        prune,
                        num_neighbors,
                        search_frontier_size,
                    );
                }
            })
        }).collect();
    
    let results = futures::future::join_all(tasks).await;

    merged_index
}
