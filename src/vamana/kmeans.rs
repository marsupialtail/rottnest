use crate::vamana::vamana::{Distance, Indexable, VectorAccessMethod};
use ndarray::{Array2, ArrayViewMut1};
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub struct KMeansAssignment {
    assignments: Vec<(usize, usize)>,
    partition_counts_prefix_sum: Vec<usize>,
    partitions: Vec<usize>,
    local_indices: Vec<(AtomicUsize, AtomicUsize)>,
}

impl KMeansAssignment {
    fn new(k: usize, num_points: usize) -> Self {
        KMeansAssignment {
            assignments: vec![(0, 0); num_points],
            partition_counts_prefix_sum: vec![0; k + 1],
            partitions: vec![0; 2 * num_points],
            local_indices: (0..num_points)
                .map(|_| (AtomicUsize::new(0), AtomicUsize::new(0)))
                .collect(),
        }
    }

    pub fn get_partitions_for_element(&self, global_idx: usize) -> (usize, usize) {
        self.assignments[global_idx]
    }

    pub fn num_points_in_partition(&self, partition_idx: usize) -> usize {
        self.partition_counts_prefix_sum[partition_idx + 1]
            - self.partition_counts_prefix_sum[partition_idx]
    }

    pub fn get_elements_in_partition<'a>(&'a self, partition_idx: usize) -> &'a [usize] {
        let partition_start = self.partition_counts_prefix_sum[partition_idx];
        let partition_end = self.partition_counts_prefix_sum[partition_idx + 1];
        &self.partitions[partition_start..partition_end]
    }

    pub fn get_global_idx(&self, partition_idx: usize, local_idx: usize) -> usize {
        self.get_elements_in_partition(partition_idx)[local_idx]
    }

    pub fn get_local_idx(&self, global_idx: usize) -> (usize, usize) {
        (
            self.local_indices[global_idx].0.load(Ordering::Relaxed),
            self.local_indices[global_idx].1.load(Ordering::Relaxed),
        )
    }
}

fn init_centroids<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: &V,
    k: usize,
) -> Array2<T> {
    let num_points = access_method.num_points();
    let mut centroids: Array2<T> = Array2::zeros((k, access_method.dim()));

    let mut rng = rand::thread_rng();
    let mut row_id_for_cur_cent = Uniform::from(0..num_points).sample(&mut rng);
    let mut distances = vec![0.0; num_points];
    for mut c in centroids.outer_iter_mut() {
        c.as_slice_mut()
            .unwrap()
            .clone_from_slice(access_method.get_vec_sync(row_id_for_cur_cent));
        let total_distance: f64 = access_method
            .par_iter()
            .zip_eq(distances.par_iter_mut())
            .map(|(v, d)| {
                *d = D::calculate(c.as_slice().unwrap(), v);
                *d
            })
            .sum();
        if total_distance == 0.0 {
            break;
        }
        let mut sample = Uniform::new(0.0, total_distance).sample(&mut rng);
        row_id_for_cur_cent = 0;
        while sample > 0.0 {
            sample -= distances[row_id_for_cur_cent];
            row_id_for_cur_cent += 1;
        }
        row_id_for_cur_cent -= 1;
        assert!(row_id_for_cur_cent < num_points);
    }
    centroids
}

fn compute_closest_two_centroids<T: Indexable, D: Distance<T>>(
    p: &[T],
    centroids: &Array2<T>,
) -> (usize, usize) {
    let mut closest = (0, std::f64::INFINITY);
    let mut second_closest = (0, std::f64::INFINITY);
    for (i, c) in centroids.outer_iter().enumerate() {
        let dist = D::calculate(p, c.as_slice().unwrap());
        if dist < closest.1 {
            second_closest = closest;
            closest = (i, dist);
        } else if dist < second_closest.1 {
            second_closest = (i, dist);
        }
    }
    assert!(
        closest.0 != second_closest.0,
        "Point assigned to same partition twice"
    );
    (closest.0, second_closest.0)
}

fn update_single_centroid<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: &V,
    assignments: &Vec<(usize, usize)>,
    centroid_id: usize,
    centroid: &mut ArrayViewMut1<T>,
) -> (usize, usize) {
    let mut count_most_significant = 0;
    let mut count_any = 0;
    centroid.fill(T::zero());
    for (ivec, vec) in access_method.iter().enumerate() {
        let (a1, a2) = assignments[ivec];
        if a1 == centroid_id {
            centroid
                .iter_mut()
                .zip(vec.iter())
                .for_each(|(x, y)| *x += y.clone());
            count_most_significant += 1;
        }
        if a1 == centroid_id || a2 == centroid_id {
            assert!(
                a1 != centroid_id || a2 != centroid_id,
                "Point {} assigned to centroid {} twice!",
                ivec,
                centroid_id
            );
            count_any += 1;
        }
    }
    if count_most_significant != 0 {
        *centroid /= T::from_usize(count_most_significant).unwrap();
    }
    (count_any, count_most_significant)
}

fn update_assignments<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: &V,
    assignments: &mut KMeansAssignment,
    centroids: &Array2<T>,
) {
    assignments
        .assignments
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x = compute_closest_two_centroids::<T, D>(access_method.get_vec_sync(i), centroids)
        })
}

fn update_centroids<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: &V,
    assignments: &mut KMeansAssignment,
    centroids: &mut Array2<T>,
    histogram: &mut Vec<usize>,
) {
    centroids
        .outer_iter_mut()
        .into_par_iter()
        .zip(assignments.partition_counts_prefix_sum.par_iter_mut())
        .zip(histogram.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((mut x, count_any), count_most_significant))| {
            (*count_any, *count_most_significant) = update_single_centroid::<T, D, V>(
                &access_method,
                &assignments.assignments,
                i,
                &mut x,
            );
        });
}

fn estimate_evenness(histogram: &Vec<usize>, num_points: usize) -> f64 {
    let mut sum: usize = 0;
    let mut result: f64 = 0.0;
    let pdf: f64 = 1.0 / (histogram.len() as f64);
    for (i, c) in histogram.iter().enumerate() {
        sum += c;
        let actual = (sum as f64) / (num_points as f64);
        let expected = ((i + 1) as f64) * pdf;
        let diff = (actual - expected).abs();
        result = diff.max(result);
    }
    result
}

fn compute_local_indices(assignments: &mut KMeansAssignment, k: usize, num_points: usize) {
    let ptr = assignments.partitions.as_ptr() as usize;
    (0..k).into_par_iter().
        map(|i|
            {
                let start = assignments.partition_counts_prefix_sum[i];
                let end = assignments.partition_counts_prefix_sum[i + 1];
                let result = unsafe
                {
                    let slice = std::slice::from_raw_parts_mut(
                        (ptr as *mut usize).add(start),
                        end - start,
                    );
                    (i, slice)
                };
                result
            })
        .for_each(|(ipart, part)|
                  {
                      let mut ilocal = 0;
                      for iglobal in (0..num_points)
                      {
                          let (a1, a2) = assignments.get_partitions_for_element(iglobal);
                          if a1 == ipart
                          {
                              assignments.local_indices[iglobal].0.store(ilocal, Ordering::Relaxed);
                              part[ilocal] = iglobal;
                              ilocal += 1;
                          }
                          else if a2 == ipart
                          {
                              assignments.local_indices[iglobal].1.store(ilocal, Ordering::Relaxed);
                              part[ilocal] = iglobal;
                              ilocal += 1;
                          }
                          assert!(
                              ilocal <= assignments.num_points_in_partition(ipart),
                              "Found more points assigned to partition than expected: iglobal={} ipart={} ilocal={} assignments=({}, {})", iglobal, ipart, ilocal, a1, a2,
                          );
                      }
                  });
}

pub fn kmeans<T: Indexable, D: Distance<T>, V: VectorAccessMethod<T>>(
    access_method: &V,
    k: usize,
) -> KMeansAssignment {
    let num_points = access_method.num_points();

    let mut assignments = KMeansAssignment::new(k, num_points);
    let mut histogram = vec![0; k];

    let mut centroids = init_centroids::<T, D, V>(&access_method, k);
    loop {
        update_assignments::<T, D, V>(access_method, &mut assignments, &centroids);
        update_centroids::<T, D, V>(
            access_method,
            &mut assignments,
            &mut centroids,
            &mut histogram,
        );
        let evenness = estimate_evenness(&histogram, num_points);
        if evenness < 0.10 {
            break;
        }
    }
    let mut sum: usize = 0;
    for cnt in &mut assignments.partition_counts_prefix_sum {
        let element = *cnt;
        *cnt = sum;
        sum += element;
    }

    compute_local_indices(&mut assignments, k, num_points);
    assignments
}
