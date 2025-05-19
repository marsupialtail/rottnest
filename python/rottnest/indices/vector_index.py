from rottnest.indices.index_interface import RottnestIndex, CacheRanges
import rottnest.rottnest as rottnest
import pyarrow
import polars
from typing import List
import numpy as np
import pyarrow.parquet as pq
import json
import os

class VectorIndex(RottnestIndex):
    def __init__(self, index_mode: str = 'physical', data_type: str = 'binary', brute_force_threshold: int = 1000):
        super().__init__('vector', index_mode, data_type, brute_force_threshold)
    
    def brute_force(self, data: pyarrow.Table, column_name: str, query: str, K: int) -> pyarrow.Table:
        buffers = data[column_name].combine_chunks().buffers()

        if type(data[column_name][0]) == pyarrow.lib.BinaryScalar:
            offsets = np.frombuffer(buffers[1], dtype = np.uint32)
        elif type(data[column_name][0]) == pyarrow.lib.LargeBinaryScalar:
            offsets = np.frombuffer(buffers[1], dtype = np.uint64)
        diffs = np.unique(offsets[1:] - offsets[:-1])
        assert len(diffs) == 1, "vectors have different length!"
        dim = diffs.item() // 4
        vecs = np.frombuffer(buffers[2], dtype = np.float32).reshape(len(data), dim)
        results = np.linalg.norm(query - vecs, axis = 1).argsort()[:K]
        return data[results].to_arrow()

    def build_index(self, data_arr: pyarrow.Array, uid_arr: pyarrow.Array, index_name: str, gpu: bool = False) -> CacheRanges:
        try:
            import faiss
            from tqdm import tqdm
            import zstandard as zstd
        except:
            print("Please pip install faiss zstandard tqdm")
            return
        
        dtype_size = 4

        uid = uid_arr.to_numpy()

        # arr will be a array of largebinary, we need to convert it into numpy, time for some arrow ninja
        buffers = data_arr.buffers()
        offsets = np.frombuffer(buffers[1], dtype = np.uint64)
        diffs = np.unique(offsets[1:] - offsets[:-1])
        assert len(diffs) == 1, "vectors have different length!"
        dim = diffs.item() // dtype_size
        x = np.frombuffer(buffers[2], dtype = np.float32).reshape(len(data_arr), dim)

        num_centroids = len(data_arr) // 10_000

        kmeans = faiss.Kmeans(128,num_centroids, niter=30, verbose=True, gpu = gpu)
        kmeans.train(x)
        centroids = kmeans.centroids

        pqer = faiss.ProductQuantizer(dim, 32, 8)
        pqer.train(x)
        codes = pqer.compute_codes(x)

        batch_size = 10_000

        posting_lists = [[] for _ in range(num_centroids)]
        codes_lists = [[] for _ in range(num_centroids)]

        if gpu:

            res = faiss.StandardGpuResources()
            d = centroids.shape[1]
            index = faiss.GpuIndexFlatL2(res, d)
            index.add(centroids.astype('float32'))

            # Process batches
            for i in tqdm(range(len(data_arr) // batch_size)):
                batch = x[i * batch_size:(i + 1) * batch_size].astype('float32')
                k = 20 
                distances, indices = index.search(batch, k)
                
                # The indices are already sorted by distance, so we don't need to sort again
                closest_centroids = indices[:, 0]

                for k in range(batch_size):
                    # TODO: this uses UID! Just a warning. because gemv is fast even on lowly CPUs for final reranking.
                    posting_lists[closest_centroids[k]].append(uid[i * batch_size + k])
                    codes_lists[closest_centroids[k]].append(codes[i * batch_size + k])

        
        else:
            for i in tqdm(range(len(data_arr) // batch_size)):
                batch = x[i * batch_size:(i + 1) * batch_size]

                distances = -np.sum(centroids ** 2, axis=1, keepdims=True).T + 2 * np.dot(batch, centroids.T)
                indices = np.argpartition(-distances, kth=20, axis=1)[:, :20]
                sorted_indices = np.argsort(-distances[np.arange(distances.shape[0])[:, None], indices], axis=1)
                indices = indices[np.arange(indices.shape[0])[:, None], sorted_indices]     

                closest_centroids = list(indices[:,0])
                # closest2_centroids = list(indices[:,1])

                for k in range(batch_size):
                    # TODO: this uses UID! Just a warning. because gemv is fast even on lowly CPUs for final reranking.
                    posting_lists[closest_centroids[k]].append(uid[i * batch_size + k])
                    codes_lists[closest_centroids[k]].append(codes[i * batch_size + k])

        
        f = open(f"{index_name}.lava", "wb")
        centroid_offsets = [0]

        compressor = zstd.ZstdCompressor(level = 10)
        for i in range(len(posting_lists)):
            posting_lists[i] = np.array(posting_lists[i]).astype(np.uint32)
            codes_lists[i] = np.vstack(codes_lists[i]).reshape((-1))
            my_bytes = np.array([len(posting_lists[i])]).astype(np.uint32).tobytes()
            my_bytes += posting_lists[i].tobytes()
            my_bytes += codes_lists[i].tobytes()
            # compressed = compressor.compress(my_bytes)
            f.write(my_bytes)
            centroid_offsets.append(f.tell())

        # now time for the cacheable metadata page
        # pq_index, centroids, centroid_offsets

        cache_start = f.tell()

        offsets = [cache_start]
        faiss.write_ProductQuantizer(pqer, f"tmp.pq")
        # read the bytes back in 
        pq_index_bytes = open("tmp.pq", "rb").read()
        os.remove("tmp.pq")
        f.write(pq_index_bytes)
        offsets.append(f.tell())

        centroid_offset_bytes = compressor.compress(np.array(centroid_offsets).astype(np.uint64).tobytes())
        f.write(centroid_offset_bytes)
        offsets.append(f.tell())

        centroid_vectors_bytes = compressor.compress(centroids.astype(np.float32).tobytes())
        f.write(centroid_vectors_bytes)
        offsets.append(f.tell())

        f.write(np.array(offsets).astype(np.uint64).tobytes())

        cache_end = f.tell()

        return CacheRanges([(cache_start, cache_end)])

    def search_index(self, indices: List[str], query: str, K: int, nprobes: int = 50, refine: int = 50) -> List[tuple[int, int]]:
        import time
        try:
            import faiss
        except:
            print("Please pip install faiss")
            return
        
        # uids and codes are list of lists, where each sublist corresponds to an index. pq is a list of bytes
        # length is the same as the list of indices
        start = time.time()
        valid_file_ids, pq_bytes, arrs = rottnest.search_lava_vector([f"{index_name}.lava" for index_name in indices], query, nprobes, "aws")
        print("INDEX SEARCH TIME", time.time() - start)

        file_ids = []
        uids = []
        codes = []

        pqs = {}

        start = time.time()
        for i, pq_bytes in zip(valid_file_ids, pq_bytes):
            f = open("tmp.pq", "wb")
            f.write(pq_bytes.tobytes())
            pqs[i] = faiss.read_ProductQuantizer("tmp.pq")
            os.remove("tmp.pq")

        for (file_id, arr) in arrs:
            plist_length = np.frombuffer(arr[:4], dtype = np.uint32).item()
            plist = np.frombuffer(arr[4: plist_length * 4 + 4], dtype = np.uint32)
            this_codes = np.frombuffer(arr[plist_length * 4 + 4:], dtype = np.uint8).reshape((plist_length, -1))
            
            decoded = pqs[file_id].decode(this_codes)
            this_norms = np.linalg.norm(decoded - query, axis = 1).argsort()[:refine]
            codes.append(decoded[this_norms])
            uids.append(plist[this_norms])
            file_ids.append(np.ones(len(this_norms)) * file_id)
        
        file_ids = np.hstack(file_ids).astype(np.int64)
        uids = np.hstack(uids).astype(np.int64)
        codes = np.vstack(codes)
        fp_rerank = np.linalg.norm(query - codes, axis = 1).argsort()[:refine]

        print("PQ COMPUTE TIME", time.time() - start)

        file_ids = file_ids[fp_rerank]
        uids = uids[fp_rerank]

        # there could be redundancies here, since the uid is pointed to the page. two high ranked codes could be in the same page

        return list(set([(file_id, uid) for file_id, uid in zip(file_ids, uids)]))

        
    
    def compact_indices(self, new_index_name: str, indices: List[str], offsets: np.array):
        raise NotImplementedError("Compact indices logic not implemented")
