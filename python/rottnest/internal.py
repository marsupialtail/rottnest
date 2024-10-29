import pyarrow
import pyarrow.parquet as pq
from typing import List
import rottnest.rottnest as rottnest
from typing import List, Optional
import uuid
import polars
import numpy as np
import os
import pyarrow.compute as pac
import json
import daft
import multiprocessing
import time


from .nlp import query_expansion_keyword, query_expansion_llm
from .utils import get_daft_io_config_from_file_path, get_fs_from_file_path, get_physical_layout, get_virtual_layout, read_columns, read_metadata_file,\
    get_metadata_and_populate_cache, get_result_from_index_result, return_full_result


def index_files_logcloud(file_paths: List[str], column_name: str, name = uuid.uuid4().hex, remote = None, 
                         prefix_bytes = None, prefix_format = None, batch_files = 8, wavelet = False):
    from tqdm import tqdm
    num_groups = 0
    curr_max = 0
    file_paths = sorted(file_paths)
    for i in tqdm(range(0, len(file_paths),batch_files)):
        
        batch = file_paths[i:i+batch_files]
        data, uid, metadata = get_physical_layout(batch, column_name, remote = remote)
        uid = pac.cast(pac.add(uid, curr_max), pyarrow.uint64())
        metadata = metadata.with_columns([polars.col('uid') + curr_max])
        curr_max = pac.max(uid).as_py() + 1
       
        metadata.write_parquet(f'{num_groups}.maui')

        p = multiprocessing.Process(target=rottnest.compress_logs, args=(data, uid, name, num_groups, prefix_bytes, prefix_format))
        p.start()
        p.join() 
        num_groups += 1
    
    polars.concat([polars.read_parquet(f"{i}.maui") for i in range(num_groups)]).write_parquet(f"{name}.maui")
    rottnest.index_logcloud(name, num_groups, wavelet_tree = wavelet)

def index_files_bm25(file_paths: list[str], column_name: str, name = uuid.uuid4().hex, index_mode = "physical", tokenizer_file = None):

    arr, uid, file_data = get_physical_layout(file_paths, column_name) if index_mode == "physical" else get_virtual_layout(file_paths, column_name, "uid")

    cache_ranges = rottnest.build_lava_bm25(f"{name}.lava", arr, uid, tokenizer_file)

    # do not attempt to manually edit the metadata. It is Parquet, but it is Varsity Parquet to ensure performance.
    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')

def index_files_substring(file_paths: list[str], column_name: str, name = uuid.uuid4().hex, index_mode = "physical", tokenizer_file = None, token_skip_factor = None, remote = None, char_index = False):

    arr, uid, file_data = get_physical_layout(file_paths, column_name, remote = remote) if index_mode == "physical" else get_virtual_layout(file_paths, column_name, "uid", remote = remote)

    cache_ranges = rottnest.build_lava_substring(f"{name}.lava", arr, uid, tokenizer_file, token_skip_factor, char_index)

    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')

def index_files_uuid(file_paths: list[str], column_name: str, name = uuid.uuid4().hex, index_mode = "physical", remote = None):

    arr, uid, file_data = get_physical_layout(file_paths, column_name, remote = remote) if index_mode == "physical" else get_virtual_layout(file_paths, column_name, "uid", remote = remote)

    idx = pac.sort_indices(arr)
    arr = arr.take(idx)
    uid = uid.take(idx)

    cache_ranges = rottnest.build_lava_uuid(f"{name}.lava", arr, uid)
    
    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')



def index_files_vector(file_paths: list[str], column_name: str, name = uuid.uuid4().hex, dtype = 'f32', index_mode = "physical", gpu = False, remote = None):

    try:
        import faiss
        from tqdm import tqdm
        import zstandard as zstd
    except:
        print("Please pip install faiss zstandard tqdm")
        return
    
    assert dtype == 'f32'
    dtype_size = 4

    arr, uid, file_data = get_physical_layout(file_paths, column_name, type = "binary", remote = remote) if index_mode == "physical" else get_virtual_layout(file_paths, column_name, "uid", type = "binary", remote = remote)
    uid = uid.to_numpy()

    # arr will be a array of largebinary, we need to convert it into numpy, time for some arrow ninja
    buffers = arr.buffers()
    offsets = np.frombuffer(buffers[1], dtype = np.uint64)
    diffs = np.unique(offsets[1:] - offsets[:-1])
    assert len(diffs) == 1, "vectors have different length!"
    dim = diffs.item() // dtype_size
    x = np.frombuffer(buffers[2], dtype = np.float32).reshape(len(arr), dim)

    num_centroids = len(arr) // 10_000

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
        for i in tqdm(range(len(arr) // batch_size)):
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
        for i in tqdm(range(len(arr) // batch_size)):
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

    
    f = open(f"{name}.lava", "wb")
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

    file_data = file_data.to_arrow().replace_schema_metadata({"cache_ranges": json.dumps([(cache_start, cache_end)])})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')


def merge_metadatas(index_names: List[str], suffix = "meta"):
    assert len(index_names) > 1
    metadatas = daft.table.read_parquet_into_pyarrow_bulk([f"{index_name}.{suffix}" for index_name in index_names], io_config = get_daft_io_config_from_file_path(index_names[0]))
    # discard cache ranges in metadata, don't need them
    metadatas = [polars.from_arrow(i) for i in metadatas]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    return offsets, polars.concat(metadatas)

def merge_index_bm25(new_index_name: str, index_names: List[str]):
    
    offsets, file_data = merge_metadatas(index_names)
    
    cache_ranges = rottnest.merge_lava_generic(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets, 0)
    
    file_data = file_data.to_arrow().replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{new_index_name}.meta", write_statistics = False, compression = 'zstd')

def merge_index_substring(new_index_name: str, index_names: List[str]):
    
    offsets, file_data = merge_metadatas(index_names)
    
    cache_ranges = rottnest.merge_lava_generic(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets, 1)
    
    file_data = file_data.to_arrow().replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{new_index_name}.meta", write_statistics = False, compression = 'zstd')

def merge_index_uuid(new_index_name: str, index_names: List[str]):
    
    offsets, file_data = merge_metadatas(index_names)
    
    cache_ranges = rottnest.merge_lava_generic(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets, 2)
    
    file_data = file_data.to_arrow().replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{new_index_name}.meta", write_statistics = False, compression = 'zstd')

def merge_index_vector(new_index_name: str, index_names: List[str]):

    try:
        import faiss
        import zstandard as zstd
    except:
        print("faiss zstandard not installed")
        return

    raise NotImplementedError

    def read_range(file_handle, start, end):
        file_handle.seek(start)
        return file_handle.read(end - start)

    offsets = merge_metadatas(new_index_name, index_names)

    # assume things are on disk for now
    assert len(index_names) == 2

    index1 = open(f"{index_names[0]}.lava", "rb")
    index2 = open(f"{index_names[1]}.lava", "rb")
    output = open(f"{new_index_name}.lava", "wb")

    # get the length of index1
    index1_size = index1.seek(0,2)
    index2_size = index2.seek(0,2)
    offsets1 = read_range(index1, index1_size - 8 * 4, index1_size)
    offsets2 = read_range(index2, index2_size - 8 * 4, index2_size)

    output.write(read_range(index1, 0, offsets1[0]))
    unprocessed = read_range(index2, 0, offsets2[0])
    while unprocessed:
        l = np.frombuffer(unprocessed[:4], np.uint32).item()


    decompressor = zstd.ZstdDecompressor()

def search_index_uuid(indices: List[str], query: str, K: int, columns = []):

    metadata = get_metadata_and_populate_cache(indices)
    
    index_search_results = rottnest.search_lava_uuid([f"{index_name}.lava" for index_name in indices], query, K, "aws")
    print(index_search_results)

    if len(index_search_results) == 0:
        return None

    result, column_name, metadata = get_result_from_index_result(metadata, index_search_results)
    result =  polars.from_arrow(result).filter(polars.col(column_name) == query)

    return return_full_result(result, metadata, column_name, columns)


def search_index_logcloud(indices: List[str], query: str, K: int, columns = [], wavelet_tree = False, exact = False):

    start = time.time()
    metadata = get_metadata_and_populate_cache(indices, suffix="maui")
    index_search_results = rottnest.search_logcloud(indices, query, K, None, wavelet_tree, exact)
    print("INDEX SEARCH TIME", time.time() - start)
    # print(index_search_results)

    flag, index_search_results = index_search_results

    if flag == 1:
        if len(index_search_results) == 0:
            return None
        index_search_results = index_search_results[:K]
        
        start = time.time()
        result, column_name, metadata = get_result_from_index_result(metadata, index_search_results)
        result =  polars.from_arrow(result).filter(polars.col(column_name).str.contains(query))
        result = return_full_result(result, metadata, column_name, columns)
        print("PARQUET LOAD TIME", time.time() - start)
        return result
    elif flag == 0:
        reversed_filenames = sorted(metadata['file_path'].unique().to_list())[::-1]
        results = []
        start_time = time.time()
        # you should search in reverse order in batches of 10
        for start in range(0, len(reversed_filenames), 10):
            batch = reversed_filenames[start:start+10]
            a = daft.daft.read_parquet_into_pyarrow_bulk(batch)
            df = polars.DataFrame([polars.from_arrow(pyarrow.concat_arrays([
                pac.cast(pyarrow.concat_arrays(k[2][0]), pyarrow.large_string()) for k in a]))])
            result = df.filter(polars.col('column_0').str.contains(query))
            if len(result) > 0:
                results.append(result)
                if sum([len(r) for r in results]) > K:
                    break

        print("PARQUET LOAD TIME", time.time() - start_time)
        if len(results) > 0:
            return polars.concat(results)
        else:
            return None



def search_index_substring(indices: List[str], query: str, K: int, sample_factor = None, token_viable_limit = 10, columns = [], char_index = False):

    metadata = get_metadata_and_populate_cache(indices)
    
    index_search_results = rottnest.search_lava_substring([f"{index_name}.lava" for index_name in indices], query, K, "aws", sample_factor = sample_factor, token_viable_limit = token_viable_limit, char_index = char_index)
    print(index_search_results)

    if len(index_search_results) == 0:
        return None
    
    if len(index_search_results) > 10000:
        return "Brute Force Please"

    result, column_name, metadata = get_result_from_index_result(metadata, index_search_results)
    result =  polars.from_arrow(result).filter(polars.col(column_name).str.to_lowercase().str.contains(query.lower(), literal=True))

    return return_full_result(result, metadata, column_name, columns)

def search_index_vector(indices: List[str], query: np.array, K: int, columns = [], nprobes = 500, refine = 500):

    import time
    try:
        import faiss
    except:
        print("Please pip install faiss")
        return

    metadata = get_metadata_and_populate_cache(indices)
    
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

    index_search_results = list(set([(file_id, uid) for file_id, uid in zip(file_ids, uids)]))

    print(index_search_results)

    start = time.time()
    result, column_name, metadata = get_result_from_index_result(metadata, index_search_results)
    print("RESULT TIME", time.time() - start)

    buffers = result[column_name].combine_chunks().buffers()

    if type(result[column_name][0]) == pyarrow.lib.BinaryScalar:
        offsets = np.frombuffer(buffers[1], dtype = np.uint32)
    elif type(result[column_name][0]) == pyarrow.lib.LargeBinaryScalar:
        offsets = np.frombuffer(buffers[1], dtype = np.uint64)
    diffs = np.unique(offsets[1:] - offsets[:-1])
    assert len(diffs) == 1, "vectors have different length!"
    dim = diffs.item() // 4
    vecs = np.frombuffer(buffers[2], dtype = np.float32).reshape(len(result), dim)
    results = np.linalg.norm(query - vecs, axis = 1).argsort()[:K]
    result = polars.from_arrow(result)[results]

    return return_full_result(result, metadata, column_name, columns)

    
def search_index_bm25(indices: List[str], query: str, K: int, query_expansion = "bge", quality_factor = 0.2, expansion_tokens = 20, cache_dir = None, reader_type = None):

    assert query_expansion in {"bge", "openai", "keyword", "none"}
    
    tokenizer_vocab = rottnest.get_tokenizer_vocab([f"{index_name}.lava" for index_name in indices])

    if query_expansion in {"bge","openai"}:
        tokens, token_ids, weights = query_expansion_llm(tokenizer_vocab, query, method = query_expansion, expansion_tokens=expansion_tokens, cache_dir = cache_dir)
    elif query_expansion == "keyword":
        tokens, token_ids, weights = query_expansion_keyword(tokenizer_vocab, query)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file("../tok/tokenizer.json")
        token_ids = tok.encode(query).ids
        tokens = [tokenizer_vocab[i] for i in token_ids]
        weights = [1] * len(token_ids)
        print(tokens)

    # metadata_file = f"{index_name}.meta"
    index_search_results = rottnest.search_lava_bm25([f"{index_name}.lava" for index_name in indices], token_ids, weights, int(K * quality_factor), reader_type = reader_type)
    
    if len(index_search_results) == 0:
        return None
    
    uids = polars.from_dict({"file_id": [i[0] for i in index_search_results], "uid": [i[1] for i in index_search_results]})

    metadatas = [read_metadata_file(f"{index_name}.meta")[0].with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, index_name in enumerate(indices)]
    metadata = polars.concat(metadatas)
    metadata = metadata.join(uids, on = ["file_id", "uid"])
    
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"
    column_name = metadata["column_name"].unique()[0]

    result = pyarrow.chunked_array(rottnest.read_indexed_pages(column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list()))
    result = pyarrow.table([result], names = ["text"])
    result = result.append_column('row_nr', pyarrow.array(np.arange(len(result)), pyarrow.int64()))

    import duckdb
    con = duckdb.connect()
    con.register('test_table', result)
    con.execute("CREATE TABLE table_copy AS (SELECT * FROM test_table)")

    con.execute("""
    PRAGMA create_fts_index(
        'table_copy', 'row_nr', 'text'
    );
    """)

    result = polars.from_arrow(con.execute(f"""
        SELECT row_nr, text, score
        FROM (
            SELECT *, fts_main_table_copy.match_bm25(
                    row_nr,
                    '{" ".join(tokens)}',
                    fields := 'text'
                ) AS score
                FROM table_copy
        ) sq
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT {K};
        """).arrow())

    return result