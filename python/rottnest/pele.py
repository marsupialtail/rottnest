import pyarrow
import pyarrow.parquet as pq
from typing import List
import rottnest.rottnest as rottnest
from typing import List, Optional
import uuid
import polars
import numpy as np
import boto3
from botocore.config import Config
import os
import pyarrow.compute as pac
from pyarrow.fs import S3FileSystem, LocalFileSystem
from .nlp import query_expansion_keyword, query_expansion_llm
import json

def get_fs_from_file_path(filepath):

    if filepath.startswith("s3://"):
        if os.getenv('AWS_VIRTUAL_HOST_STYLE'):
            try:
                s3fs = S3FileSystem(endpoint_override = os.getenv('AWS_ENDPOINT_URL'), force_virtual_addressing = True )
            except:
                raise ValueError("Requires pyarrow >= 16.0.0 for virtual addressing.")
        else:
            s3fs = S3FileSystem(endpoint_override = os.getenv('AWS_ENDPOINT_URL'))
    else:
        s3fs = LocalFileSystem()

    return s3fs

def read_metadata_file(file_path: str):

    table = pq.read_table(file_path.lstrip("s3://"), filesystem = get_fs_from_file_path(file_path))

    try:
        cache_ranges = json.loads(table.schema.metadata[b'cache_ranges'].decode())
    except:
        cache_ranges = []

    return polars.from_arrow(table), cache_ranges

def read_columns(file_path: str, ):

    fs = get_fs_from_file_path(file_path)
    return pq.read_table(file_path, filesystem=fs)

def get_physical_layout(file_path: str, column_name: str, type = "str"):

    assert type in {"str", "binary"}
    arrs, layout = rottnest.get_parquet_layout(column_name, file_path)
    arr = pyarrow.concat_arrays([i.cast(pyarrow.large_string() if type == 'str' else pyarrow.large_binary()) for i in arrs])
    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) + 1

    # Code tries to compute the starting row offset of each page in its row group.
    # The following three lines are definitely easier to read than to write.

    x = np.cumsum(np.hstack([[0],layout.data_page_num_rows[:-1]]))
    y = np.repeat(x[np.cumsum(np.hstack([[0],layout.row_group_data_pages[:-1]])).astype(np.uint64)], layout.row_group_data_pages)
    page_row_offsets_in_row_group = x - y

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows) + 1),
            "file_path": [file_path] * (len(data_page_num_rows) + 1),
            "column_name": [column_name] * (len(data_page_num_rows) + 1),
            # TODO: figure out a better way to handle this. Currently this is definitely not a bottleneck. Write ampl factor is almost 10x
            # writing just one row followed by a bunch of Nones don't help, likely because it's already smart enough to do dict encoding.
            # but we should probably still do this to save memory once loaded in!
            "metadata_bytes": [layout.metadata_bytes]  + [None] * (len(data_page_num_rows)),
            "data_page_offsets": [-1] + layout.data_page_offsets,
            "data_page_sizes": [-1] + layout.data_page_sizes,
            "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
            "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
            "page_row_offset_in_row_group": np.hstack([[-1], page_row_offsets_in_row_group])
        }
    )

    return arr, pyarrow.array(uid.astype(np.uint64)), file_data

def get_virtual_layout(file_path: str, column_name: str, key_column_name: str, type = "str", stride = 500):

    fs = get_fs_from_file_path(file_path)
    table = pq.read_table(file_path, filesystem=fs, columns = [key_column_name, column_name])
    table = table.with_row_count('__row_count__').with_columns((polars.col('__row_count__') // stride).alias('__uid__'))

    arr = table[column_name].to_arrow().cast(pyarrow.large_string() if type == 'str' else pyarrow.large_binary())
    uid = table['__uid__'].to_arrow().cast(pyarrow.uint64())

    metadata = table.groupby("__uid__").agg([polars.col(key_column_name).min().alias("min"), polars.col(key_column_name).max().alias("max")]).sort("__uid__")
    return arr, uid, metadata

def index_file_bm25(file_path: str, column_name: str, name = uuid.uuid4().hex, index_mode = "physical", tokenizer_file = None):

    arr, uid, file_data = get_physical_layout(file_path, column_name) if index_mode == "physical" else get_virtual_layout(file_path, column_name, "uid")

    cache_ranges = rottnest.build_lava_bm25(f"{name}.lava", arr, uid, tokenizer_file)

    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')

def index_file_substring(file_path: str, column_name: str, name = uuid.uuid4().hex, index_mode = "physical", tokenizer_file = None, token_skip_factor = None):

    arr, uid, file_data = get_physical_layout(file_path, column_name) if index_mode == "physical" else get_virtual_layout(file_path, column_name, "uid")

    cache_ranges = rottnest.build_lava_substring(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64)), tokenizer_file, token_skip_factor)

    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')

def index_file_uuid(file_path: str, column_name: str, name = uuid.uuid4().hex, index_mode = "physical"):

    arr, uid, file_data = get_physical_layout(file_path, column_name) if index_mode == "physical" else get_virtual_layout(file_path, column_name, "uid")

    idx = pac.sort_indices(arr)
    arr = arr.take(idx)
    uid = uid.take(idx)

    cache_ranges = rottnest.build_lava_uuid(f"{name}.lava", arr, uid)
    
    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges)})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')

def index_file_vector(file_path: str, column_name: str, name = uuid.uuid4().hex, dtype = 'f32', index_mode = "physical", gpu = False):

    try:
        import faiss
        from tqdm import tqdm
        import zstandard as zstd
    except:
        print("Please pip install faiss zstandard tqdm")
        return
    
    assert dtype == 'f32'
    dtype_size = 4

    arr, uid, file_data = get_physical_layout(file_path, column_name, type = "binary") if index_mode == "physical" else get_virtual_layout(file_path, column_name, "uid", type = "binary")
    uid = uid.to_numpy()

    # arr will be a array of largebinary, we need to convert it into numpy, time for some arrow ninja
    buffers = arr.buffers()
    offsets = np.frombuffer(buffers[1], dtype = np.uint64)
    diffs = np.unique(offsets[1:] - offsets[:-1])
    assert len(diffs) == 1, "vectors have different length!"
    dim = diffs.item() // dtype_size
    x = np.frombuffer(buffers[2], dtype = np.float32).reshape(len(arr), dim)

    kmeans = faiss.Kmeans(128, len(arr) // 10_000, niter=30, verbose=True, gpu = gpu)
    kmeans.train(x)
    centroids = kmeans.centroids

    pqer = faiss.ProductQuantizer(dim, 32, 8)
    pqer.train(x)
    codes = pqer.compute_codes(x)

    batch_size = 10_000

    posting_lists = [[] for _ in range(len(arr) // 10_000)]
    codes_lists = [[] for _ in range(len(arr) // 10_000)]

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

    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps([(cache_start, cache_end)])})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')


def merge_metadatas(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1
    # first read the metadata files and merge those
    # discard the cache ranges
    metadatas = [read_metadata_file(f"{name}.meta")[0] for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta", statistics=False)
    return offsets

def merge_index_bm25(new_index_name: str, index_names: List[str]):
    
    offsets = merge_metadatas(new_index_name, index_names)
    rottnest.merge_lava_bm25(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)

def merge_index_uuid(new_index_name: str, index_names: List[str]):
    
    offsets = merge_metadatas(new_index_name, index_names)
    rottnest.merge_lava_uuid(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)

def merge_index_substring(new_index_name: str, index_names: List[str]):
    
    offsets = merge_metadatas(new_index_name, index_names)
    rottnest.merge_lava_substring(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)

def merge_index_vector(new_index_name: str, index_names: List[str]):

    offsets = merge_metadatas(new_index_name, index_names)

def get_result_from_index_result(metadata: polars.DataFrame, index_search_results: list):
    uids = polars.from_dict({"file_id": [i[0] for i in index_search_results], "uid": [i[1] for i in index_search_results]})

    file_metadatas = metadata.filter(polars.col("metadata_bytes").is_not_null()).group_by("file_path").first().select(["file_path", "metadata_bytes"])
    metadata = metadata.join(uids, on = ["file_id", "uid"])
    
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"
    column_name = metadata["column_name"].unique()[0]

    file_metadatas = {d["file_path"]: d["metadata_bytes"] for d in file_metadatas.to_dicts()}

    result = pyarrow.chunked_array(rottnest.read_indexed_pages(column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list(),
                                     "aws", file_metadatas))
    result = pyarrow.table([result], names = [column_name])
    return result, column_name

def get_metadata_and_populate_cache(indices: List[str]):
    metadatas = [read_metadata_file(f"{index_name}.meta") for i, index_name in enumerate(indices)]
    metadata = polars.concat([f[0].with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, f in enumerate(metadatas)])
    cache_dir = os.getenv("ROTTNEST_CACHE_DIR")
    if cache_dir:
        cache_ranges = {f"{indices[i]}.lava": f[1] for i, f in enumerate(metadatas) if len(f[1]) > 0}
        cached_files = list(cache_ranges.keys())
        ranges = [[tuple(k) for k in cache_ranges[f]] for f in cached_files]
        rottnest.populate_cache(cached_files, ranges, cache_dir, "aws")

    return metadata

def search_index_uuid(indices: List[str], query: str, K: int):

    metadata = get_metadata_and_populate_cache(indices)
    
    index_search_results = rottnest.search_lava_uuid([f"{index_name}.lava" for index_name in indices], query, K)
    print(index_search_results)

    if len(index_search_results) == 0:
        return None

    result, column_name = get_result_from_index_result(metadata, index_search_results)
    result =  polars.from_arrow(result).filter(polars.col(column_name) == query)

    return result


def search_index_substring(indices: List[str], query: str, K: int):

    metadata = get_metadata_and_populate_cache(indices)
    
    index_search_results = rottnest.search_lava_substring([f"{index_name}.lava" for index_name in indices], query, K)
    print(index_search_results)

    if len(index_search_results) == 0:
        return None

    result, column_name = get_result_from_index_result(metadata, index_search_results)
    result =  polars.from_arrow(result).filter(polars.col(column_name).str.to_lowercase().str.contains(query.lower()))

    return result

def search_index_vector(indices: List[str], query: np.array, K: int):

    pass

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