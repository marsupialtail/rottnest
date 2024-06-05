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

    return polars.from_arrow(pq.read_table(file_path, filesystem = get_fs_from_file_path(file_path)))

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

def index_file_bm25(file_path: str, column_name: str, name = None, index_mode = "physical", tokenizer_file = None):

    arr, uid, file_data = get_physical_layout(file_path, column_name) if index_mode == "physical" else get_virtual_layout(file_path, column_name, "uid")

    name = uuid.uuid4().hex if name is None else name
    file_data.write_parquet(f"{name}.meta")
    rottnest.build_lava_bm25(f"{name}.lava", arr, uid, tokenizer_file)

def index_file_substring(file_path: str, column_name: str, name = None, index_mode = "physical", tokenizer_file = None, token_skip_factor = None):

    arr, uid, file_data = get_physical_layout(file_path, column_name) if index_mode == "physical" else get_virtual_layout(file_path, column_name, "uid")

    name = uuid.uuid4().hex if name is None else name
    file_data.write_parquet(f"{name}.meta")
    rottnest.build_lava_substring(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64)), tokenizer_file, token_skip_factor)

def index_file_uuid(file_path: str, column_name: str, name = None, index_mode = "physical"):

    arr, uid, file_data = get_physical_layout(file_path, column_name) if index_mode == "physical" else get_virtual_layout(file_path, column_name, "uid")

    idx = pac.sort_indices(arr)
    arr = arr.take(idx)
    uid = uid.take(idx)

    name = uuid.uuid4().hex if name is None else name
    file_data.write_parquet(f"{name}.meta")
    rottnest.build_lava_uuid(f"{name}.lava", arr, uid)

def index_file_vector(file_path: str, column_name: str, name = None, gpu = False):

    try:
        import faiss
    except:
        print("Please install faiss")
        return

    arrs, layout = rottnest.get_parquet_layout(column_name, file_path)
    arr = pyarrow.concat_arrays([i.cast(pyarrow.large_binary()) for i in arrs])
    # convert arr into numpy
    arr = np.vstack([np.frombuffer(i, dtype = np.float32) for i in arr.to_pylist()])

    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) 

    x = np.cumsum(np.hstack([[0],layout.data_page_num_rows[:-1]]))
    y = np.repeat(x[np.cumsum(np.hstack([[0],layout.row_group_data_pages[:-1]])).astype(np.uint64)], layout.row_group_data_pages)
    page_row_offsets_in_row_group = x - y

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows)),
            "file_path": [file_path] * (len(data_page_num_rows)),
            "column_name": [column_name] * (len(data_page_num_rows)),
            "data_page_offsets": layout.data_page_offsets,
            "data_page_sizes":  layout.data_page_sizes,
            "dictionary_page_sizes": layout.dictionary_page_sizes,
            "row_groups": np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages),
            "page_row_offset_in_row_group": page_row_offsets_in_row_group,
            "data_page_rows": data_page_num_rows
        }
    )
    name = uuid.uuid4().hex if name is None else name
    file_data.write_parquet(f"{name}.meta")

    
    print(rottnest.build_lava_vector(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64))))

def merge_metadatas(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1
    # first read the metadata files and merge those
    metadatas = [read_metadata_file(f"{name}.meta")for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")
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

    assert len(index_names) > 1

    # first read the metadata files and merge those
    metadatas = [read_metadata_file(f"{index_name}.meta").with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, index_name in enumerate(index_names)]
    column_names = set(metadata["column_name"].unique()[0] for metadata in metadatas)
    assert len(column_names) == 1, "index is not allowed to span multiple column names"
    column_name = column_names.pop()

    vectors = [pyarrow.chunked_array(rottnest.read_indexed_pages(column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list()))
                for metadata in metadatas]
    

    vectors = [np.vstack([np.frombuffer(i, dtype = np.float32) for i in arr.to_pylist()]) for arr in vectors]

    import pdb;pdb.set_trace()

    rottnest.merge_lava_vector(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], vectors)
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")

def search_index_vector_mem(indices: List[str], arr: np.array, queries: List[np.array], K: int):
    
    index_search_results = rottnest.search_lava_vector_mem([f"{index_name}.lava" for index_name in indices], arr, queries, K)
    # print(index_search_results)
    return index_search_results

def search_index_vector(indices: List[str], query: np.array, K: int):
    
    metadatas = [read_metadata_file(f"{index_name}.meta").with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, index_name in enumerate(indices)]
    data_page_rows = [np.cumsum(np.hstack([[0] , np.array(metadata["data_page_rows"])])) for metadata in metadatas]
    uid_to_metadata = [[(a,b,c,d,e) for a,b,c,d,e in zip(metadata["file_path"], metadata["row_groups"], metadata["data_page_offsets"], 
                                                        metadata["data_page_sizes"], metadata["dictionary_page_sizes"])] for metadata in metadatas]
    
    metadata = polars.concat(metadatas)
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"
    column_name = metadata["column_name"].unique()[0]

    index_search_results, vectors = rottnest.search_lava_vector([f"{index_name}.lava" for index_name in indices], column_name, data_page_rows, uid_to_metadata, query, K)
    
    # import pdb; pdb.set_trace()
    return index_search_results
    # print(index_search_results)
    # print(vectors)
    
    # if len(index_search_results) == 0:
    #     return None

    # uids = polars.from_dict({"file_id": [i[0] for i in index_search_results], "uid": [i[1] for i in index_search_results]})

    # metadata = metadata.join(uids, on = ["file_id", "uid"])

    # result = pyarrow.chunked_array(rottnest.read_indexed_pages(column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
    #                                  metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list()))
    # result = pyarrow.table([result], names = ["text"])
    
    # return polars.from_arrow(result).filter(polars.col("text").str.to_lowercase().str.contains(query.lower()))

def get_result_from_index_result(indices: List[str], index_search_results: list):
    uids = polars.from_dict({"file_id": [i[0] for i in index_search_results], "uid": [i[1] for i in index_search_results]})
    metadatas = [read_metadata_file(f"{index_name}.meta").with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, index_name in enumerate(indices)]
    metadata = polars.concat(metadatas)
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

def search_index_uuid(indices: List[str], query: str, K: int):
    
    index_search_results = rottnest.search_lava_uuid([f"{index_name}.lava" for index_name in indices], query, K)
    print(index_search_results)

    if len(index_search_results) == 0:
        return None

    result, column_name = get_result_from_index_result(indices, index_search_results)
    result =  polars.from_arrow(result).filter(polars.col(column_name) == query)

    return result


def search_index_substring(indices: List[str], query: str, K: int):
    
    index_search_results = rottnest.search_lava_substring([f"{index_name}.lava" for index_name in indices], query, K)
    print(index_search_results)

    if len(index_search_results) == 0:
        return None

    result, column_name = get_result_from_index_result(indices, index_search_results)
    result =  polars.from_arrow(result).filter(polars.col(column_name).str.to_lowercase().str.contains(query.lower()))

    return result

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

    metadatas = [read_metadata_file(f"{index_name}.meta").with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, index_name in enumerate(indices)]
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