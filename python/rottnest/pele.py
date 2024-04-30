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
from pyarrow.fs import S3FileSystem
from .nlp import query_expansion_keyword, query_expansion_llm

def read_metadata_file(file_path: str):

    # currently only support aws and s3 compatible things, this wrapper is temporary, eventually move 
    # entirely to Rust

    if file_path.startswith("s3://"):
        style = os.getenv("AWS_VIRTUAL_HOST_STYLE")
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        if style:
            s3 = boto3.client('s3', config = Config(s3 = {'addressing_style': 'virtual'}), endpoint_url = endpoint_url if endpoint_url else None)
        else:
            s3 = boto3.client('s3', endpoint_url = endpoint_url if endpoint_url else None)

        obj = s3.get_object(Bucket=file_path.split("/")[2], Key="/".join(file_path.split("/")[3:]))
        return polars.read_parquet(obj['Body'].read())
    else:
        return polars.read_parquet(file_path)

def read_columns():

    try:
        s3fs = S3FileSystem(endpoint_override = 'https://tos-s3-cn-beijing.volces.com', force_virtual_addressing = True)
    except:
        raise ValueError("Requires pyarrow >= 16.0.0.")

def index_file_bm25(file_path: str, column_name: str, name = None, tokenizer_file = None):

    arrs, layout = rottnest.get_parquet_layout(column_name, file_path)
    arr = pyarrow.concat_arrays([i.cast(pyarrow.large_string()) for i in arrs])
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
            "data_page_offsets": [-1] + layout.data_page_offsets,
            "data_page_sizes": [-1] + layout.data_page_sizes,
            "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
            "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
            "page_row_offset_in_row_group": np.hstack([[-1], page_row_offsets_in_row_group])
        }
    )

    name = uuid.uuid4().hex if name is None else name

    file_data.write_parquet(f"{name}.meta")
    print(rottnest.build_lava_bm25(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64)), tokenizer_file))

def index_file_substring(file_path: str, column_name: str, name = None, tokenizer_file = None):

    arrs, layout = rottnest.get_parquet_layout(column_name, file_path)
    arr = pyarrow.concat_arrays([i.cast(pyarrow.large_string()) for i in arrs])
    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) + 1

    x = np.cumsum(np.hstack([[0],layout.data_page_num_rows[:-1]]))
    y = np.repeat(x[np.cumsum(np.hstack([[0],layout.row_group_data_pages[:-1]])).astype(np.uint64)], layout.row_group_data_pages)
    page_row_offsets_in_row_group = x - y

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows) + 1),
            "file_path": [file_path] * (len(data_page_num_rows) + 1),
            "column_name": [column_name] * (len(data_page_num_rows) + 1),
            "data_page_offsets": [-1] + layout.data_page_offsets,
            "data_page_sizes": [-1] + layout.data_page_sizes,
            "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
            "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
            "page_row_offset_in_row_group": np.hstack([[-1], page_row_offsets_in_row_group])
        }
    )
    name = uuid.uuid4().hex if name is None else name

    file_data.write_parquet(f"{name}.meta")
    print(rottnest.build_lava_substring(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64)), tokenizer_file))


def index_file_kmer(file_path: str, column_name: str, name = None, tokenizer_file = None):

    arrs, layout = rottnest.get_parquet_layout(column_name, file_path)
    arr = pyarrow.concat_arrays([i.cast(pyarrow.large_string()) for i in arrs])
    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) + 1

    x = np.cumsum(np.hstack([[0],layout.data_page_num_rows[:-1]]))
    y = np.repeat(x[np.cumsum(np.hstack([[0],layout.row_group_data_pages[:-1]])).astype(np.uint64)], layout.row_group_data_pages)
    page_row_offsets_in_row_group = x - y

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows) + 1),
            "file_path": [file_path] * (len(data_page_num_rows) + 1),
            "column_name": [column_name] * (len(data_page_num_rows) + 1),
            "data_page_offsets": [-1] + layout.data_page_offsets,
            "data_page_sizes": [-1] + layout.data_page_sizes,
            "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
            "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
            "page_row_offset_in_row_group": np.hstack([[-1], page_row_offsets_in_row_group])
        }
    )
    name = uuid.uuid4().hex if name is None else name

    file_data.write_parquet(f"{name}.meta")
    print(rottnest.build_lava_kmer(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64)), tokenizer_file))

def index_file_vector(file_path: str, column_name: str, name = None):

    arr, layout = rottnest.get_parquet_layout(column_name, file_path)
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

    # convert arr into numpy
    arr = np.vstack([np.frombuffer(i, dtype = np.float32) for i in arr.to_pylist()])
    print(rottnest.build_lava_vector(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64))))

def merge_index_bm25(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1

    # first read the metadata files and merge those
    metadatas = [read_metadata_file(f"{name}.meta")for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    rottnest.merge_lava_bm25(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")

def merge_index_substring(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1

    # first read the metadata files and merge those
    metadatas = [read_metadata_file(f"{name}.meta")for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    print(offsets)
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    rottnest.merge_lava_substring(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")

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

def search_index_substring(indices: List[str], query: str, K: int):
    
    index_search_results = rottnest.search_lava_substring([f"{index_name}.lava" for index_name in indices], query, K)
    print(index_search_results)

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
    
    return polars.from_arrow(result).filter(polars.col("text").str.to_lowercase().str.contains(query.lower()))

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