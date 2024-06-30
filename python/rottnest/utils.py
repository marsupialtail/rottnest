import pyarrow
import pyarrow.parquet as pq
import rottnest.rottnest as rottnest
from typing import List, Optional
import polars
import numpy as np
from botocore.config import Config
import os
from pyarrow.fs import S3FileSystem, LocalFileSystem
import json
import daft
from concurrent.futures import ThreadPoolExecutor

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

def get_daft_io_config_from_file_path(filepath):
    
    if filepath.startswith("s3://"):
        fs = daft.io.IOConfig(s3 = daft.io.S3Config(force_virtual_addressing = (True if os.getenv('AWS_VIRTUAL_HOST_STYLE') else False), endpoint_url = os.getenv('AWS_ENDPOINT_URL')))
    else:
        fs = daft.io.IOConfig()

    return fs

def read_metadata_file(file_path: str):

    table = pq.read_table(file_path.replace("s3://",''), filesystem = get_fs_from_file_path(file_path))

    try:
        cache_ranges = json.loads(table.schema.metadata[b'cache_ranges'].decode())
    except:
        cache_ranges = []

    return polars.from_arrow(table), cache_ranges

def read_columns(file_paths: list, row_groups: list, row_nr: list):

    def read_parquet_file(file, row_group, row_nr):
        f = pq.ParquetFile(file.replace("s3://",''), filesystem=get_fs_from_file_path(file))
        return f.read_row_group(row_group, columns=['id']).take(row_nr)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:  # Control the number of parallel threads
        results = list(executor.map(read_parquet_file, file_paths, row_groups, row_nr))
    
    return pyarrow.concat_tables(results)


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

def get_metadata_and_populate_cache(indices: List[str]):
    
    metadatas = daft.table.read_parquet_into_pyarrow_bulk([f"{index_name}.meta" for index_name in indices], io_config = get_daft_io_config_from_file_path(indices[0]))
    metadatas = [(polars.from_arrow(i), json.loads(i.schema.metadata[b'cache_ranges'].decode())) for i in metadatas]

    metadata = polars.concat([f[0].with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, f in enumerate(metadatas)])
    if os.getenv("CACHE_ENABLE").lower() == "true":
        cache_ranges = {f"{indices[i]}.lava": f[1] for i, f in enumerate(metadatas) if len(f[1]) > 0}
        cached_files = list(cache_ranges.keys())
        ranges = [[tuple(k) for k in cache_ranges[f]] for f in cached_files]
        rottnest.populate_cache(cached_files, ranges,  "aws")

    return metadata

def get_result_from_index_result(metadata: polars.DataFrame, index_search_results: list):
    
    uids = polars.from_dict({"file_id": [i[0] for i in index_search_results], "uid": [i[1] for i in index_search_results]})
    file_metadatas = metadata.filter(polars.col("metadata_bytes").is_not_null()).group_by("file_path").first().select(["file_path", "metadata_bytes"])
    metadata = metadata.join(uids, on = ["file_id", "uid"])
    
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"
    column_name = metadata["column_name"].unique()[0]

    file_metadatas = {d["file_path"]: d["metadata_bytes"] for d in file_metadatas.to_dicts()}

    result = rottnest.read_indexed_pages(column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list(),
                                     "aws", file_metadatas)

    row_group_rownr = [pyarrow.array(np.arange(metadata['page_row_offset_in_row_group'][i], metadata['page_row_offset_in_row_group'][i] + len(arr))) for i, arr in enumerate(result)]
    
    metadata_key = [pyarrow.array(np.ones(len(arr)).astype(np.uint32) * i) for i, arr in enumerate(result)]
    
    result = pyarrow.table([pyarrow.chunked_array(result), pyarrow.chunked_array(row_group_rownr), 
                            pyarrow.chunked_array(metadata_key)], names = [column_name, '__row_group_rownr__', '__metadata_key__'])
    
    return result, column_name, metadata.with_row_count('__metadata_key__')

def return_full_result(result: polars.DataFrame, metadata: polars.DataFrame, column_name: str, columns: List[str]):
    if columns != []:
        result = result.join(metadata.select(["__metadata_key__", "file_path", "row_groups"]), on = "__metadata_key__", how = "left")
        grouped = result.groupby(["file_path", "row_groups"]).agg([polars.col('__metadata_key__'), polars.col('__row_group_rownr__')])
        collected_results = polars.from_arrow(read_columns(grouped["file_path"].to_list(), grouped["row_groups"].to_list(), grouped["__row_group_rownr__"].to_list()))
        unnested_metadata_key = grouped['__metadata_key__'].explode()
        unnested_row_group_rownr = grouped['__row_group_rownr__'].explode()
        collected_results = collected_results.with_columns([unnested_metadata_key, unnested_row_group_rownr])
        result = result.join(collected_results, on = ["__metadata_key__", "__row_group_rownr__"], how = "left")
        return result.select(columns + [column_name])
    else:
        return result.select([column_name])