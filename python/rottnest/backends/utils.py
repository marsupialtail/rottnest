import pyarrow
import pyarrow.parquet as pq
import rottnest.rottnest as rottnest
from typing import List, Optional, Tuple
import polars
import numpy as np
from botocore.config import Config
import os
from pyarrow.fs import S3FileSystem, LocalFileSystem
import json
from concurrent.futures import ThreadPoolExecutor
from rottnest.indices.index_interface import RottnestIndex
import uuid

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

def read_row_groups(file_paths: list, row_groups: list, row_ranges: list, column: str):

    def read_parquet_file(file, row_group, row_range):
        f = pq.ParquetFile(file.replace("s3://",''), filesystem=get_fs_from_file_path(file))
        return f.read_row_group(row_group, columns=[column])[column].combine_chunks()[row_range[0]:row_range[1]]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:  # Control the number of parallel threads
        results = list(executor.map(read_parquet_file, file_paths, row_groups, row_ranges))

    return results

def get_physical_layout(file_paths: list, column_name: str, type = "str", remote = None):

    assert type in {"str", "binary"}

    metadatas = []
    all_arrs = []
    all_uids = []
    for file_path in file_paths:
        arrs, layout = rottnest.get_parquet_layout(column_name, file_path)
        arr = pyarrow.concat_arrays([i.cast(pyarrow.large_string() if type == 'str' else pyarrow.large_binary()) for i in arrs])
        data_page_num_rows = np.array(layout.data_page_num_rows)
        uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) + 1

        # Code tries to compute the starting row offset of each page in its row group.
        # The following three lines are definitely easier to read than to write.

        x = np.cumsum(np.hstack([[0],layout.data_page_num_rows[:-1]]))
        y = np.repeat(x[np.cumsum(np.hstack([[0],layout.row_group_data_pages[:-1]])).astype(np.uint64)], layout.row_group_data_pages)
        page_row_offsets_in_row_group = x - y

        metadata = polars.from_dict({
                "uid": np.arange(len(data_page_num_rows) + 1),
                "file_path": [file_path if remote is None else remote + file_path] * (len(data_page_num_rows) + 1),
                "column_name": [column_name] * (len(data_page_num_rows) + 1),
                # TODO: figure out a better way to handle this. Currently this is definitely not a bottleneck. Write ampl factor is almost 10x
                # writing just one row followed by a bunch of Nones don't help, likely because it's already smart enough to do dict encoding.
                # but we should probably still do this to save memory once loaded in!
                "metadata_bytes": [layout.metadata_bytes]  + [None] * (len(data_page_num_rows)),
                "data_page_offsets": [-1] + layout.data_page_offsets,
                "data_page_sizes": [-1] + layout.data_page_sizes,
                "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
                "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
                "page_row_offset_in_row_group": np.hstack([[-1], page_row_offsets_in_row_group]).astype(np.int64)
            } # type: ignore
        )

        metadatas.append(metadata)
        all_arrs.append(arr)
        all_uids.append(uid)
    
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    all_uids = np.hstack([uid + offsets[i] for i, uid in enumerate(all_uids)])

    return pyarrow.concat_arrays(all_arrs), pyarrow.array(all_uids.astype(np.uint64)), polars.concat(metadatas)

def get_virtual_layout(file_paths: list, column_name: str, key_column_name: str, type = "str", stride = 500, remote = None):

    fs = get_fs_from_file_path(file_paths[0])
    metadatas = []
    all_arrs = []
    all_uids = []

    for file_path in file_paths:
        table = pq.read_table(file_path, filesystem=fs, columns = [key_column_name, column_name])
        table = table.with_row_count('__row_count__').with_columns((polars.col('__row_count__') // stride).alias('__uid__'))

        arr = table[column_name].to_arrow().cast(pyarrow.large_string() if type == 'str' else pyarrow.large_binary())
        uid = table['__uid__'].to_arrow().cast(pyarrow.uint64())

        metadata = table.group_by("__uid__").agg([polars.col(key_column_name).min().alias("min"), polars.col(key_column_name).max().alias("max")]).sort("__uid__")
        
    return arr, uid, metadata

def get_metadata_and_populate_cache(indices: List[str], suffix = "meta"):
    
    # Read parquet files using PyArrow instead of Daft
    fs = get_fs_from_file_path(indices[0])
    metadatas = [pq.read_table(f"{index_name}.{suffix}".replace("s3://", ""), filesystem=fs) for index_name in indices]
    
    if os.getenv("CACHE_ENABLE") and os.getenv("CACHE_ENABLE").lower() == "true":
        metadatas = [(polars.from_arrow(i), json.loads(i.schema.metadata[b'cache_ranges'].decode())) for i in metadatas]
        metadata = polars.concat([f[0].with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, f in enumerate(metadatas)])
        cache_ranges = {f"{indices[i]}.lava": f[1] for i, f in enumerate(metadatas) if len(f[1]) > 0}
        cached_files = list(cache_ranges.keys())
        ranges = [[tuple(k) for k in cache_ranges[f]] for f in cached_files]
        rottnest.populate_cache(cached_files, ranges, "aws")
    else:
        metadatas = [polars.from_arrow(i) for i in metadatas]
        metadata = polars.concat([f.with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, f in enumerate(metadatas)])

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
    
    # magic number 2044 for vetors
    # result = read_row_groups(metadata["file_path"].to_list(), metadata["row_groups"].to_list(), [(i, i + 2044) for i in metadata['page_row_offset_in_row_group'].to_list()], column_name)

    row_group_rownr = [pyarrow.array(np.arange(metadata['page_row_offset_in_row_group'][i], metadata['page_row_offset_in_row_group'][i] + len(arr))) for i, arr in enumerate(result)]
    
    metadata_key = [pyarrow.array(np.ones(len(arr)).astype(np.uint32) * i) for i, arr in enumerate(result)]
    
    result = pyarrow.table([pyarrow.chunked_array(result), pyarrow.chunked_array(row_group_rownr), 
                            pyarrow.chunked_array(metadata_key)], names = [column_name, '__row_group_rownr__', '__metadata_key__'])
    
    return result, column_name, metadata.with_row_count('__metadata_key__')

def return_full_result(result: polars.DataFrame, metadata: polars.DataFrame, column_name: str, columns: List[str]):
    if columns != []:
        result = result.join(metadata.select(["__metadata_key__", "file_path", "row_groups"]), on = "__metadata_key__", how = "left")
        grouped = result.group_by(["file_path", "row_groups"]).agg([polars.col('__metadata_key__'), polars.col('__row_group_rownr__')])
        collected_results = polars.from_arrow(read_columns(grouped["file_path"].to_list(), grouped["row_groups"].to_list(), grouped["__row_group_rownr__"].to_list()))
        unnested_metadata_key = grouped['__metadata_key__'].explode()
        unnested_row_group_rownr = grouped['__row_group_rownr__'].explode()
        collected_results = collected_results.with_columns([unnested_metadata_key, unnested_row_group_rownr])
        result = result.join(collected_results, on = ["__metadata_key__", "__row_group_rownr__"], how = "left")
        return result.select(columns + [column_name])
    else:
        return result.select([column_name])

def index_files(index: RottnestIndex, file_paths: list[str], column_name: str, name = uuid.uuid4().hex, index_mode = "physical", remote = None):
    
    arr, uid, file_data = get_physical_layout(file_paths, column_name, type=index.data_type, remote = remote) if index_mode == "physical" else get_virtual_layout(file_paths, column_name, "uid", remote = remote)
    cache_ranges = index.build_index(arr, uid, f"{name}.lava")
    file_data = file_data.to_arrow()
    file_data = file_data.replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges.cache_ranges)})
    pq.write_table(file_data, f"{name}.meta", write_statistics = False, compression = 'zstd')

def merge_metadatas(index_names: List[str], suffix = "meta"):
    assert len(index_names) > 1
    
    metadatas = []
    for index_name in index_names:
        fs = get_fs_from_file_path(f"{index_name}.{suffix}")
        table = pq.read_table(f"{index_name.replace('s3://', '')}.{suffix}", filesystem=fs)
        metadatas.append(polars.from_arrow(table))
    
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    return offsets, polars.concat(metadatas)

def merge_indices(index: RottnestIndex, new_index_name: str, indices: List[str]):
    offsets, file_data = merge_metadatas(indices)
    cache_ranges = index.compact_indices(new_index_name, indices, offsets)
    file_data = file_data.to_arrow().replace_schema_metadata({"cache_ranges": json.dumps(cache_ranges.cache_ranges)})
    pq.write_table(file_data, f"{new_index_name}.meta", write_statistics = False, compression = 'zstd')

def search_index(index: RottnestIndex, indices: List[str], query: str, K: int, columns: list[str] = [], **kwargs):
    metadata = get_metadata_and_populate_cache(indices)
    index_search_results = index.search_index([f"{index_name}.lava" for index_name in indices], query, K, **kwargs)

    print(index_search_results)

    if len(index_search_results) == 0:
        return None
    
    if len(index_search_results) > index.brute_force_threshold:
        return "Brute Force Everything Please"

    result, column_name, metadata = get_result_from_index_result(metadata, index_search_results)
    result =  polars.from_arrow(index.brute_force(result, column_name, query, K))

    return return_full_result(result, metadata, column_name, columns)

import logging
from logging import Logger

def search_parquet_lake(index_files: list[str], indexed_files: list[str], unindexed_files: list[str], column: str, query: str, index: RottnestIndex, 
                        K: int = 10, extra_search_configs = {}, log: Logger = logging.getLogger(__name__)) -> polars.DataFrame:
    
    # Step 2: Query Index - Search each index file in parallel
    all_results = []

    if len(indexed_files) > 0:
        log.info(f"Searching {len(indexed_files)} indexed files")
        results = search_index(index, index_files, query, K , **extra_search_configs)
        all_results.append(results)
    
    print(all_results)
    
    # Step 3: In-situ Probing - Scan unindexed files if necessary
    if unindexed_files and (not all_results or len(all_results[0]) < K):
        log.info(f"Scanning {len(unindexed_files)} unindexed files")
        
        # Scan unindexed files
        try:
            # Create a temporary table with just the unindexed files
            scan_results = []
            
            # Read and scan each unindexed file
            for file_path in unindexed_files:
                try:
                    # Use PyArrow to read the file
                    fs = get_fs_from_file_path(file_path)
                    table = pq.read_table(file_path.replace("s3://", ""), columns=[column], filesystem=fs)
                    filtered = polars.from_arrow(index.brute_force(table, column, query, K))
                    
                    # Add file path information
                    if not filtered.is_empty():
                        filtered = filtered.with_columns(polars.lit(file_path).alias("file_path"))
                        scan_results.append(filtered)
                except Exception as e:
                    log.error(f"Error scanning file {file_path}: {e}")
            
            if scan_results:
                all_results.append(polars.concat(scan_results))
        except Exception as e:
            log.error(f"Error during unindexed file scan: {e}")
    
    # Combine all results and apply top-K
    if not all_results:
        return polars.DataFrame()
    
    final_results = polars.concat(all_results)
    return final_results

def group_mergeable_indices(mergeable_indices: polars.DataFrame, binpack_row_threshold: int) -> Tuple[List[List[str]], List[List[int]], List[List[str]]]:
    """Group mergeable indices based on row count threshold.
    
    Args:
        mergeable_indices: DataFrame containing index files and their record counts
        binpack_row_threshold: Maximum number of rows allowed in a group
        
    Returns:
        Tuple containing:
        - List of index file groups
        - List of record count groups
        - List of covered file groups
    """
    index_groups = [[]]
    record_counts = [[]]
    covered_files = [[]]

    for row in mergeable_indices.iter_rows(named=True):
        # If current group is empty, always add the file regardless of size
        record_count = sum(row['record_counts'])
        if not index_groups[-1]:
            index_groups[-1].append(row['index_file'])
            record_counts[-1].extend(row['record_counts'])
            covered_files[-1].extend(row['file_path'])
            current_group_row_count = record_count
        # If adding this file would exceed threshold and current group has files
        elif current_group_row_count + record_count > binpack_row_threshold and len(index_groups[-1]) > 0:
            index_groups.append([row['index_file']])  # Start new group with current file
            record_counts.append(row['record_counts'])
            covered_files.append(row['file_path'])
            current_group_row_count = record_count
        else:
            # Add to current group
            index_groups[-1].append(row['index_file'])
            record_counts[-1].extend(row['record_counts'])
            covered_files[-1].extend(row['file_path'])
            current_group_row_count += record_count

    # Filter out groups that have only one file
    index_groups_to_keep = [i for i, group in enumerate(index_groups) if len(group) > 1]
    if len(index_groups_to_keep) == 0:
        return [], [], []
        
    record_counts = [record_counts[i] for i in index_groups_to_keep]
    covered_files = [covered_files[i] for i in index_groups_to_keep]
    index_groups = [index_groups[i] for i in index_groups_to_keep]
    
    return index_groups, record_counts, covered_files