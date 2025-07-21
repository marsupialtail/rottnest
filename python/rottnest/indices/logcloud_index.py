from rottnest.indices.index_interface import RottnestIndex, CacheRanges
import rottnest.rottnest as rottnest
import pyarrow
import polars
from typing import List
import numpy as np
import os
import uuid
import pyarrow.compute as pac
import time
import multiprocessing
from rottnest.backends.utils import get_physical_layout, get_metadata_and_populate_cache, get_result_from_index_result, return_full_result
# refactor later 

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
    # clean up compressed folder and all the {num_groups}.maui files
    for i in range(num_groups):
        os.remove(f"{i}.maui")
    os.rmdir(f"compressed_{name}")
    rottnest.index_logcloud(name, num_groups, wavelet_tree = wavelet)



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
        try:
            import daft
        except ImportError:
            raise ImportError(
                "getdaft is required for LogCloud search functionality. "
                "Install it with: pip install rottnest[logcloud]"
            )
        
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

