# A very simple backend assuming you store data in a Parquet prefix
# this can be used as an example to develop other backends
# this only supports appends, does not support deletes/updates etc.

from dataclasses import dataclass, field
from rottnest.indices.index_interface import RottnestIndex
from .s3_utils import list_files, upload_index_files, delete_s3_files, upload_parquet_to_s3_atomic
import boto3
import polars
import time
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem
import logging, os
import threading
import uuid
from .utils import get_fs_from_file_path, get_physical_layout, index_files, merge_indices, search_index, search_parquet_lake, group_mergeable_indices

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class ParquetConfig:
    
    data_prefix: str
    column: str
    index_prefix: str
    binpack_row_threshold: int = 10000
    index_timeout: int = 3600
    extra_configs: dict = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing.
        
        Strips trailing slashes from index_prefix to ensure consistent path handling.
        """
        self.index_prefix = self.index_prefix.rstrip('/')
        self.data_prefix = self.data_prefix.rstrip('/')

"""
metadata_records = [{
    'prefix': config.data_prefix,
    'column_name': config.column,
    'file_path': group,
    'record_counts': record_count,
    'index_file': f"{config.index_prefix}/{index_file_path}",
    'index_type': index.name,
    'index_impl': index.index_mode,
    'rows_indexed': 0,  # This could be populated with actual count if needed
    'index_timestamp': int(time.time())
}]

"""

# consistency is provided with a metadata parquet file stored in index_prefix
# relies on S3 strong consistency and atomic PUT operations

def index_parquet(config: ParquetConfig, index: RottnestIndex, extra_index_kwargs = {}):

    # list all the files in data_prefix using boto3 
    data_prefix_without_s3 = config.data_prefix.replace('s3://', '')
    bucket_name = data_prefix_without_s3.split('/')[0]
    key_prefix = '/'.join(data_prefix_without_s3.split('/')[1:])
    data_files = ['s3://' + bucket_name + '/' + f for f in list_files(bucket_name, key_prefix, None)]

    # figure out if metadata.parquet exists in index_prefix
    index_prefix_without_s3 = config.index_prefix.replace('s3://', '')
    bucket_name = index_prefix_without_s3.split('/')[0]

    s3 = boto3.client('s3')
    existing_metadata = None
    try:
        s3.head_object(Bucket=bucket_name, Key='metadata.parquet')

        existing_metadata = polars.from_arrow(pq.read_table(index_prefix_without_s3 + '/metadata.parquet',
                                                             filesystem = S3FileSystem()))

        indexed_files = set(existing_metadata.explode('file_path')['file_path'])
    except Exception as e:
        indexed_files = set()
    
    files_to_index = [file for file in data_files if file not in indexed_files]
    # TODO: this line could be very expensive, try to parallelize
    record_counts = [pq.ParquetFile(file.replace('s3://',''), filesystem=S3FileSystem()) for file in files_to_index]

    index_groups = [[]]
    record_counts = [[]]
    current_group_row_count = 0 
    for file_path, record_count in zip(files_to_index, record_counts):
        # If current group is empty, always add the file regardless of size
        if not index_groups[-1]:
            index_groups[-1].append(file_path)
            record_counts[-1].append(record_count)
            current_group_row_count = record_count
        # If adding this file would exceed threshold and current group has files
        elif current_group_row_count + record_count > config.binpack_row_threshold and len(index_groups[-1]) > 0:
            index_groups.append([file_path])  # Start new group with current file
            record_counts.append([record_count])
            current_group_row_count = record_count
        else:
            # Add to current group
            index_groups[-1].append(file_path)
            record_counts[-1].append(record_count)
            current_group_row_count += record_count
        
        
    print(index_groups)

    # the different groups should be distributed not parallelized since each indexing
    # function could be multithreaded. No plans for distributed indexing to be in open source.

    index_file_paths = []

    for group, record_count in zip(index_groups, record_counts):
        log.info(f"Indexing group of {len(group)} files")
        
        # Step 2: Index - Build an index for the new files
        index_file_path = f"{uuid.uuid4().hex[:8]}"
        index_file_paths.append(index_file_path)
        
        worker_exception = None
        indexing_success = [False]

        # Define the worker function that will run in a separate thread
        def indexing_worker():
            try:
                index_files(index, group, config.column, index_file_path, index_mode = "physical", remote = None)
                
                # If we get here, indexing was successful
                indexing_success[0] = True

                # copy the index file to the specified prefix
                upload_index_files(index_file_path, config.index_prefix)

            except Exception as e:
                # Store the exception
                worker_exception = e
                log.error(f"Error during indexing: {e}")
        
        # Start the indexing in a separate thread
        worker_thread = threading.Thread(target=indexing_worker)
        worker_thread.daemon = True  # Allow the thread to be terminated when the main thread exits
        worker_thread.start()
        
        # Wait for the worker to complete or timeout
        worker_thread.join(config.index_timeout)
        
        # Check if the thread is still alive (timeout occurred)
        if worker_thread.is_alive():
            log.error(f"Indexing operation timed out after {config.index_timeout} seconds")
            # We can't forcibly terminate the thread in Python, but we can proceed
            # and let it run in the background
            return None
        
        if worker_exception is not None:
            raise worker_exception  
        if not indexing_success[0]:
            log.error("Indexing operation failed without an exception")
            return None
    
    try:
    # Step 3: Commit - Insert records into the metadata table
        metadata_records = [{
            'prefix': config.data_prefix,
            'column_name': config.column,
            'file_path': group,
            'record_counts': record_count,
            'index_file': f"{config.index_prefix}/{index_file_path}",
            'index_type': index.name,
            'index_impl': index.index_mode,
            'rows_indexed': 0,  # This could be populated with actual count if needed
            'index_timestamp': int(time.time())
        } for group, index_file_path in zip(index_groups, index_file_paths)]
        
        if existing_metadata is None:
            new_metadata = polars.DataFrame(metadata_records)
        else:
            new_metadata = polars.concat([existing_metadata, polars.DataFrame(metadata_records)])

        # this needs to be atomic. So we will write to a local file and then upload
        upload_parquet_to_s3_atomic(new_metadata, config.index_prefix)
        
        log.info(f"Successfully indexed {len(files_to_index)} files for {config.table}.{config.column} with index {index_file_path}")
        return index_file_paths      
        
    except Exception as e:
        log.error(f"Error during metadata commit: {e}")
        # Cleanup any partial index files that might have been created
        try:
            if os.path.exists(f"{index_file_path}.lava"):
                os.remove(f"{index_file_path}.lava")
            if os.path.exists(f"{index_file_path}.meta"):
                os.remove(f"{index_file_path}.meta")
        except Exception as cleanup_error:
            log.error(f"Error during cleanup: {cleanup_error}")
        return None

def search_parquet(config: ParquetConfig, query: str, index: RottnestIndex, K: int = 10, extra_search_configs = {}):
    # list all the files in data_prefix using boto3 
    data_prefix_without_s3 = config.data_prefix.replace('s3://', '')
    bucket_name = data_prefix_without_s3.split('/')[0]
    key_prefix = '/'.join(data_prefix_without_s3.split('/')[1:])
    data_files = ['s3://' + bucket_name + '/' + f for f in list_files(bucket_name, key_prefix, None)]
    
    # read the metadata file
    
    try:
        metadata_file = f"{config.index_prefix}/metadata.parquet"
        indexed_df = polars.from_arrow(pq.read_table(metadata_file, filesystem = S3FileSystem()))
        indexed_files = set(indexed_df.explode('file_path')['file_path'])
        index_files = list(indexed_df['index_file'])
        
    except:
        log.error(f"No index has been built at {config.index_table}")
        indexed_files = set()
    
    # Determine which files are indexed and which need to be scanned
    indexed_files = set(indexed_files)
    unindexed_files = [file for file in data_files if file not in indexed_files]
    
    final_results = search_parquet_lake(index_files, indexed_files, unindexed_files, config.column, 
                                        query, index, K, extra_search_configs)
    
    return final_results.head(K)

def vacuum_parquet_indices(config: ParquetConfig):
    """
    1) Plan: determine which index files in the metadata table
    to keep based on snapshot_id. Rottnest currently
    uses a simple heuristic: it first computes all Parquet files
    included in all the snapshots past snapshot_id. Then
    it greedily selects index files that cover the most number
    of active Parquet files. The procedure stops when the
    number of covered Parquet files cannot be increased. This
    procedure maximizes the number of covered Parquet files
    while heuristically minimizing the number of index files
    
    Args:
        index_table: The name of the Iceberg table containing index metadata
        index_prefix: The prefix of the index files
    """

    # list all the files in config.data_prefix to get the live files
    bucket_name = config.data_prefix.split('/')[0]
    key_prefix = '/'.join(config.data_prefix.split('/')[1:])
    live_files = list_files(bucket_name, key_prefix, 0)
    
    # load the index_table into polars dataframe
    metadata_table = polars.from_arrow(pq.read_table(config.index_prefix + '/metadata.parquet', filesystem = S3FileSystem()))
    # figure out what index files can be dropped
    keep_rows = []
    to_delete_index_files = []
    for i in range(len(metadata_table)):
        covered_files = metadata_table['file_path'][i]
        if any(file in live_files for file in covered_files):
            keep_rows.append(i)
        else:
            to_delete_index_files.append(metadata_table['index_file'][i])

    metadata_table = metadata_table[keep_rows]

    # Upload metadata table to S3
    upload_parquet_to_s3_atomic(metadata_table, config.index_prefix)
    
    index_files = list_files(bucket_name, key_prefix, config.index_timeout)
    
    log.info(f"Found {len(index_files)} index files older than {config.index_timeout} seconds in {config.index_prefix}")
    
    # Now index_files contains all the keys from all pages
    # Filter out index files that are still in the metadata table
    live_index_files = set(metadata_table['index_file'])
    # expand live index files to include the .lava and .meta suffix for each file
    live_index_files = set([f + ".lava" for f in live_index_files] + [f + ".meta" for f in live_index_files])
    files_to_delete = [f for f in index_files if f"s3://{bucket_name}/{f}" not in live_index_files]
    
    if files_to_delete:
        log.info(f"Deleting {len(files_to_delete)} obsolete index files")
        delete_s3_files(bucket_name, files_to_delete)
    else:
        log.info("No obsolete index files to delete")
    
    print(f"Vacuuming complete, deleted {to_delete_index_files}")

def compact_parquet_indices(config: ParquetConfig, index: RottnestIndex, extra_configs = {}):
    """
    Compact Rottnest indices
    """

    # load the index_table into polars dataframe

    metadata_table = polars.from_arrow(pq.read_table(config.index_prefix + '/metadata.parquet', filesystem = S3FileSystem()))

    mergeable_indices = metadata_table.filter(polars.col('record_counts').list.sum() < config.binpack_row_threshold)

    index_groups, record_counts, covered_files = group_mergeable_indices(mergeable_indices, config.binpack_row_threshold)
    if not index_groups:
        print("No indices to merge")
        return

    index_file_paths = []

    for group in index_groups:
        log.info(f"Indexing group of {len(group)} files")
        
        # Step 2: Index - Build an index for the new files
        index_file_path = f"{uuid.uuid4().hex[:8]}"
        index_file_paths.append(index_file_path)
        
        # Set timeout for the indexing operation
        timeout = config.index_timeout
        
        worker_exception = None
        indexing_success = [False]

        # Define the worker function that will run in a separate thread
        def indexing_worker():
            try:
                merge_indices(index, index_file_path, group)
                
                # If we get here, indexing was successful
                indexing_success[0] = True

                # Upload index files to S3 and clean up local files
                upload_index_files(index_file_path, config.index_prefix)

            except Exception as e:
                # Store the exception
                worker_exception = e
                log.error(f"Error during indexing: {e}")
        
        # Start the indexing in a separate thread
        worker_thread = threading.Thread(target=indexing_worker)
        worker_thread.daemon = True  # Allow the thread to be terminated when the main thread exits
        worker_thread.start()
        
        # Wait for the worker to complete or timeout
        worker_thread.join(timeout)
        
        # Check if the thread is still alive (timeout occurred)
        if worker_thread.is_alive():
            log.error(f"Indexing operation timed out after {timeout} seconds")
            # We can't forcibly terminate the thread in Python, but we can proceed
            # and let it run in the background
            return None
        
        if worker_exception is not None:
            raise worker_exception  
        if not indexing_success[0]:
            log.error("Indexing operation failed without an exception")
            return None
    
    try:
    # Step 3: Commit - Insert records into the metadata table
        metadata_records = [{
            'prefix': config.data_prefix,
            'column_name': config.column,
            'file_path': covered_files,
            'record_counts': record_count,
            'index_file': f"{config.index_prefix}/{index_file_path}",
            'index_type': index.name,
            'index_impl': index.index_mode,
            'rows_indexed': 0,  # This could be populated with actual count if needed
            'index_timestamp': int(time.time())
        } for index_file_path, record_count, covered_files in zip(index_file_paths, record_counts, covered_files)]
        
        index_groups = [item for sublist in index_groups for item in sublist]   
        remaining_metadata_df = metadata_table.filter(~polars.col('index_file').is_in(index_groups))
        new_metadata_df = polars.DataFrame(metadata_records)
        metadata_df = polars.concat([remaining_metadata_df, new_metadata_df])
        upload_parquet_to_s3_atomic(metadata_df, config.index_prefix)

        return index_file_paths      
        
    except Exception as e:
        log.error(f"Error during metadata commit: {e}")
        # Cleanup any partial index files that might have been created
        try:
            if os.path.exists(f"{index_file_path}.lava"):
                os.remove(f"{index_file_path}.lava")
            if os.path.exists(f"{index_file_path}.meta"):
                os.remove(f"{index_file_path}.meta")
        except Exception as cleanup_error:
            log.error(f"Error during cleanup: {cleanup_error}")
        return None
