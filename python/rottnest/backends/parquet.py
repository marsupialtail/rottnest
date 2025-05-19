# A very simple backend assuming you store data in a Parquet prefix
# this can be used as an example to develop other backends
# this only supports appends, does not support deletes/updates etc.

from dataclasses import dataclass, field
from rottnest.indices.index_interface import RottnestIndex
from .s3_utils import list_files
import boto3
import polars
import time
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem
import logging, os
import threading
import uuid
from .utils import get_fs_from_file_path, get_physical_layout, index_files, merge_indices, search_index

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
    for i, row in enumerate(files_to_index.iter_rows(named=True)):
        # If current group is empty, always add the file regardless of size
        if not index_groups[-1]:
            index_groups[-1].append(row['file_path'])
            record_counts[-1].append(row['record_count'])
            current_group_row_count = row['record_count']
        # If adding this file would exceed threshold and current group has files
        elif current_group_row_count + row['record_count'] > config.binpack_row_threshold and len(index_groups[-1]) > 0:
            index_groups.append([row['file_path']])  # Start new group with current file
            record_counts.append([row['record_count']])
            current_group_row_count = row['record_count']
        else:
            # Add to current group
            index_groups[-1].append(row['file_path'])
            record_counts[-1].append(row['record_count'])
            current_group_row_count += row['record_count']
        
        
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
                s3 = boto3.client('s3')
                # Parse the S3 path to extract bucket and key
                index_prefix_without_s3 = config.index_prefix.replace('s3://', '')
                bucket_name = index_prefix_without_s3.split('/')[0]
                key_prefix = '/'.join(index_prefix_without_s3.split('/')[1:])
                key = f"{key_prefix}/{index_file_path}" if key_prefix else index_file_path
                s3.upload_file(index_file_path + ".lava", bucket_name, key + ".lava")
                s3.upload_file(index_file_path + ".meta", bucket_name, key + ".meta")
                log.info(f"Index file {index_file_path} uploaded to s3://{bucket_name}/{key}")
                # Remove the local index file after uploading
                os.remove(index_file_path + ".lava")
                os.remove(index_file_path + ".meta")

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
        
        metadata_df = polars.DataFrame(metadata_records).to_arrow()
        

        
        
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
