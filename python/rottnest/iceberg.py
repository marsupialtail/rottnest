import os
import time
import logging
import uuid
import json
import pyarrow
import pyarrow.parquet as pq
import polars
import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple
from pyiceberg.catalog.glue import (
    AWS_REGION as ICEBERG_GLUE_PROPERTY_KEY_AWS_REGION,
    GLUE_REGION as ICEBERG_GLUE_PROPERTY_KEY_GLUE_REGION,
)
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.io.pyarrow import pyarrow_to_schema
from .utils import get_fs_from_file_path, get_physical_layout
from .internal import index_files_bm25, index_files_substring, index_files_vector, index_files_uuid
import boto3
CATALOG_NAME = os.getenv("CATALOG_NAME")
CATALOG_AWS_REGION = os.getenv("CATALOG_AWS_REGION")

INDEX_TIMEOUT = 3600

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def default_catalog():
    from pyiceberg.catalog import load_catalog

    return load_catalog(
        CATALOG_NAME,
        type="glue",
        region_name=CATALOG_AWS_REGION,
        **{
            ICEBERG_GLUE_PROPERTY_KEY_AWS_REGION: CATALOG_AWS_REGION,
            ICEBERG_GLUE_PROPERTY_KEY_GLUE_REGION: CATALOG_AWS_REGION,
        },
    )

def load_table(table_name: str, retries = 10):
    catalog = default_catalog()
    for i in range(retries):
        try:
            return catalog.load_table(f"{table_name}")
        except Exception as e:
            log.error(f"Failed to load table {table_name}: {e}")
            if i == retries - 1:
                raise
            time.sleep(2)

def index_iceberg(table: str, column: str, index_table: str, iceberg_location: str, index_prefix: str, type: str, index_impl = 'physical', extra_configs = {}):
    """
    Index an Iceberg table with Rottnest.
    
    Args:
        table: The name of the Iceberg table to index
        column: The column to index
        index_table: The name of the Iceberg table to store index metadata
        type: The type of index to build ('bm25', 'substring', 'vector', 'uuid')
        index_impl: The implementation of the index ('physical' or 'virtual')
        extra_configs: Additional configuration parameters for the indexing
    
    Returns:
        The name of the index file created
    """
    
    assert 's3://' in index_prefix, "Index prefix must be an S3 path"
    assert type.lower() in ['bm25', 'substring', 'vector', 'uuid'], "Unsupported index type"
    
    # Step 1: Plan - Find Parquet files in the current snapshot that aren't indexed yet
    iceberg_table = load_table(table)
    assert iceberg_table is not None, f"Table {table} not found"
    
    data_files = iceberg_table.inspect.data_files()['file_path'].to_pylist()
    
    if not data_files:
        log.info(f"No Parquet files found in the current snapshot of {table}")
        return None
    
    # Get the list of files that have already been indexed
    metadata_table = None
    try:
        metadata_table = load_table(index_table, retries=1)
        indexed_files_df = polars.from_pandas(metadata_table.scan(selected_fields=("file_path",)).to_pandas())
        indexed_files = set(indexed_files_df.explode('file_path')['file_path'])
        log.info(f"Found {len(indexed_files)} indexed files in {index_table}")
    except NoSuchTableError:
        log.error(f"Failed to load metadata table {index_table}, no existing index, constructing new index.")
        indexed_files = set()
    
    # Find files that need to be indexed
    files_to_index = [f for f in data_files if f not in indexed_files]
    
    if not files_to_index:
        log.info(f"All files in the current snapshot of {table} are already indexed")
        return None
    
    log.info(f"Found {len(files_to_index)} new files to index out of {len(data_files)} total files")
    
    # Step 2: Index - Build an index for the new files
    index_file_path = f"{uuid.uuid4().hex[:8]}"
    
    # Set timeout for the indexing operation
    timeout = extra_configs.get('timeout', INDEX_TIMEOUT)  # Default timeout: 1 hour
    
    worker_exception = None
    indexing_success = [False]

    # Define the worker function that will run in a separate thread
    def indexing_worker():
        try:
            # Call the appropriate indexing function based on the type
            if type.lower() == 'bm25':
                index_files_bm25(files_to_index, column, index_file_path, index_mode=index_impl)
            elif type.lower() == 'substring':
                token_skip_factor = extra_configs.get('token_skip_factor', None)
                char_index = extra_configs.get('char_index', False)
                index_files_substring(files_to_index, column, index_file_path, index_mode=index_impl, 
                                    token_skip_factor=token_skip_factor, char_index=char_index)
            elif type.lower() == 'vector':
                dtype = extra_configs.get('dtype', 'f32')
                gpu = extra_configs.get('gpu', False)
                index_files_vector(files_to_index, column, index_file_path, dtype=dtype, 
                                index_mode=index_impl, gpu=gpu)
            elif type.lower() == 'uuid':
                index_files_uuid(files_to_index, column, index_file_path, index_mode=index_impl)
            else:
                raise ValueError(f"Unsupported index type: {type}")
            
            # If we get here, indexing was successful
            indexing_success[0] = True

            # copy the index file to the specified prefix
            s3 = boto3.client('s3')
            # Parse the S3 path to extract bucket and key
            index_prefix_without_s3 = index_prefix.replace('s3://', '')
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
            'table_name': table,
            'column_name': column,
            'file_path': files_to_index,
            'index_file': f"{index_prefix}/{index_file_path}",
            'index_type': type,
            'index_impl': index_impl,
            'rows_indexed': 0,  # This could be populated with actual count if needed
            'index_timestamp': int(time.time())
        }]
        
        metadata_df = polars.DataFrame(metadata_records).to_arrow()

        print(metadata_table)

        if metadata_table is None:
            catalog = default_catalog()
            # Create the table with the schema from our metadata DataFrame
            iceberg_location = iceberg_location.rstrip('/')
            db, table = index_table.split('.')
            metadata_table = catalog.create_table(index_table, location = f"{iceberg_location}/{db}.db/{table}", schema=metadata_df.schema, properties={})

        metadata_table.append(metadata_df)
        
        log.info(f"Successfully indexed {len(files_to_index)} files for {table}.{column} with index {index_file_path}")
        return index_file_path      
        
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


def search_iceberg(table: str, column: str, index_table: str, query: Any, type: str, K: int = 10, extra_configs = {}):
    """
    Search an Iceberg table using Rottnest indices.
    
    Args:
        table: The name of the Iceberg table to search
        column: The column to search
        index_table: The name of the Iceberg table containing index metadata
        query: The query to search for (string for bm25/substring, vector for vector search)
        type: The type of index to use ('bm25', 'substring', 'vector', 'uuid')
        K: Number of top results to return
        extra_configs: Additional configuration parameters for the search
    
    Returns:
        The search results as a DataFrame
    """
    # Step 1: Plan - Find which index files cover the Parquet files in the snapshot
    iceberg_table = load_table(table)
    metadata_table = load_table(index_table)
    
    # Get the list of data files in the current snapshot
    data_files = iceberg_table.inspect.data_files()['file_path'].to_pylist()
    if not data_files:
        log.info(f"No data files found in the snapshot of {table}")
        return polars.DataFrame()
    
    # Get the list of indexed files and their corresponding index files
    try:
        indexed_df = polars.from_pandas(metadata_table.scan(
            selected_fields=("file_path", "index_file")
        ).to_pandas())
        indexed_files = set(indexed_df.explode('file_path')['file_path'])
        index_files = list(indexed_df['index_file'])
        
    except NoSuchTableError:
        log.error(f"No index has been built at {index_table}")
        indexed_files = set()
    
    # Determine which files are indexed and which need to be scanned
    indexed_files = set(indexed_files)
    unindexed_files = [file for file in data_files if file not in indexed_files]
    
    # Step 2: Query Index - Search each index file in parallel
    all_results = []

    if len(indexed_files) > 0:
        log.info(f"Searching {len(indexed_files)} indexed files")
    
        if type.lower() == 'bm25':
            from .internal import search_index_bm25
            results = search_index_bm25(index_files, query, K * 2)  # Get more results to account for filtering
        elif type.lower() == 'substring':
            from .internal import search_index_substring
            results = search_index_substring(index_files, query, K * 2)
        elif type.lower() == 'vector':
            from .internal import search_index_vector
            results = search_index_vector(index_files, query, K * 2)
        elif type.lower() == 'uuid':
            from .internal import search_index_uuid
            results = search_index_uuid(index_files, query, K * 2)
        else:
            raise ValueError(f"Unsupported index type: {type}")
                
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
                    df = polars.from_arrow(table)
                    
                    # Apply the search predicate based on the index type
                    if type.lower() == 'bm25':
                        # Simple contains for demonstration - in practice, would use a proper BM25 implementation
                        filtered = df.filter(polars.col(column).str.contains(query))
                    elif type.lower() == 'substring':
                        filtered = df.filter(polars.col(column).str.contains(query))
                    elif type.lower() == 'vector':
                        # Vector search would require computing distances - simplified here
                        log.warning("Vector search on unindexed files not fully implemented")
                        filtered = df.head(0)  # Empty dataframe as placeholder
                    elif type.lower() == 'uuid':
                        filtered = df.filter(polars.col(column) == query)
                    else:
                        filtered = df.head(0)  # Empty dataframe as placeholder
                    
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
    
    # Sort and limit to top K results
    if type.lower() in ['bm25', 'vector']:
        # These types have a score column
        if 'score' in final_results.columns:
            final_results = final_results.sort('score', descending=True).head(K)
    
    return final_results.head(K)
