from google.cloud import storage
from datetime import datetime, timedelta, timezone
import os
import logging
from typing import List
import uuid
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

def list_files(bucket_name, key_prefix, index_timeout_seconds: int | None = None):
    
    results = []
    
    if index_timeout_seconds is None:
        index_timeout_seconds = 0
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    current_time = datetime.now(timezone.utc)
    timeout_threshold = current_time - timedelta(seconds=index_timeout_seconds)
    
    # List all blobs with the given prefix
    blobs = bucket.list_blobs(prefix=key_prefix)
    
    for blob in blobs:
        # Only include files older than INDEX_TIMEOUT
        if blob.updated and blob.updated < timeout_threshold:
            results.append(blob.name)
    
    return results

def upload_index_files(index_file_path: str, index_prefix: str) -> None:
    """Upload index files to GCS and clean up local files.
    
    Args:
        index_file_path: The base path of the index file (without extensions)
        index_prefix: The GCS prefix where index files will be stored (e.g. 'gs://bucket/prefix/')
    """
    client = storage.Client()
    # Parse the GCS path to extract bucket and key
    index_prefix_without_gs = index_prefix.replace('gs://', '').replace('gcs://', '')
    bucket_name = index_prefix_without_gs.split('/')[0]
    key_prefix = '/'.join(index_prefix_without_gs.split('/')[1:])
    key = f"{key_prefix}/{index_file_path}" if key_prefix else index_file_path
    
    bucket = client.bucket(bucket_name)
    
    # Upload both .lava and .meta files
    blob_lava = bucket.blob(key + ".lava")
    blob_lava.upload_from_filename(index_file_path + ".lava")
    
    blob_meta = bucket.blob(key + ".meta")
    blob_meta.upload_from_filename(index_file_path + ".meta")
    
    log.info(f"Index file {index_file_path} uploaded to gs://{bucket_name}/{key}")
    
    # Remove the local index files after uploading
    os.remove(index_file_path + ".lava")
    os.remove(index_file_path + ".meta")

def delete_gcs_files(bucket_name: str, files_to_delete: List[str], batch_size: int = 100) -> None:
    """Delete multiple files from GCS in batches.
    
    Args:
        bucket_name: The name of the GCS bucket
        files_to_delete: List of GCS blob names to delete
        batch_size: Maximum number of objects to delete in a single batch (default: 100)
    """
    if not files_to_delete:
        return
        
    log.info(f"Deleting {len(files_to_delete)} obsolete index files")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # GCS batch operations
    for i in range(0, len(files_to_delete), batch_size):
        batch = files_to_delete[i:i + batch_size]
        with client.batch():
            for blob_name in batch:
                blob = bucket.blob(blob_name)
                blob.delete()

def upload_parquet_to_gcs_atomic(table, gcs_prefix: str, filename: str = "metadata.parquet") -> None:
    """Upload a Parquet table to GCS atomically.
    
    This function ensures atomic upload by:
    1. Writing to a temporary local file
    2. Uploading to GCS in a single operation
    3. Cleaning up the temporary file
    
    Args:
        table: The table to upload (can be polars DataFrame or pyarrow Table)
        gcs_prefix: The GCS prefix where the file will be stored (e.g. 'gs://bucket/prefix/')
        filename: The name of the file to create in GCS (default: 'metadata.parquet')
    """
    client = storage.Client()
    temp_file = f"{uuid.uuid4().hex[:8]}.parquet"
    
    # Parse the GCS path to extract bucket and key
    gcs_prefix_without_gs = gcs_prefix.replace('gs://', '').replace('gcs://', '')
    bucket_name = gcs_prefix_without_gs.split('/')[0]
    key_prefix = '/'.join(gcs_prefix_without_gs.split('/')[1:])
    
    bucket = client.bucket(bucket_name)
    
    # Write table to temporary file and upload to GCS
    pq.write_table(table.to_arrow() if hasattr(table, 'to_arrow') else table, temp_file)
    blob = bucket.blob(f"{key_prefix}/{filename}")
    blob.upload_from_filename(temp_file)
    
    # Clean up temporary file
    os.remove(temp_file)