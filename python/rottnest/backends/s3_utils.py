import boto3
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

    s3 = boto3.client('s3')
    current_time = datetime.now(timezone.utc)  # S3 timestamps are in UTC
    timeout_threshold = current_time - timedelta(seconds=index_timeout_seconds)
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=key_prefix):
        if 'Contents' in page:  # Check if the page has any contents
            # Only include files older than INDEX_TIMEOUT
            old_files = [
                file for file in page['Contents'] 
                if file['LastModified'] < timeout_threshold
            ]
            results.extend([file['Key'] for file in old_files])

    return results

def upload_index_files(index_file_path: str, index_prefix: str) -> None:
    """Upload index files to S3 and clean up local files.
    
    Args:
        index_file_path: The base path of the index file (without extensions)
        index_prefix: The S3 prefix where index files will be stored (e.g. 's3://bucket/prefix/')
    """
    s3 = boto3.client('s3')
    # Parse the S3 path to extract bucket and key
    index_prefix_without_s3 = index_prefix.replace('s3://', '')
    bucket_name = index_prefix_without_s3.split('/')[0]
    key_prefix = '/'.join(index_prefix_without_s3.split('/')[1:])
    key = f"{key_prefix}/{index_file_path}" if key_prefix else index_file_path
    
    # Upload both .lava and .meta files
    s3.upload_file(index_file_path + ".lava", bucket_name, key + ".lava")
    s3.upload_file(index_file_path + ".meta", bucket_name, key + ".meta")
    log.info(f"Index file {index_file_path} uploaded to s3://{bucket_name}/{key}")
    
    # Remove the local index files after uploading
    os.remove(index_file_path + ".lava")
    os.remove(index_file_path + ".meta")

def delete_s3_files(bucket_name: str, files_to_delete: List[str], batch_size: int = 1000) -> None:
    """Delete multiple files from S3 in batches.
    
    Args:
        bucket_name: The name of the S3 bucket
        files_to_delete: List of S3 keys to delete
        batch_size: Maximum number of objects to delete in a single API call (default: 1000)
    """
    if not files_to_delete:
        return
        
    log.info(f"Deleting {len(files_to_delete)} obsolete index files")
    # S3 delete_objects requires keys to be in a specific format
    objects_to_delete = [{'Key': key} for key in files_to_delete]
    
    # S3 can only delete up to 1000 objects in one call, so we need to batch
    s3 = boto3.client('s3')
    for i in range(0, len(objects_to_delete), batch_size):
        batch = objects_to_delete[i:i + batch_size]
        s3.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': batch,
                'Quiet': True  # Don't return the result of each deletion
            }
        )

def upload_parquet_to_s3_atomic(table, s3_prefix: str, filename: str = "metadata.parquet") -> None:
    """Upload a Parquet table to S3 atomically.
    
    This function ensures atomic upload by:
    1. Writing to a temporary local file
    2. Uploading to S3 in a single operation
    3. Cleaning up the temporary file
    
    Args:
        table: The table to upload (can be polars DataFrame or pyarrow Table)
        s3_prefix: The S3 prefix where the file will be stored (e.g. 's3://bucket/prefix/')
        filename: The name of the file to create in S3 (default: 'metadata.parquet')
    """
    s3 = boto3.client('s3')
    temp_file = f"{uuid.uuid4().hex[:8]}.parquet"
    
    # Parse the S3 path to extract bucket and key
    s3_prefix_without_s3 = s3_prefix.replace('s3://', '')
    bucket_name = s3_prefix_without_s3.split('/')[0]
    key_prefix = '/'.join(s3_prefix_without_s3.split('/')[1:])
    
    # Write table to temporary file and upload to S3
    pq.write_table(table.to_arrow() if hasattr(table, 'to_arrow') else table, temp_file)
    s3.upload_file(temp_file, bucket_name, f"{key_prefix}/{filename}")
    
    # Clean up temporary file
    os.remove(temp_file)