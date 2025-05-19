import boto3
from datetime import datetime, timedelta, timezone

def list_files(bucket_name, key_prefix, index_timeout_seconds: int | None = None):

    results = []

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