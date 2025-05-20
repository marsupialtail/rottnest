import boto3
import pyarrow.parquet as pq
import pyarrow
import os
import polars
from typing import List, Optional

# Configuration
BUCKET_NAME = os.environ.get('ROTTNEST_PARQUET_TEST_PREFIX', None)  # Read from env or use default
assert BUCKET_NAME is not None, "ROTTNEST_PARQUET_TEST_PREFIX environment variable must be set"
S3_PREFIX = f's3://{BUCKET_NAME}/parquet_data/test/'

# Initialize AWS clients
s3 = boto3.client('s3')

def upload_to_s3():
    """Upload the parquet file to S3"""
    try:
        s3.upload_file('test.parquet', BUCKET_NAME, 'parquet_data/test/test.parquet')
        print(f"Successfully uploaded test.parquet to s3://{BUCKET_NAME}/parquet_data/test/")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        raise

def create_parquet_table(total_files: int = 1):
    """Create multiple parquet files in the S3 prefix"""
    try:
        # Read parquet file using PyArrow
        df = pq.read_table("test.parquet").select(['index', 'text', 'timestamp', 'url'])
        rows_per_file = len(df) // total_files

        for i in range(total_files):
            chunk = df[i*rows_per_file:(i+1)*rows_per_file]
            # Write to temporary file
            temp_file = f"test_{i}.parquet"
            pq.write_table(chunk, temp_file)
            # Upload to S3
            s3.upload_file(temp_file, BUCKET_NAME, f'parquet_data/test_{total_files}/test_{i}.parquet')
            # Clean up temporary file
            os.remove(temp_file)
        
        print(f"Successfully created {total_files} parquet files in {S3_PREFIX}")
        
    except Exception as e:
        print(f"Error creating parquet files: {e}")
        raise

def read_parquet_table():
    """Read all parquet files from the S3 prefix"""
    try:
        # List all parquet files in the prefix
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='parquet_data/test/'
        )
        
        if 'Contents' not in response:
            print("No files found in the prefix")
            return None
            
        # Read each parquet file
        tables = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.parquet'):
                # Download file temporarily
                temp_file = f"temp_{os.path.basename(obj['Key'])}"
                s3.download_file(BUCKET_NAME, obj['Key'], temp_file)
                
                # Read parquet file
                table = pq.read_table(temp_file)
                tables.append(table)
                
                # Clean up temporary file
                os.remove(temp_file)
        
        if not tables:
            print("No parquet files found")
            return None
            
        # Combine all tables
        combined_table = pyarrow.concat_tables(tables)
        print(f"Successfully read {len(tables)} parquet files from {S3_PREFIX}")
        print("\nTable Schema:")
        print(combined_table.schema)
        print("\nFirst few rows:")
        print(combined_table.slice(0, 5).to_pandas())
        
        return combined_table
        
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        raise

def cleanup_parquet_table():
    """Delete all parquet files in the S3 prefix"""
    try:
        # List all objects in the prefix
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='parquet_data/test/'
        )
        
        if 'Contents' not in response:
            print("No files found to delete")
            return
            
        # Delete each object
        for obj in response['Contents']:
            s3.delete_object(
                Bucket=BUCKET_NAME,
                Key=obj['Key']
            )
            
        print(f"Successfully deleted all files in {S3_PREFIX}")
        
    except Exception as e:
        print(f"Error cleaning up parquet files: {e}")
        raise

if __name__ == "__main__":
    # Clean up any existing files
    # cleanup_parquet_table()
    
    # Upload and create test files
    upload_to_s3()
    create_parquet_table()
    create_parquet_table(10)  # Create 10 files
    
    # Read the files back
    print("\nReading back the files:")
    arrow_table = read_parquet_table() 