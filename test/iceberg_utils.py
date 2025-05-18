import boto3
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField, LongType, StringType, DoubleType
)
import pyarrow.parquet as pq
import os

# Configuration
BUCKET_NAME = os.environ.get('ROTTNEST_BUCKET_NAME', None)  # Read from env or use default
assert BUCKET_NAME is not None, "ROTTNEST_BUCKET_NAME environment variable must be set"
DATABASE_NAME = 'rottnest'    # Replace with your Glue database name
S3_PREFIX = f's3://{BUCKET_NAME}/iceberg_data/test/'

# Initialize AWS clients
s3 = boto3.client('s3')

def upload_to_s3():
    """Upload the parquet file to S3"""
    try:
        s3.upload_file('test.parquet', BUCKET_NAME, 'iceberg_data/test/test.parquet')
        print(f"Successfully uploaded test.parquet to s3://{BUCKET_NAME}/iceberg_data/test/")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        raise

def get_catalog():
    catalog = load_catalog(
        "glue",
        **{
            "type": "glue",
            "warehouse": S3_PREFIX,
            "region": s3.meta.region_name
        }
    )
    return catalog

def delete_if_exists(table_name: str, catalog = get_catalog()):
    """Delete table if it already exists"""
    database_name, table_name = table_name.split('.')[0], table_name.split('.')[1]
    if table_name in [i[1] for i in catalog.list_tables(database_name)]:
        print(f"Deleting table {database_name}.{table_name}")
        catalog.drop_table(identifier=(database_name, table_name))

def create_iceberg_table(table_name: str, total_files: int = 1):
    """Create an Iceberg table and load data from the parquet file"""
    try:
        # Create catalog
        catalog = get_catalog()
        
        # Create namespace if it doesn't exist
        if DATABASE_NAME not in [i[0] for i in catalog.list_namespaces()]:
            catalog.create_namespace(DATABASE_NAME)
        
        # Define schema based on the parquet file
        schema = Schema(
            NestedField(1, "index", LongType(), required=False),
            NestedField(2, "text", StringType(), required=False),
            NestedField(3, "timestamp", StringType(), required=False),
            NestedField(5, "url", StringType(), required=False),
        )

        # Delete table if it already exists
        delete_if_exists(catalog, table_name)
        
        # Create table
        table = catalog.create_table(
            identifier=(DATABASE_NAME, table_name),
            schema=schema,
            location=S3_PREFIX,
            properties={
                "format-version": "2",
                "write.parquet.compression-codec": "snappy"
            }
        )
        
        print(f"Successfully created Iceberg table {DATABASE_NAME}.{table_name}")
        
        # Read parquet file using PyArrow
        df = pq.read_table("test.parquet").select(['index', 'text', 'timestamp', 'url'])
        rows_per_file = len(df) // total_files

        for i in range(total_files):
            chunk = df[i*rows_per_file:(i+1)*rows_per_file]
            # Write to Iceberg table
            table.append(chunk)
        
        print(f"Successfully loaded data into {DATABASE_NAME}.{table_name}")
        
    except Exception as e:
        print(f"Error creating Iceberg table: {e}")
        raise

def read_iceberg_table(table_name: str):
    """Read the Iceberg table back to a local PyArrow table"""
    try:
        # Create catalog
        catalog = get_catalog()
        
        # Load the table
        table = catalog.load_table((DATABASE_NAME, table_name))
        
        # Read all data from the table
        arrow_table = table.scan().to_arrow()
        print(f"Successfully read table {DATABASE_NAME}.{table_name}")
        print("\nTable Schema:")
        print(arrow_table.schema)
        print("\nFirst few rows:")
        print(arrow_table.slice(0, 5).to_pandas())
        
        return arrow_table
        
    except Exception as e:
        print(f"Error reading Iceberg table: {e}")
        raise

if __name__ == "__main__":
    # Comment out these lines if you just want to read the table
    upload_to_s3()
    create_iceberg_table('test_table')
    create_iceberg_table('test_table_10', 10)
    
    # Read the table back
    print("\nReading back the table:")
    arrow_table = read_iceberg_table('test_table') 
    arrow_table = read_iceberg_table('test_table_10') 