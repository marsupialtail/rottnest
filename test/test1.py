from rottnest.iceberg import index_iceberg, search_iceberg, compact_iceberg_indices, vacuum_iceberg_indices
from iceberg_utils import delete_if_exists
import os
from rottnest import SubstringIndex

# Read bucket name from environment variable, falling back to default if not set
BUCKET_NAME = os.environ.get('ROTTNEST_BUCKET_NAME', None)
assert BUCKET_NAME is not None, "ROTTNEST_BUCKET_NAME environment variable must be set"
DATABASE_NAME = 'rottnest'    # Replace with your Glue database name
S3_PREFIX = f's3://{BUCKET_NAME}/iceberg_data/'

index = SubstringIndex(char_index = True)

def test_single():
    TABLE_NAME = 'test_table'
    INDEX_TABLE_NAME = 'test_index_4'
    delete_if_exists(f"{DATABASE_NAME}.{INDEX_TABLE_NAME}")

    index_iceberg(
        table = f"{DATABASE_NAME}.{TABLE_NAME}", 
        column = "text", 
        index_table = f"{DATABASE_NAME}.{INDEX_TABLE_NAME}", 
        iceberg_location = S3_PREFIX,
        index_prefix = f"s3://{BUCKET_NAME}/rottnest_data/", 
        index = index)

    search_iceberg(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
        column = "text", 
        index_table = f"{DATABASE_NAME}.{INDEX_TABLE_NAME}", 
        query = "We want to welcome you", 
        index = index,
        K = 10)

def test_compaction():

    TABLE_NAME = 'test_table_10'

    # delete_if_exists(f"{DATABASE_NAME}.test_index_10_2")

    # index_iceberg(
    #     table = f"{DATABASE_NAME}.{TABLE_NAME}", 
    #     column = "text", 
    #     index_table = f"{DATABASE_NAME}.test_index_10_2", 
    #     iceberg_location = S3_PREFIX,
    #     index_prefix = f"s3://{BUCKET_NAME}/rottnest_data_10/", 
    #     index = index,
    #     binpack_row_threshold = 10)

    # search_iceberg(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
    #     column = "text", 
    #     index_table = f"{DATABASE_NAME}.test_index_10_2", 
    #     query = "We want to welcome you", 
    #     index = index,
    #     K = 10)

    # compact_iceberg_indices(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
    #     column = "text", 
    #     index_table = f"{DATABASE_NAME}.test_index_10_2", 
    #     index_prefix = f"s3://{BUCKET_NAME}/rottnest_data_10/", 
    #     index = index,
    #     binpack_row_threshold = 100000)

    # search_iceberg(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
    #     column = "text", 
    #     index_table = f"{DATABASE_NAME}.test_index_10_2", 
    #     query = "We want to welcome you", 
    #     index = index,
    #     K = 10)

    vacuum_iceberg_indices(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
        index_table = f"{DATABASE_NAME}.test_index_10_2", 
        index_prefix = f"s3://{BUCKET_NAME}/rottnest_data_10/", 
        history = 0)

# test_single()
test_compaction()