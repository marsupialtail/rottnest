from rottnest import iceberg
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

    config = iceberg.IcebergConfig(
        table = f"{DATABASE_NAME}.{TABLE_NAME}", 
        column = "text", 
        index_table = f"{DATABASE_NAME}.{INDEX_TABLE_NAME}", 
        iceberg_location = S3_PREFIX,
        index_prefix = f"s3://{BUCKET_NAME}/rottnest_data/", 
    )

    iceberg.index_iceberg(config = config, index = index)

    iceberg.search_iceberg(config = config, query = "We want to welcome you", index = index, K = 10)

def test_compaction():

    TABLE_NAME = 'test_table_10'
    INDEX_TABLE_NAME = 'test_index_10_2'

    delete_if_exists(f"{DATABASE_NAME}.test_index_10_2")

    config = iceberg.IcebergConfig(
        table = f"{DATABASE_NAME}.{TABLE_NAME}", 
        column = "text", 
        index_table = f"{DATABASE_NAME}.{INDEX_TABLE_NAME}", 
        iceberg_location = S3_PREFIX,
        index_prefix = f"s3://{BUCKET_NAME}/rottnest_data/", 
        index_timeout = 1000,
        binpack_row_threshold=10,
    )

    iceberg.index_iceberg(
        config = config,
        index = index,
    )

    iceberg.search_iceberg(config = config, query = "We want to welcome you", index = index, K = 10)

    config.binpack_row_threshold = 10_000

    iceberg.compact_iceberg_indices(config = config, index = index)

    iceberg.search_iceberg(config = config, query = "We want to welcome you", index = index, K = 10)

    config.index_timeout = 1

    iceberg.vacuum_iceberg_indices(config = config, history = 0)

# test_single()
test_compaction()