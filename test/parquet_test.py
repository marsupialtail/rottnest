from rottnest.backends.parquet import ParquetConfig, index_parquet, search_parquet, compact_parquet_indices, vacuum_parquet_indices
from rottnest.indices.substring_index import SubstringIndex
import os
from parquet_utils import cleanup_parquet_table

# Read bucket name from environment variable, falling back to default if not set
BUCKET_NAME = os.environ.get('ROTTNEST_PARQUET_TEST_PREFIX', None)
assert BUCKET_NAME is not None, "ROTTNEST_PARQUET_TEST_PREFIX environment variable must be set"
S3_PREFIX = f's3://{BUCKET_NAME}/parquet_data'

index = SubstringIndex(char_index=True)

def test_single():
    # Clean up any existing files and create new test data
    cleanup_parquet_table(prefix=f's3://{BUCKET_NAME}/parquet_data/indices/')
    config = ParquetConfig(
        data_prefix=S3_PREFIX,
        column="text",
        index_prefix=f"{S3_PREFIX}/indices",
        binpack_row_threshold=10000
    )

    index_parquet(config=config, index=index)
    search_parquet(config=config, query="We want to welcome you", index=index, K=10)

def test_compaction():
    # Clean up any existing files and create new test data
    cleanup_parquet_table(prefix=f's3://{BUCKET_NAME}/parquet_data/indices/')
    config = ParquetConfig(
        data_prefix=S3_PREFIX,
        column="text",
        index_prefix=f"{S3_PREFIX}/indices",
        binpack_row_threshold=10,  # Small threshold to force multiple indices
        index_timeout=1000
    )

    # Initial indexing with small threshold
    index_parquet(config=config, index=index)
    search_parquet(config=config, query="We want to welcome you", index=index, K=10)

    # Increase threshold and compact
    config.binpack_row_threshold = 10000
    compact_parquet_indices(config=config, index=index)
    search_parquet(config=config, query="We want to welcome you", index=index, K=10)

    # Test vacuum with short timeout
    config.index_timeout = 1
    vacuum_parquet_indices(config=config)

if __name__ == "__main__":
    test_single()
    # test_compaction() 