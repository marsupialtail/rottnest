from rottnest.iceberg import index_iceberg, search_iceberg, compact_iceberg_indices

BUCKET_NAME = 'rottnest-iceberg-test'  # Replace with your S3 bucket name
DATABASE_NAME = 'rottnest'    # Replace with your Glue database name
S3_PREFIX = f's3://{BUCKET_NAME}/iceberg_data/'

def test_single():
    TABLE_NAME = 'test_table'

    index_iceberg(
        table = f"{DATABASE_NAME}.{TABLE_NAME}", 
        column = "text", 
        index_table = f"{DATABASE_NAME}.test_index_4", 
        iceberg_location = S3_PREFIX,
        index_prefix = "s3://rottnest-iceberg-test/rottnest_data/", 
        type = "substring", 
        index_impl = "physical",
        extra_configs = {'char_index': True})

    search_iceberg(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
        column = "text", 
        index_table = f"{DATABASE_NAME}.test_index_4", 
        query = "We want to welcome you", 
        type = "substring", 
        K = 10, 
        extra_configs = {'char_index': True})

TABLE_NAME = 'test_table_10'

# index_iceberg(
#     table = f"{DATABASE_NAME}.{TABLE_NAME}", 
#     column = "text", 
#     index_table = f"{DATABASE_NAME}.test_index_10_2", 
#     iceberg_location = S3_PREFIX,
#     index_prefix = "s3://rottnest-iceberg-test/rottnest_data_10/", 
#     type = "substring", 
#     index_impl = "physical",
#     binpack_row_threshold = 10,
#     extra_configs = {'char_index': True})

# search_iceberg(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
#     column = "text", 
#     index_table = f"{DATABASE_NAME}.test_index_10_2", 
#     query = "We want to welcome you", 
#     type = "substring", 
#     K = 10, 
#     extra_configs = {'char_index': True})

compact_iceberg_indices(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
    column = "text", 
    index_table = f"{DATABASE_NAME}.test_index_10_2", 
    index_prefix = "s3://rottnest-iceberg-test/rottnest_data_10/", 
    type = 'substring', 
    binpack_row_threshold = 100000,
    extra_configs = {'char_index': True})

search_iceberg(table = f"{DATABASE_NAME}.{TABLE_NAME}", 
    column = "text", 
    index_table = f"{DATABASE_NAME}.test_index_10_2", 
    query = "We want to welcome you", 
    type = "substring", 
    K = 10, 
    extra_configs = {'char_index': True})