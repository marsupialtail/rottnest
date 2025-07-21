from rottnest.indices.logcloud_index import index_files_logcloud, search_index_logcloud
index_files_logcloud(['0.parquet', '1.parquet'], 'log', name = 'bump')
print(search_index_logcloud(['bump'], 'broadcast_6_piece219', 10))