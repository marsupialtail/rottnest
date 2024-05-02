import rottnest

rottnest.index_file_bm25("example_data/0.parquet", "body", "index0")
rottnest.index_file_bm25("example_data/1.parquet", "body", "index1")
rottnest.merge_index_bm25("merged_index", ["index0", "index1"])
result = rottnest.search_index_bm25(["merged_index"], "cell phones", K = 10,query_expansion = "bge", reader_type = "aws")
print(result)

rottnest.index_file_substring("example_data/0.parquet", "body", "index0")
rottnest.index_file_substring("example_data/1.parquet", "body", "index1")
rottnest.merge_index_substring("merged_index", ["index0", "index1"])
result = rottnest.search_index_substring(["merged_index"], "cell phones", K = 10)
print(result)
