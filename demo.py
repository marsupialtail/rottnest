import rottnest

# rottnest.index_file_bm25("example_data/0.parquet", "body", "index0")
# rottnest.index_file_bm25("example_data/1.parquet", "body", "index1")
# rottnest.merge_index_bm25("merged_index", ["index0", "index1"])
# result = rottnest.search_index_bm25(["merged_index"], "cell phones", K = 10,query_expansion = "openai", reader_type = "aws")
# print(result)

# rottnest.index_file_substring("example_data/real.parquet", "text", "index0", token_skip_factor = 3)
# rottnest.index_file_substring("example_data/1.parquet", "body", "index1")
# rottnest.merge_index_substring("merged_index", ["index0", "index1"])
# result = rottnest.search_index_substring(["index0"], "did fake", K = 10)
# print(result)

# rottnest.index_file_uuid("0.parquet", "hashes", "index0")
result = rottnest.search_index_uuid(["index0"], "32b8fd4d808300b97b2dff451cba4185faee842a1248c84c1ab544632957eb8904dccb5880f0d4a9a7317c3a4490b0222e4deb5047abc1788665a46176009a07", K = 10)
print(result)