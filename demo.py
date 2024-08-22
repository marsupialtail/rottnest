import rottnest
import rottnest.internal as internal

# rottnest.index_file_bm25("example_data/0.parquet", "body", "index0")
# rottnest.index_file_bm25("example_data/1.parquet", "body", "index1")
# rottnest.merge_index_bm25("merged_index", ["index0", "index1"])
# result = rottnest.search_index_bm25(["merged_index"], "cell phones", K = 10,query_expansion = "openai", reader_type = "aws")
# print(result)

def substring_test():
    # internal.index_files_substring(["example_data/a.parquet"], "text", "index0", token_skip_factor = 10)
    # internal.index_files_substring(["example_data/b.parquet"], "text", "index1", token_skip_factor = 10)
    # internal.merge_index_substring("merged_index", ["index0", "index1"])
    result = internal.search_index_substring(["index0"], 
                                            "One step you have to remember not to skip is to use Disk Utility to partition the SSD as GUID partition scheme HFS+ before doing the clone.", 
                                            K = 10, token_viable_limit= 1, sample_factor = 10)
    print(result)

# table1 = polars.read_parquet("uuid_data/a.parquet")
# table2 = polars.read_parquet("uuid_data/b.parquet")
# from deltalake import WriterProperties
# write_deltalake("uuid_data_delta", table1.to_arrow(), mode = "append", engine = 'rust', writer_properties = WriterProperties(data_page_size_limit=1000000, compression = 'ZSTD'))
# write_deltalake("uuid_data_delta", table2.to_arrow(), mode = "append", engine = 'rust', writer_properties = WriterProperties(data_page_size_limit=1000000, compression = 'ZSTD'))
# rottnest.index_delta("uuid_data_delta", "hashes", "uuid_rottnest_index", "uuid")

def uuid_test():
    # internal.index_files_uuid(["uuid_data/a.parquet"], "hashes", "index0")
    # internal.index_files_uuid(["uuid_data/b.parquet"], "hashes", "index1")

    internal.index_files_uuid(["s3://txhashesbenchmark/0.parquet"], "hashes", "index0")
    internal.index_files_uuid(["s3://txhashesbenchmark/1.parquet"], "hashes", "index1")

    internal.merge_index_uuid("merged_index", ["index0", "index1"])
    result = internal.search_index_uuid(["merged_index"], "93b9f88dd22cb168cbc45000fcb05042cd1fc4b5602a56e70383fa26d33d21b08d004d78a7c97a463331da2da64e88f5546367e16e5fd2539bb9b8796ffffc7f", K = 10)
    print(result)

substring_test()

# result = rottnest.search_index_uuid(["merged_index"], "650243a9024fe6595fa953e309c722c225cb2fae1f70c74364917eb901bcdce1f9a878d22345a8576a201646b6da815ebd6397cfd313447ee3a548259f63825a", K = 10)
# print(result)
# result = rottnest.search_index_uuid(["index0", "index1"], "650243a9024fe6595fa953e309c722c225cb2fae1f70c74364917eb901bcdce1f9a878d22345a8576a201646b6da815ebd6397cfd313447ee3a548259f63825a", K = 10)
# print(result)
# result = rottnest.search_index_uuid(["merged_index"], "32b8fd4d808300b97b2dff451cba4185faee842a1248c84c1ab544632957eb8904dccb5880f0d4a9a7317c3a4490b0222e4deb5047abc1788665a46176009a07", K = 10)
# print(result)