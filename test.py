import pyarrow
import polars
import rottnest_rs
import pyarrow.parquet as pq
import numpy as np

def basic_test():
    a = polars.from_dict({"a":["你是一只小猪","hello you are happy", "hello, et tu, brutes?"]}).to_arrow()
    b = polars.from_dict({"a":["cmn","eng", "bump"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([1,2,3]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.build_lava_natural_language("output_file.bin", text, uid, language))
    print("result:", rottnest_rs.search_lava("output_file.bin", "hello"))

def merge_test():

    a = polars.from_dict({"a":["a","d", "h", "z", "f"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en", "bump","bump","f"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([1,2,3,4, 5]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.rottnest_rs.build_lava_natural_language("1.lava", text, uid, language))

    a = polars.from_dict({"a":["b","e"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([4,5]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.rottnest_rs.build_lava_natural_language("2.lava", text, uid, language))

    a = polars.from_dict({"a":["c","f"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([7,8]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.rottnest_rs.build_lava_natural_language("3.lava", text, uid, language))

    print("search result", rottnest_rs.rottnest_rs.search_lava("1.lava", "d"))

    rottnest_rs.rottnest_rs.merge_lava("merged.lava", ["1.lava", "2.lava", "3.lava"], [0,10,20])

    assert rottnest_rs.rottnest_rs.search_lava("merged.lava", "d") == [2]
    assert rottnest_rs.rottnest_rs.search_lava("merged.lava", "f") == [5,20] # the second one will be 20 because short uid list

# basic_test()
merge_test()





# index_file_natural_language("train.parquet","raw_content")

# Path: test.py
# Compare this snippet from python/rottnest_rs/make_lava.py:
# from rottnest_rs import tokenize
# import pyarrow
       
index_name = "test"
rottnest_rs.index_file_natural_language("train.parquet","raw_content", name = index_name)
rottnest_rs.search_index_natural_language(f"{index_name}", "Helsinki")

# index_name = "content_split"
# rottnest_rs.index_file_natural_language("ecom_orig.parquet","content_split", name = index_name)
# result = rottnest_rs.search_index_natural_language(f"{index_name}", "helsinki")
# print(result)

index_name = "bump1"
import time
start = time.time()
rottnest_rs.index_file_natural_language("s3://maas-data/data_for_mass_wangqian/en/part-03060-21668627-949b-4858-97ce-a4b0f4fc2df4-c000.gz.parquet","text", name = index_name)
print("index time", time.time() - start)
start = time.time()
rottnest_rs.search_index_natural_language(index_name, "Publish")
print("search time", time.time() - start)

# index_name = "bump2"
# rottnest_rs.index_file_natural_language("data/part-03062-21668627-949b-4858-97ce-a4b0f4fc2df4-c000.gz.parquet","text", name = index_name)
# rottnest_rs.search_index_natural_language(index_name, "C1X")
# rottnest_rs.merge_index_natural_language("merged", ["bump1", "bump2"])
# result = rottnest_rs.search_index_natural_language("merged", "C1X")
# search_index_natural_language("bump2", "Eric")