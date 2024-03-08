import pyarrow
import polars
import rottnest
import pyarrow.parquet as pq
import numpy as np
import logging

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

def basic_test():
    a = polars.from_dict({"a":["你是一只小猪","hello you are happy", "hello, et tu, brutes?"]}).to_arrow()
    b = polars.from_dict({"a":["cmn","eng", "bump"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([1,2,3]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest.build_lava_bm25("output_file.bin", text, uid, language))
    print("result:", rottnest.search_lava("output_file.bin", "hello"))

def merge_test():

    a = polars.from_dict({"a":["a","d", "h", "z", "f"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en", "bump","bump","f"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([1,2,3,4, 5]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest.rottnest.build_lava_bm25("1.lava", text, uid, language))

    a = polars.from_dict({"a":["b","e"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([4,5]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest.rottnest.build_lava_bm25("2.lava", text, uid, language))

    a = polars.from_dict({"a":["c","f"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([7,8]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest.rottnest.build_lava_bm25("3.lava", text, uid, language))

    print("search result", rottnest.rottnest.search_lava("1.lava", "d"))

    rottnest.rottnest.merge_lava_bm25("merged.lava", ["1.lava", "2.lava", "3.lava"], [0,10,20])

    assert rottnest.rottnest.search_lava("merged.lava", "d") == [2]
    assert rottnest.rottnest.search_lava("merged.lava", "f") == [5,20] # the second one will be 20 because short uid list

# basic_test()
# merge_test()

rottnest.index_file_bm25("msmarco/chunk_1.parquet","body", name = "bump1")
# rottnest.index_file_bm25("msmarco/chunk_2.parquet","body", name = "bump2")
# rottnest.merge_index_bm25("merged", ["bump1", "bump2"])
# result = rottnest.search_index_bm25(["merged"], "cell phones", 50)
# print(result)