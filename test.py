import pyarrow
import polars
import rottnest_rs
import pyarrow.parquet as pq

def basic_test():
    a = polars.from_dict({"a":["你是一只小猪","hello you are happy", "hello, et tu, brutes?"]}).to_arrow()
    b = polars.from_dict({"a":["cmn","eng", "bump"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([1,2,3]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.build_lava_natural_language("output_file.bin", text, uid, language))
    print(rottnest_rs.search_lava("output_file.bin", "hello"))

def merge_test():

    a = polars.from_dict({"a":["a","d", "h"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en", "bump"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([1,2,3]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.build_lava_natural_language("1.lava", text, uid, language))

    a = polars.from_dict({"a":["b","e"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([4,5]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.build_lava_natural_language("2.lava", text, uid, language))

    a = polars.from_dict({"a":["c","f"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([7,8]).cast(pyarrow.uint64())
    language = b["a"].combine_chunks().cast(pyarrow.string())
    print(rottnest_rs.build_lava_natural_language("3.lava", text, uid, language))

    rottnest_rs.merge_lava("merged.lava", ["1.lava", "2.lava", "3.lava"])

    print(rottnest_rs.search_lava("merged.lava", "d"))

# basic_test()
# merge_test()
    
def index_natural_language(file_paths, column_name):

    # first make sure that the column exists in each file, and is the same index (restriction for now)

    column_index = None
    for file in file_paths:
        pq_file = pq.ParquetFile(file)
