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
    print("result:", rottnest_rs.search_lava("output_file.bin", "hello"))

def merge_test():

    a = polars.from_dict({"a":["a","d", "h", "z", "f"]}).to_arrow()
    b = polars.from_dict({"a":["cn","en", "bump","bump","f"]}).to_arrow()
    text = a["a"].combine_chunks().cast(pyarrow.string())
    uid = pyarrow.array([1,2,3,4, 5]).cast(pyarrow.uint64())
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

    print("search result", rottnest_rs.search_lava("1.lava", "d"))

    rottnest_rs.merge_lava("merged.lava", ["1.lava", "2.lava", "3.lava"])

    assert rottnest_rs.search_lava("merged.lava", "d") == [2]

# basic_test()
# merge_test()

from typing import List, Optional
import numpy as np
import uuid

def index_file_natural_language(file_path: List[str], column_name: str, name: Optional[str]):

    arr, layout = rottnest_rs.get_parquet_layout(column_name, file_path)
    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows)

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows)),
            "data_page_offsets": layout.data_page_offsets,
            "data_page_sizes": layout.data_page_sizes,
            "dictionary_page_sizes": layout.dictionary_page_sizes,
            "row_groups": np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages),
        }
    )

    name = uuid.uuid4().hex if name is None else name

    file_data.write_parquet(f"{name}.parquet")
    print(rottnest_rs.build_lava_natural_language(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64))))

def search_index_natural_language(metadata_path: str, index_path: str, query):

    uids = rottnest_rs.search_lava(index_path, query)
    print(uids)
    # rottnest_rs.search_indexed_pages(query, )




# index_file_natural_language("train.parquet","raw_content")

# Path: test.py
# Compare this snippet from python/rottnest_rs/make_lava.py:
# from rottnest_rs import tokenize
# import pyarrow
       
# index_name = "test"
# index_file_natural_language("train.parquet","raw_content", name = index_name)
# search_index_natural_language(f"{index_name}.parquet", f"{index_name}.lava", "Helsinki")

# index_name = "content_split"
# index_file_natural_language("ecom_orig.parquet","content_split", name = index_name)
# search_index_natural_language(f"{index_name}.parquet", f"{index_name}.lava", "helsinki")

# index_name = "bump1"
# index_file_natural_language("data/part-03060-21668627-949b-4858-97ce-a4b0f4fc2df4-c000.gz.parquet","text", name = index_name)
# search_index_natural_language(f"{index_name}.parquet", f"{index_name}.lava", "C1X")
# index_name = "bump2"
# index_file_natural_language("data/part-03062-21668627-949b-4858-97ce-a4b0f4fc2df4-c000.gz.parquet","text", name = index_name)
# search_index_natural_language(f"{index_name}.parquet", f"{index_name}.lava", "C1X")

rottnest_rs.merge_lava("merged.lava", ["bump1.lava", "bump2.lava"])