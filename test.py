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

    rottnest_rs.merge_lava("merged.lava", ["1.lava", "2.lava", "3.lava"], [0,10,20])

    assert rottnest_rs.search_lava("merged.lava", "d") == [2]
    assert rottnest_rs.search_lava("merged.lava", "f") == [5,20] # the second one will be 20 because short uid list

# basic_test()
# merge_test()

from typing import List, Optional
import uuid

def index_file_natural_language(file_path: List[str], column_name: str, name: Optional[str]):

    arr, layout = rottnest_rs.get_parquet_layout(column_name, file_path)
    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) + 1

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows) + 1),
            "file_path": [file_path] * (len(data_page_num_rows) + 1),
            "column_name": [column_name] * (len(data_page_num_rows) + 1),
            "data_page_offsets": [-1] + layout.data_page_offsets,
            "data_page_sizes": [-1] + layout.data_page_sizes,
            "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
            "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
        }
    )

    name = uuid.uuid4().hex if name is None else name

    file_data.write_parquet(f"{name}.meta")
    print(rottnest_rs.build_lava_natural_language(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64))))

def merge_index_natural_language(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1

    # first read the metadata files and merge those
    metadatas = [polars.read_parquet(f"{name}.meta")for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]

    rottnest_rs.merge_lava(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")

def search_index_natural_language(index_name, query, mode = "exact"):

    assert mode in {"exact", "substring"}

    metadata_file = f"{index_name}.meta"
    index_file = f"{index_name}.lava"
    uids = polars.from_dict({"uid":rottnest_rs.search_lava(index_file, query if mode == "substring" else f"^{query}$")})
    
    print(uids)
    if len(uids) == 0:
        return
    
    metadata = polars.read_parquet(metadata_file).join(uids, on = "uid")
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"

    # now we need to do something special about -1 values that indicate we have to search the entire file

    column_name = metadata["column_name"].unique()[0]
    result = rottnest_rs.search_indexed_pages(query, column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list())
    print([item.matched for item in result])



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
# search_index_natural_language(index_name, "Publish")
# index_name = "bump2"
# index_file_natural_language("data/part-03062-21668627-949b-4858-97ce-a4b0f4fc2df4-c000.gz.parquet","text", name = index_name)
# search_index_natural_language(index_name, "C1X")

# merge_index_natural_language("merged", ["bump1", "bump2"])
search_index_natural_language("merged", "disparity")
# search_index_natural_language("bump2", "Eric")