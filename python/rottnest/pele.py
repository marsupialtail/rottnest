import pyarrow
import pyarrow.parquet as pq
from typing import List
import rottnest.rottnest as rottnest
from typing import List, Optional
import uuid
import polars
import numpy as np

def index_file_natural_language(file_path: List[str], column_name: str, name: Optional[str]):

    arr, layout = rottnest.get_parquet_layout(column_name, file_path)
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
    print(rottnest.build_lava_natural_language(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64))))

def merge_index_natural_language(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1

    # first read the metadata files and merge those
    metadatas = [polars.read_parquet(f"{name}.meta")for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]

    rottnest.merge_lava(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")

def search_index_natural_language(index_name, query, mode = "exact"):

    assert mode in {"exact", "substring"}

    metadata_file = f"{index_name}.meta"
    index_file = f"{index_name}.lava"
    uids = polars.from_dict({"uid":rottnest.search_lava(index_file, query if mode == "substring" else f"^{query}$")})
    
    print(uids)
    if len(uids) == 0:
        return
    
    metadata_orig = polars.read_parquet(metadata_file)
    metadata = metadata_orig.join(uids, on = "uid")
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"
    column_name = metadata["column_name"].unique()[0]

    # now we need to do something special about -1 values that indicate we have to search the entire file

    expanded = metadata.filter(polars.col("row_groups") == -1)\
        .select(["file_path"])\
        .join(metadata_orig, on = "file_path")\
        .filter(polars.col("row_groups") != -1)\
        .select(["uid", "file_path", "column_name", "data_page_offsets", "data_page_sizes", "dictionary_page_sizes", "row_groups"])
        
    metadata = polars.concat([metadata.filter(polars.col("row_groups") != -1), expanded])

    result = rottnest.search_indexed_pages(query, column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list())
    return result