import polars
import uuid
from . import internal
"""
the schema of the metadata table will be:

index_file: str | covered_parquet_files: list[str] | stats: json?

We seek to maintain the invariant that each parquet file is covered by only one index file.
"""

def index_delta(table: str, column: str, index_dir: str, type: str, index_impl = 'physical', extra_configs = {}):

    from deltalake import DeltaTable, write_deltalake
    from deltalake._internal import TableNotFoundError

    assert type in {"uuid", "substring", "vector"}
    assert index_impl in {"physical", "virtual"}
    if index_impl == "virtual":
        raise NotImplementedError("virtual index not yet implemented") 

    index_dir = index_dir.rstrip("/")
    metadata_table_dir = f"{index_dir}/metadata_table"

    main_table = DeltaTable(table)
    existing_parquet_files = polars.from_dict({"covered_parquet_files": main_table.file_uris()})

    try:
        index_table = polars.from_arrow(DeltaTable(metadata_table_dir).to_pyarrow_table())
        
        unindexed_parquet_files = existing_parquet_files.join(
            index_table.select(['covered_parquet_files']).explode('covered_parquet_files'), on = 'covered_parquet_files', how = 'anti')

    except TableNotFoundError:
        
        unindexed_parquet_files = existing_parquet_files
    
    unindexed_parquet_files = unindexed_parquet_files['covered_parquet_files'].to_list()
    
    index_name = uuid.uuid4().hex
    
    try:
        if type == "uuid":
            internal.index_files_uuid(unindexed_parquet_files, column, f"{index_dir}/{index_name}")
        elif type == "substring":
            internal.index_files_substring(unindexed_parquet_files, column, f"{index_dir}/{index_name}")
        elif type == "vector":
            internal.index_files_vector(unindexed_parquet_files, column, f"{index_dir}/{index_name}")
    except:
        raise Exception("Underlying data lake changed. Please retry this operation. I am idempotent.")

    # now go on and commit!
    try:
        table = polars.from_dict({"index_file": [index_name], "covered_parquet_files": [unindexed_parquet_files]})
        print(table)
        write_deltalake(metadata_table_dir, table.to_arrow(), mode = "append")
    except:
        raise Exception("commit failure. Please retry this operation. I am idempotent.")

# we should deprecate the type argument, and figure out the type automatically.
def search_delta(table: str, index_dir: str, query,  type: str, K: int, snapshot : int | None = None, extra_configs = {}):

    from deltalake import DeltaTable, write_deltalake
    from deltalake._internal import TableNotFoundError

    assert type in {"uuid", "substring", "vector"}

    main_table = DeltaTable(table)
    if snapshot is not None:
        main_table.load_as_version(snapshot)

    existing_parquet_files = set(main_table.file_uris())
    
    index_dir = index_dir.rstrip("/")
    metadata_table_dir = f"{index_dir}/metadata_table"

    selected_indices = []
    uncovered_parquets = set(existing_parquet_files)

    try:
        metadata = polars.from_arrow(DeltaTable(metadata_table_dir).to_pyarrow_table())
        # convert it to a dictionary
        metadata = {row['index_file']: row['covered_parquet_files'] for _, row in metadata.iterrows()}
        
        while True:
            overlap_to_index_file = {len(existing_parquet_files.intersection(v)): k for k, v in metadata.items()}
            max_overlap = max(overlap_to_index_file.keys())
            if max_overlap == 0:
                break
            index_file = overlap_to_index_file[max_overlap]
            selected_indices.append(index_file)
            uncovered_parquets = uncovered_parquets.difference(set(metadata[index_file]))
            del metadata[index_file]

    except TableNotFoundError:
        pass
    
    
    if type == "uuid":
        index_search_results = internal.search_lava_uuid([f"{index_name}.lava" for index_name in selected_indices], query, K, "aws")
    elif type == "substring":
        index_search_results = internal.search_lava_substring([f"{index_name}.lava" for index_name in selected_indices], query, K, "aws")
    elif type == "vector":
        index_search_results = internal.search_lava_vector([f"{index_name}.lava" for index_name in selected_indices], query, K, "aws")