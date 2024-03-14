import pyarrow
import pyarrow.parquet as pq
from typing import List
import rottnest.rottnest as rottnest
from typing import List, Optional
import uuid
import polars
import numpy as np
from tqdm import tqdm
import hashlib

def index_file_bm25(file_path: List[str], column_name: str, name = None, tokenizer_file = None):

    arr, layout = rottnest.get_parquet_layout(column_name, file_path)
    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) + 1

    # Code tries to compute the starting row offset of each page in its row group.
    # The following three lines are definitely easier to read than to write.

    x = np.cumsum(np.hstack([[0],layout.data_page_num_rows[:-1]]))
    y = np.repeat(x[np.cumsum(np.hstack([[0],layout.row_group_data_pages[:-1]]))], layout.row_group_data_pages)
    page_row_offsets_in_row_group = x - y

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows) + 1),
            "file_path": [file_path] * (len(data_page_num_rows) + 1),
            "column_name": [column_name] * (len(data_page_num_rows) + 1),
            "data_page_offsets": [-1] + layout.data_page_offsets,
            "data_page_sizes": [-1] + layout.data_page_sizes,
            "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
            "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
            "page_row_offset_in_row_group": np.hstack([[-1], page_row_offsets_in_row_group])
        }
    )

    name = uuid.uuid4().hex if name is None else name

    file_data.write_parquet(f"{name}.meta")
    print(rottnest.build_lava_bm25(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64)), tokenizer_file))

def index_file_substring(file_path: List[str], column_name: str, name = None, tokenizer_file = None):

    arr, layout = rottnest.get_parquet_layout(column_name, file_path)
    data_page_num_rows = np.array(layout.data_page_num_rows)
    uid = np.repeat(np.arange(len(data_page_num_rows)), data_page_num_rows) + 1

    x = np.cumsum(np.hstack([[0],layout.data_page_num_rows[:-1]]))
    y = np.repeat(x[np.cumsum(np.hstack([[0],layout.row_group_data_pages[:-1]]))], layout.row_group_data_pages)
    page_row_offsets_in_row_group = x - y

    file_data = polars.from_dict({
            "uid": np.arange(len(data_page_num_rows) + 1),
            "file_path": [file_path] * (len(data_page_num_rows) + 1),
            "column_name": [column_name] * (len(data_page_num_rows) + 1),
            "data_page_offsets": [-1] + layout.data_page_offsets,
            "data_page_sizes": [-1] + layout.data_page_sizes,
            "dictionary_page_sizes": [-1] + layout.dictionary_page_sizes,
            "row_groups": np.hstack([[-1] , np.repeat(np.arange(layout.num_row_groups), layout.row_group_data_pages)]),
            "page_row_offset_in_row_group": np.hstack([[-1], page_row_offsets_in_row_group])
        }
    )
    name = uuid.uuid4().hex if name is None else name

    file_data.write_parquet(f"{name}.meta")
    print(rottnest.build_lava_substring(f"{name}.lava", arr, pyarrow.array(uid.astype(np.uint64)), tokenizer_file))

def merge_index_bm25(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1

    # first read the metadata files and merge those
    metadatas = [polars.read_parquet(f"{name}.meta")for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    rottnest.merge_lava_bm25(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")

def merge_index_substring(new_index_name: str, index_names: List[str]):
    assert len(index_names) > 1

    # first read the metadata files and merge those
    metadatas = [polars.read_parquet(f"{name}.meta")for name in index_names]
    metadata_lens = [len(metadata) for metadata in metadatas]
    offsets = np.cumsum([0] + metadata_lens)[:-1]
    metadatas = [metadata.with_columns(polars.col("uid") + offsets[i]) for i, metadata in enumerate(metadatas)]
    rottnest.merge_lava_substring(f"{new_index_name}.lava", [f"{name}.lava" for name in index_names], offsets)
    polars.concat(metadatas).write_parquet(f"{new_index_name}.meta")

def query_expansion_llm(tokenizer_vocab: List[str], query: str, model = "text-embedding-3-large", expansion_tokens = 20):
    import os, pickle
    try:
        import faiss
        from openai import OpenAI
    except:
        raise Exception("LLM based query expansion requires installation of FAISS and OpenAI python packages. pip install faiss-cpu openai")

    assert type(query) == str, "query must be string. If you have a list of keywords, concatenate them with spaces."

    cache_dir = os.path.expanduser('~/.cache')
    # make a subdirectory rottnest under cache_dir if it's not there already
    cache_dir = os.path.join(cache_dir, 'rottnest')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    tokenizer_hash = hashlib.sha256(("".join(tokenizer_vocab[::1000])).encode('utf-8')).hexdigest()[:8]

    # check if the tokenizer_embeddings.pkl file exists in the cache directory
    tokenizer_embeddings_path = os.path.join(cache_dir, f"tokenizer_embeddings_{tokenizer_hash}_{model}.pkl")
    client = OpenAI()

    if not os.path.exists(tokenizer_embeddings_path):
        print("First time doing LLM query expansion with this tokenizer, need to compute tokenizer embeddings.")
        all_vecs = []
        tokenizer_vocab = [tok  if tok else "[]" for tok in tokenizer_vocab]
        for i in tqdm(range(0, len(tokenizer_vocab), 1000)):
            results = client.embeddings.create(input = tokenizer_vocab[i:i+1000], model = model)
            vecs = np.vstack([results.data[i].embedding for i in range(len(results.data))])
            all_vecs.append(vecs)

        pickle.dump({"words": tokenizer_vocab, "vecs": np.vstack(all_vecs)}, 
                    open(os.path.join(cache_dir, f"tokenizer_embeddings_{tokenizer_hash}_{model}.pkl"), "wb"))
    
    embeddings = pickle.load(open(tokenizer_embeddings_path, "rb"))

    tokens = embeddings['words']
    db_vectors = embeddings['vecs']

    index = faiss.IndexFlatL2(db_vectors.shape[1])  # Use the L2 distance for similarity
    index.add(db_vectors) 
    query_vectors = np.expand_dims(np.array(client.embeddings.create(input = query, model = model).data[0].embedding), 0)
    distances, indices = index.search(query_vectors, expansion_tokens)  # Perform the search
    print("Expanded tokens: ", [tokens[i] for i in indices[0]])

    return [tokens[i] for i in indices[0]], list(indices[0]), list(1 / (distances[0] + 1))

def query_expansion_keyword(tokenizer_vocab: List[str], query: str):

    # simply check what words in tokenizer_vocab appears in query, and the weight is how many times it appears
    token_ids = []
    tokens = []
    weights = []
    for i, token in tokenizer_vocab:
        if token in query:
            token_ids.append(i)
            tokens.append(token)
            weights.append(query.count(token))

    print("Expanded tokens: ", tokens)
    return tokens, token_ids, weights

def search_index_substring(indices: List[str], query: str, K: int):
    
    index_search_results = rottnest.search_lava_substring([f"{index_name}.lava" for index_name in indices], query, K)
    print(index_search_results)

    if len(index_search_results) == 0:
        return None

    uids = polars.from_dict({"file_id": [i[0] for i in index_search_results], "uid": [i[1] for i in index_search_results]})

    metadatas = [polars.read_parquet(f"{index_name}.meta").with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, index_name in enumerate(indices)]
    metadata = polars.concat(metadatas)
    metadata = metadata.join(uids, on = ["file_id", "uid"])
    
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"
    column_name = metadata["column_name"].unique()[0]

    result = pyarrow.chunked_array(rottnest.read_indexed_pages(column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list()))
    result = pyarrow.table([result], names = ["text"])
    
    return polars.from_arrow(result).filter(polars.col("text").str.to_lowercase().str.contains(query.lower()))

def search_index_bm25(indices: List[str], query: str, K: int, query_expansion = "openai", quality_factor = 0.2, expansion_tokens = 20):

    assert query_expansion in {"openai", "keyword", "none"}
    
    tokenizer_vocab = rottnest.get_tokenizer_vocab([f"{index_name}.lava" for index_name in indices])

    if query_expansion == "openai":
        tokens, token_ids, weights = query_expansion_llm(tokenizer_vocab, query, expansion_tokens=expansion_tokens)
    elif query_expansion == "keyword":
        tokens, token_ids, weights = query_expansion_keyword(tokenizer_vocab, query)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file("../tok/tokenizer.json")
        token_ids = tok.encode(query).ids
        tokens = [tokenizer_vocab[i] for i in token_ids]
        weights = [1] * len(token_ids)
        print(tokens)

    # metadata_file = f"{index_name}.meta"
    index_search_results = rottnest.search_lava_bm25([f"{index_name}.lava" for index_name in indices], token_ids, weights, int(K * quality_factor))
    
    if len(index_search_results) == 0:
        return None
    
    uids = polars.from_dict({"file_id": [i[0] for i in index_search_results], "uid": [i[1] for i in index_search_results]})

    metadatas = [polars.read_parquet(f"{index_name}.meta").with_columns(polars.lit(i).alias("file_id").cast(polars.Int64)) for i, index_name in enumerate(indices)]
    metadata = polars.concat(metadatas)
    metadata = metadata.join(uids, on = ["file_id", "uid"])

    print(metadata)
    
    assert len(metadata["column_name"].unique()) == 1, "index is not allowed to span multiple column names"
    column_name = metadata["column_name"].unique()[0]

    result = pyarrow.chunked_array(rottnest.read_indexed_pages(column_name, metadata["file_path"].to_list(), metadata["row_groups"].to_list(),
                                     metadata["data_page_offsets"].to_list(), metadata["data_page_sizes"].to_list(), metadata["dictionary_page_sizes"].to_list()))
    result = pyarrow.table([result], names = ["text"])
    result = result.append_column('row_nr', pyarrow.array(np.arange(len(result)), pyarrow.int64()))

    print(polars.from_arrow(result))

    import duckdb
    con = duckdb.connect()
    con.register('test_table', result)
    con.execute("CREATE TABLE table_copy AS (SELECT * FROM test_table)")

    con.execute("""
    PRAGMA create_fts_index(
        'table_copy', 'row_nr', 'text'
    );
    """)

    result = polars.from_arrow(con.execute(f"""
        SELECT row_nr, text, score
        FROM (
            SELECT *, fts_main_table_copy.match_bm25(
                    row_nr,
                    '{" ".join(tokens)}',
                    fields := 'text'
                ) AS score
                FROM table_copy
        ) sq
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT {K};
        """).arrow())

    return result