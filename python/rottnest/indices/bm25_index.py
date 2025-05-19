from rottnest.indices.index_interface import RottnestIndex, CacheRanges
import rottnest.rottnest as rottnest
import pyarrow
import polars
from typing import List
import numpy as np
import os, pickle
from typing import List
from tqdm import tqdm
import hashlib

def embed_batch_bgem3(tokens: List[str]):
    try:
        from FlagEmbedding import BGEM3FlagModel
    except:
        raise Exception("BGEM3FlagModel requires installation of the FlagEmbedding python package. pip install FlagEmbedding")

    model = BGEM3FlagModel('BAAI/bge-m3')
    
    embeddings = model.encode(tokens, 
                                batch_size=12, 
                                max_length=max([len(i) for i in tokens])
                                )['dense_vecs']
    return embeddings

def embed_batch_openai(tokens: List[str], model = "text-embedding-3-large"):
    try:
        from openai import OpenAI
    except:
        raise Exception("OpenAI python package required. pip install openai")
    client = OpenAI()
    all_vecs = []
    for i in tqdm(range(0, len(tokens), 1000)):
        results = client.embeddings.create(input = tokens[i:i+1000], model = model)
        vecs = np.vstack([results.data[i].embedding for i in range(len(results.data))])
        all_vecs.append(vecs)
    
    return np.vstack(all_vecs)

def query_expansion_llm(tokenizer_vocab: List[str], query: str, method = "bge", expansion_tokens = 20, cache_dir = None):

    assert type(query) == str, "query must be string. If you have a list of keywords, concatenate them with spaces."

    cache_dir = os.path.join(os.path.expanduser('~/.cache'), 'rottnest') if cache_dir is None else cache_dir
    # make a subdirectory rottnest under cache_dir if it's not there already
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    tokenizer_hash = hashlib.sha256(("".join(tokenizer_vocab[::1000])).encode('utf-8')).hexdigest()[:8]

    # check if the tokenizer_embeddings.pkl file exists in the cache directory
    tokenizer_embeddings_path = os.path.join(cache_dir, f"tokenizer_embeddings_{tokenizer_hash}_{method}.pkl")

    if not os.path.exists(tokenizer_embeddings_path):
        print(f"First time doing LLM query expansion with this tokenizer, computing tokenizer embeddings with {method}.")
        tokenizer_vocab = [tok  if tok else "[]" for tok in tokenizer_vocab]
        
        if method == "bge":
            all_vecs = embed_batch_bgem3(tokenizer_vocab)
        elif method == "openai":
            all_vecs = embed_batch_openai(tokenizer_vocab)

        pickle.dump({"words": tokenizer_vocab, "vecs": all_vecs}, 
                    open(os.path.join(cache_dir, f"tokenizer_embeddings_{tokenizer_hash}_{method}.pkl"), "wb"))

    embeddings = pickle.load(open(tokenizer_embeddings_path, "rb"))

    tokens = embeddings['words']
    db_vectors = embeddings['vecs']

    if method == "bge":
        query_vec = embed_batch_bgem3([query])[0]
    elif method == "openai":
        query_vec = embed_batch_openai([query])[0]

    distances = np.dot(db_vectors, query_vec) / np.linalg.norm(db_vectors, axis = 1) / np.linalg.norm(query_vec)
    indices = np.argsort(-distances)[:expansion_tokens]
    print("Expanded tokens: ", [tokens[i] for i in indices])

    return  [tokens[i] for i in indices], list(indices), list(distances[indices])

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


class Bm25Index(RottnestIndex):
    def __init__(self, index_mode: str = 'physical', brute_force_threshold: int = 1000, tokenizer_file: str | None = None):
        super().__init__('bm25', index_mode, 'str', brute_force_threshold)
        self.tokenizer_file = tokenizer_file
        self.tokens = None
    
    def brute_force(self, data: pyarrow.Table, column_name: str, query: str, K: int) -> pyarrow.Table:
        import duckdb
        con = duckdb.connect()
        con.register('test_table', data)
        con.execute("CREATE TABLE table_copy AS (SELECT * FROM test_table)")

        con.execute("""
        PRAGMA create_fts_index(
            'table_copy', 'row_nr', 'text'
        );
        """)

        result = con.execute(f"""
            SELECT row_nr, text, score
            FROM (
                SELECT *, fts_main_table_copy.match_bm25(
                        row_nr,
                        '{" ".join(self.tokens)}',
                        fields := 'text'
                    ) AS score
                    FROM table_copy
            ) sq
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT {K};
            """).arrow()

        # reset the tokens to none after consuming it.
        # this assumes brute force inv. directly follow search_index inv. that sets this.    
        self.tokens = None

        return result

    def build_index(self, data_arr: pyarrow.Array, uid_arr: pyarrow.Array, index_name: str) -> CacheRanges:
        return CacheRanges(rottnest.build_lava_bm25(index_name, data_arr, uid_arr, self.tokenizer_file))

    def search_index(self, indices: List[str], query: str, K: int, query_expansion = "bge", quality_factor = 0.2, expansion_tokens = 20, cache_dir = None) -> List[tuple[int, int]]:
        assert query_expansion in {"bge", "openai", "keyword", "none"}
        
        tokenizer_vocab = rottnest.get_tokenizer_vocab([f"{index_name}.lava" for index_name in indices])

        if query_expansion in {"bge","openai"}:
            tokens, token_ids, weights = query_expansion_llm(tokenizer_vocab, query, method = query_expansion, expansion_tokens=expansion_tokens, cache_dir = cache_dir)
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
        index_search_results = rottnest.search_lava_bm25([f"{index_name}.lava" for index_name in indices], token_ids, weights, int(K * quality_factor), reader_type = reader_type)
        self.tokens = tokens
        return index_search_results
    
    def compact_indices(self, new_index_name: str, indices: List[str], offsets: np.array) -> CacheRanges:
        cache_ranges = rottnest.merge_lava_generic(f"{new_index_name}.lava", [f"{name}.lava" for name in indices], offsets, 0)
        return CacheRanges(cache_ranges)
