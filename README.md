# Rottnest : Data Lake Indices

**Despite our affiliations, this is not an official Anthropic or Bytedance supported project!** Please raise issues here on Github.

You don't need ElasticSearch or some vector database to do full text search or vector search. Parquet + Rottnest is all you need. Rottnest is like Postgres indices for Parquet-based data lakes. Currently it supports an append-only prefix of Parquet files and Iceberg. 

Similar to Iceberg, Rottnest is a [protocol specification](https://www.computer.org/csdl/proceedings-article/icde/2025/360300b814/26FZAoq0tXy). It specifies how you can build indices against object-storage based data lakes like Iceberg. This repo here is an example implmentation of the protocol, together with a few useful indices, such as substring search, BM25 full text search, vector search and log search.

## Installation

Currently, the recommended installation is build from source.
```
maturin develop --release --features py
```

## LogCloud
Rottnest supports the LogCloud index, a tool for compressing and searching log data.
```
maturin develop --release --features "py,logcloud"
```

## How to use

Build indices on your Parquet files, merge them, and query them. Very simple. Let's walk through a very simple example, in `demo.py`. It builds a BM25 index on two Parquet files, merges the indices, and searches the merged index for records related to cell phones. The code is here:

```
from rottnest import iceberg
from rottnest import SubstringIndex

index = SubstringIndex(char_index = True)

config = iceberg.IcebergConfig(
    table = f"rottnest.test_table", 
    column = "text", 
    index_table = f"rottnest.test_index", 
    iceberg_location = 's3://...',
    index_prefix = f"s3://rottnest_data/rottnest_data/", 
)

iceberg.index_iceberg(config = config, index = index)

iceberg.search_iceberg(config = config, query = "We want to welcome you", index = index, K = 10)

```

This code will build a Rottnest substring search index on the `text` column of the Iceberg table at `rottnest.test_table`. It will store index files under `s3://rottnest_data/rottnest_data/`. It will keep its own metadata Iceberg table at `rottnest.test_index`, which will be stored in `iceberg_location`. Once the index is built, the client can be used to query it for different queries. Not this is a pure object-storage based architecture with no caches / longstanding services required (with the exception of the Glue Iceberg catalog).

If you are using S3-compatible file systems, like Ceph, MinIO, Alibaba or Volcano Cloud that might require virtual host style and different endpoint URL, you should set the following environment variables:

```
export AWS_ENDPOINT_URL=https://tos-s3-cn-beijing.volces.com
export AWS_VIRTUAL_HOST_STYLE=true
```

More documentation is forthcoming! Feel free to poke around the codebase and ask it questions in Cursor / Claude Code to learn more!
